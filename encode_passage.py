"""
 Module that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""

import os
import pathlib
import gc
import argparse
import csv
import logging
import pickle
from typing import List, Tuple, Dict, Iterator
import pandas as pd
import numpy as np
import torch
from torch import nn
import json
from time import time
from app.passage_retrieval.DPR.dpr.models import init_biencoder_components
from app.passage_retrieval.DPR.dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    print_args,
    set_encoder_params_from_state,
    add_tokenizer_params,
    add_cuda_params,
)
from app.passage_retrieval.DPR.dpr.utils.data_utils import Tensorizer
from app.passage_retrieval.DPR.dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)
from tqdm import tqdm

import glob
import json
import gzip
import time
from typing import List, Tuple, Dict, Iterator
from torch import Tensor as T

from app.passage_retrieval.DPR.dpr.data.qa_validation import calculate_matches
from app.passage_retrieval.DPR.dpr.indexer.faiss_indexers import (
    DenseIndexer,
    DenseHNSWFlatIndexer,
    DenseFlatIndexer,
)
from collections import defaultdict, OrderedDict
import requests
from app.passage_retrieval.DPR.config import PRConfig

logger = logging.getLogger("mserve")

def gen_ctx_vectors(
    ctx_rows: List[Tuple[object, str, str]],
    model: nn.Module,
    tensorizer: Tensorizer,
    insert_title: bool = True,
) -> List[Tuple[object, np.array]]:
    """[summary]

    Parameters
    ----------
    ctx_rows : List[Tuple[object, str, str]]
        [description]
    model : nn.Module
        [description]
    tensorizer : Tensorizer
        [description]
    insert_title : bool, optional
        [description], by default True

    Returns
    -------
    List[Tuple[object, np.array]]
        [description]
    """
    n = len(ctx_rows)
    bsz = args.batch_size
    total = 0
    results = []
    for j, batch_start in enumerate(range(0, n, bsz)):

        batch_token_tensors = [
            tensorizer.text_to_tensor(ctx[1], title=ctx[2] if insert_title else None)
            for ctx in ctx_rows[batch_start : batch_start + bsz]
        ]

        ctx_ids_batch = move_to_device(
            torch.stack(batch_token_tensors, dim=0), args.device
        )
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), args.device)
        ctx_attn_mask = move_to_device(
            tensorizer.get_attn_mask(ctx_ids_batch), args.device
        )
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()

        ctx_ids = [r[0] for r in ctx_rows[batch_start : batch_start + bsz]]

        assert len(ctx_ids) == out.size(0)

        total += len(ctx_ids)

        results.extend(
            [(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))]
        )

    return results


saved_state = load_states_from_checkpoint(os.path.join(PRConfig.MODEL_DIR,PRConfig.ALL_MODELS[0]))

parser = argparse.ArgumentParser()

add_encoder_params(parser)
add_tokenizer_params(parser)
add_cuda_params(parser)

parser.add_argument(
    "--ctx_file", type=str, default=f'{PRConfig.DATA_DIR}/all_ctx.tsv', help="Path to passages set .tsv file"
)
parser.add_argument(
    "--out_file",
    required=False,
    type=str,
    default=f'{PRConfig.DATA_DIR}/all_emb_ctx',
    help="output .tsv file path to write results to ",
)
parser.add_argument(
    "--shard_id",
    type=int,
    default=0,
    help="Number(0-based) of data shard to process",
)
parser.add_argument(
    "--num_shards", type=int, default=1, help="Total amount of data shards"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for the passage encoder forward pass",
)

args, unknown = parser.parse_known_args()
args.device = PRConfig.DEVICE
args.no_cuda = True if PRConfig.DEVICE == 'cpu' else False
setup_args_gpu(args)
set_encoder_params_from_state(saved_state.encoder_params, args)
args.batch_size = PRConfig.BATCH_SIZE

tensorizer, encoder, _ = init_biencoder_components(
    args.encoder_model_type, args, inference_only=True
)

context_encoder = encoder.ctx_model

context_encoder, _ = setup_for_distributed_mode(
    context_encoder,
    None,
    args.device,
    args.n_gpu,
    args.local_rank,
    args.fp16,
    args.fp16_opt_level,
)
encoder.eval()

# load weights from the model file
model_to_load = get_model_obj(context_encoder)
prefix_len = len("ctx_model.")
ctx_state = {
    key[prefix_len:]: value
    for (key, value) in saved_state.model_dict.items()
    if key.startswith("ctx_model.")
    }
model_to_load.load_state_dict(ctx_state)

parser = argparse.ArgumentParser()

add_encoder_params(parser)
add_tokenizer_params(parser)
add_cuda_params(parser)


parser.add_argument(
    "--qa_file",
    required=False,
    type=str,
    default=f"{PRConfig.DATA_DIR}/embibe/val.json",
    help="Question and answers file of the format: question \\t ['answer1','answer2', ...]",
)
parser.add_argument(
    "--ctx_file",
    required=False,
    type=str,
    default=f"{PRConfig.DATA_DIR}all_ctx.tsv",
    help="All passages file in the tsv format: id \\t passage_text \\t title",
)
parser.add_argument(
    "--encoded_ctx_file",
    type=str,
    default=f"{PRConfig.DATA_DIR}/all_emb_ctx_5_0.pkl",
    help="Glob path to encoded passages (from generate_dense_embeddings tool)",
)
parser.add_argument(
    "--out_file",
    type=str,
    default=f"{PRConfig.DATA_DIR}/embibe/val_processed",
    help="output .tsv file path to write results to ",
)
parser.add_argument(
    "--match",
    type=str,
    default="string",
    choices=["regex", "string"],
    help="Answer matching logic type",
)
parser.add_argument(
    "--n-docs", type=int, default=100, help="Amount of top docs to return"
)
parser.add_argument(
    "--validation_workers",
    type=int,
    default=16,
    help="Number of parallel processes to validate results",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for question encoder forward pass",
)
parser.add_argument(
    "--index_buffer",
    type=int,
    default=50000,
    help="Temporal memory data buffer size (in samples) for indexer",
)
parser.add_argument(
    "--hnsw_index",
    action="store_true",
    help="If enabled, use inference time efficient HNSW index",
)
parser.add_argument(
    "--save_or_load_index",default=True, action="store_true", help="If enabled, save index"
)

args, unknown = parser.parse_known_args()
args.model_file = os.path.join(PRConfig.MODEL_DIR,PRConfig.ALL_MODELS[0])
args.device = PRConfig.DEVICE
args.no_cuda = True if PRConfig.DEVICE == 'cpu' else False
setup_args_gpu(args)
args.n_docs = 5


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
    ):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [
                    self.tensorizer.text_to_tensor(q)
                    for q in questions[batch_start : batch_start + bsz]
                ]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).to(args.device)
                q_seg_batch = torch.zeros_like(q_ids_batch).to(args.device)
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

        query_tensor = torch.cat(query_vectors, dim=0)

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        results = self.index.search_knn(query_vectors, top_docs)
        return results


def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        samples = json.loads("".join(ifile.readlines()))
        for sample in samples:
            yield sample['question'], sample['answers']


def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    if ctx_file.endswith(".gz"):
        with gzip.open(ctx_file, "rt") as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file,encoding="utf-8") as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    return docs


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector

saved_state = load_states_from_checkpoint(args.model_file)
set_encoder_params_from_state(saved_state.encoder_params, args)

tensorizer, encoder, _ = init_biencoder_components(
    args.encoder_model_type, args, inference_only=True
)
question_encoder = encoder.question_model

question_encoder, _ = setup_for_distributed_mode(
    question_encoder, None, args.device, args.n_gpu, args.local_rank, args.fp16
)
question_encoder.eval()
model_to_load = get_model_obj(question_encoder)
prefix_len = len("question_model.")
question_encoder_state = {
    key[prefix_len:]: value
    for (key, value) in saved_state.model_dict.items()
    if key.startswith("question_model.")
    }
model_to_load.load_state_dict(question_encoder_state)
vector_size = model_to_load.get_out_size()

def encode_passages_oc(all_passages):
    to_encode = [(str(i), j, 'title') for i,j in enumerate(all_passages)]
    all_passages = {str(i): (j,'title') for i,j in enumerate(all_passages)}
    data = gen_ctx_vectors(to_encode, context_encoder, tensorizer, True)
    return data, all_passages

index_ob = DenseFlatIndexer(vector_size, args.index_buffer)
index_ob.deserialize_from(f'{PRConfig.DATA_DIR}emb')
retriever_ob = DenseRetriever(question_encoder, args.batch_size, tensorizer, index_ob)
all_passages_ob = load_passages(args.ctx_file)

def get_chapter_texts(uid):
    url = f"http://10.141.11.220/api/v1/bs-filters/?entity_type=chapter_text&text_id={uid}"
    response = requests.request("GET", url)
    return response.json()['result']

indices_dict_oc = defaultdict()
def initialize_index(uid):
    if indices_dict_oc.get(uid) is None:
        index_oc = DenseFlatIndexer(vector_size, args.index_buffer)
        all_passages = get_chapter_texts(uid)
        all_passages = all_passages.split('\n\n')
        to_encode = [(str(i), j, 'title') for i,j in enumerate(all_passages)]
        all_passages = {str(i): (j,'title') for i,j in enumerate(all_passages)}
        data = gen_ctx_vectors(to_encode, context_encoder, tensorizer, True)
        index_oc._index_batch(data)
        indices_dict_oc[uid] = [index_oc, all_passages]
        return indices_dict_oc.get(uid)
    else:
        return indices_dict_oc.get(uid)

def get_passages(questions, top_k = 2, mode='openbook', uid=None):
    if mode == 'openbook':
        questions_tensor = retriever_ob.generate_question_vectors(questions)
        top_ids_and_scores = retriever_ob.get_top_docs(questions_tensor.numpy(), top_k)
        passages = []
        for i in top_ids_and_scores:
            passages.append([all_passages_ob[i[0][j]][0] for j in range(top_k)])
        return passages
    elif mode == 'openchapter':
        if uid is None:
            return "Provide Unique ID for the chapter"
        index_oc, all_passages_oc = initialize_index(uid)
        retriever_oc = DenseRetriever(question_encoder, args.batch_size, tensorizer, index_oc)
        questions_tensor = retriever_oc.generate_question_vectors(questions)
        top_ids_and_scores = retriever_oc.get_top_docs(questions_tensor.numpy(), top_k)
        passages = []
        for i in top_ids_and_scores:
            passages.append([all_passages_oc[i[0][j]][0] for j in range(top_k)])
        return passages
    else:
        return "Requested mode is not Supported."

# def get_answer_deprecated(question, passages, payload):
#     url = "http://10.145.0.15:9000/api/v1/task-question-answer"
#     headers = {'Content-Type': 'application/json'}
#     callbacks = []
#     for i, j in enumerate(passages):
#         payload['sample'] = [j, question]
#         payload["sample_id"] =  i
#         response = requests.request("POST", url, headers=headers, json=payload)
#         if response.status_code != 202:
#             logger.error("error QA MODEL: %s", str(response.reason), exc_info=True)
#             # print("QA MODEL:", response.reason)
#             return None
#         callbacks.append(response.json()['data']['callback_url'])
#     is_completed = [False]*len(callbacks)
#     pred_answers = []
#     while not all(is_completed):
#         for i,j in enumerate(callbacks):
#             url = f"http://10.145.0.15:9000/{j}"
#             if not is_completed[i]:
#                 response = requests.request("GET", url)
#                 if response.status_code != 200:
#                     logger.error("error TASK TRACKER: %s", str(response.reason), exc_info=True)
#                     # print("TASK TRACKER:", response.reason)
#                     return None
#                 response = response.json()
#                 if response['task_progress'] == 100:
#                     pred_answers.append(response['task_result']['data']['prediction'])
#                     is_completed[i] = True
#                 else:
#                     continue
#             else:
#                 continue
#         time.sleep(2)
#     pred_answers = [i for i in pred_answers if i is not None]
#     pred_answers = [0] if len(pred_answers) > 0 else None
#     return pred_answers