"""
 Module that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""

import argparse
import io
import gc
import html
import logging
import os
import pickle
import re
import unicodedata as uni
from time import time
from typing import List

import numpy as np
import pandas as pd
import requests
from PIL import Image

import elasticsearch
import faiss
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as TRANSFORM
from app.connectors.redis_connector import ImageStoreRedisApp
from app.passage_retrieval.DPR.config import PRConfig
from app.passage_retrieval.DPR.dpr.models import init_biencoder_components
from app.passage_retrieval.DPR.dpr.options import (
    add_cuda_params,
    add_encoder_params,
    add_tokenizer_params,
    set_encoder_params_from_state,
    setup_args_gpu,
)
from app.passage_retrieval.DPR.dpr.utils.data_utils import Tensorizer
from app.passage_retrieval.DPR.dpr.utils.model_utils import (
    get_model_obj,
    load_states_from_checkpoint,
    setup_for_distributed_mode,
)
from app.utils.common import is_url
from app.utils.custom_exeptions import ImageRequestError
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
from torch import Tensor as T
from torch import nn as nn

es_logger = elasticsearch.logger
es_logger.setLevel(elasticsearch.logging.WARNING)
es_client = Elasticsearch(
    ["10.141.11.88", "10.141.11.89", "10.141.11.90"],
    timeout=30,
    max_retries=5,
    retry_on_timeout=False,
)
es_index_name = "cg-doubt_solving-question_embeddings"


logger = logging.getLogger("most-similar-questions-dpr")


def get_cleantext(string):
    try:
        string = re.sub("<[^>]*>", " ", string).strip()
        string = html.unescape(string)
        string = " $$ ".join(string.split("$$"))
        string = re.sub(
            "(https?:\/\/.*)|(www.*?\s)|(en.wikipedia.*?\s)|(visit:)", " ", string
        )
        string = re.sub("\s+", " ", uni.normalize("NFKD", string))
    except Exception as error:
        logger.exception("Error in text cleaing %s", str(error))
    return string


saved_state = load_states_from_checkpoint(
    os.path.join(PRConfig.MODEL_DIR, PRConfig.ALL_MODELS[0])
)

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
    "--save_or_load_index",
    default=True,
    action="store_true",
    help="If enabled, save index",
)

args, unknown = parser.parse_known_args()
args.model_file = os.path.join(PRConfig.MODEL_DIR, PRConfig.ALL_MODELS[0])
args.device = PRConfig.DEVICE
args.no_cuda = True if PRConfig.DEVICE == "cpu" else False
setup_args_gpu(args)
args.n_docs = 5
device = args.device


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index,
    ):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

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

                query_vectors.extend(out.detach().cpu().split(1, dim=0))
                del out
                del q_ids_batch
                del q_seg_batch
                del q_attn_mask

        query_tensor = np.concatenate(query_vectors, axis=0)

        assert query_tensor.shape[0] == len(questions)
        del query_vectors
        del questions
        torch.cuda.empty_cache()
        gc.collect()

        return query_tensor


saved_state = load_states_from_checkpoint(
    os.path.join(PRConfig.MODEL_DIR, PRConfig.ALL_MODELS[0])
)
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
    and key != "question_model.embeddings.position_ids"
}
model_to_load.load_state_dict(question_encoder_state)
vector_size = model_to_load.get_out_size()
del question_encoder_state
gc.collect()
retriever = DenseRetriever(question_encoder, args.batch_size, tensorizer, index=None)

faiss_index = faiss.read_index(PRConfig.QUESTION_EMB_FAISS)
txt_index_id_to_question_id = pickle.load(open(PRConfig.QUESTION_EMB_ID_MAPPING, "rb"))
faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)

image_index = faiss.read_index(PRConfig.IMAGE_INDEX)
image_index = faiss.index_cpu_to_all_gpus(image_index)
img_names = torch.load(PRConfig.IMG_NAMES)

# question_id_to_code = pickle.load(open(PRConfig.QUESTION_ID_TO_CODE, "rb"))
# question_code_to_id = {question_id_to_code[i]: i for i in question_id_to_code}
# img_names_to_q_id = {
#    j: question_code_to_id[i.split("_")[0]] for j, i in enumerate(img_names)
# }

img_names_to_q_id = {j: int(i.split("_")[0]) for j, i in enumerate(img_names)}


def separate_main_text_v2(question_text):
    list1 = ["(1)", "(i)", "(I)", "i)", "1)", "(a)", "a)", "(A)", "A)", "A."]

    pattern1 = ['(1)',"(2)","(3)","(4)","(5)","(6)","(7)", "(8)", "(9)", "(10)", "(11)", "(12)", "(13)", "(14)", "(15)"]
    pattern2 = ['(i)',"(ii)","(iii)","(iv)","(v)","(vi)","(vii)", "(viii)", "(ix)", "(x)", "(xi)", "(xii)", "(xiii)","(xiv)", "(xv)"]
    pattern3 = ['(I)',"(II)","(III)","(IV)","(V)","(VI)","(VII)", "(VIII)", "(IX)", "(X)", "(XII)" , "(XII)", "(XIII)", "(XIV)", "(XV)"]
    pattern4 = ['i)',"ii)","iii)","iv)","v)","vi)","vii)", "viii)", "ix)", "x)","xi)", "xii)", "xiii)", "xiv)", "xv)"]
    pattern5 = ['1)',"2)","3)","4)","5)","6)","7)", "8)", "9)", "10)", "11)", "12)", "13)", "14)","15)"]
    pattern6 = ['(a)',"(b)","(c)","(d)","(e)","(f)","(g)", "(h)", "(i)", "(j)", "(k)", "(l)", "(m)", "(n)", "(o)"]
    pattern7 = ['a)',"b)","c)","d)","e)","f)","g)", "h)", "i)", "j)", "k)", "l)", "m)", "n)", "o)"]
    pattern8 = ["(A)", "(B)","(C)","(D)","(E)","(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)", "(M)", "(N)", "(O)"]
    pattern9 = ["A)", "B)","C)","D)","E)","F)", "G)", "H)", "I)", "J)", "K)", "L)", "M)", "N)", "O)"]
    pattern10 = ["A.", "B.","C.","D.","E.","F.", "G.", "H.", "I.", "J.", "K.", "L.", "M.", "N.", "O."]
    pattern = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8, pattern9, pattern10]

    separated_text = []
    start = 0
    end = 0

    try:
        question_text = question_text.replace("\n"," ")
        def get_type(text, list1):
            pattern_ind = 99999
            pattern_type = -1
            for i,v in enumerate(list1):
                if(re.search(re.escape(v),text)): 
                    #if the given numeral occurs before the other numeral type(to find which occurs first) 
                        temp_ind = text.find(v)
                        if(temp_ind<pattern_ind):
                            pattern_ind = temp_ind
                            pattern_type=i
            return pattern_type

        pattern_type = get_type(question_text, list1)
        print("####pattern type####:"+str(pattern_type))
        print(pattern[pattern_type])
        for i,v in enumerate(pattern[pattern_type]):
            if(i==0):
                res = (question_text.find(v))
                    
                end = res
                separated_text.append(question_text[start:end])
                start = res
            else:
                if(re.search(re.escape(v), question_text) and re.search(re.escape(pattern[pattern_type][i-1]), question_text)):
                    #this can be confused in cases of functions of (v) and (x) with roman 5 and 10, hence above condition
                    res = (question_text.find(v))
                    end = res
                    separated_text.append(question_text[start:end])
                    start = res
        separated_text.append(question_text[start:])    
        separated_text = [x for x in separated_text if(len(x)>1)]
        #if there is no main text before the sub-questions, pass separated_text as is
        if(re.search(re.escape(pattern[pattern_type][0]), separated_text[0])):
            return separated_text
        #if there is main text before the sub-questions, append main_text before each of the sub_questions
        else:
            mod_separated_text = [separated_text[0]+sub_qn for sub_qn in separated_text[1:]]
            #return ONLY the first sub_question for v1
            return mod_separated_text[0]

    except Exception as e:
        return question_text


def encode_questions(questions: list):

    urls = []
    for i in questions:
        soup = BeautifulSoup(i, "html.parser")
        imgs = soup.find_all("img")
        for img in imgs:
            urls.append(img.get("src"))

    questions = [get_cleantext(i) for i in questions]
    questions = [separate_main_text_v2(i) for i in questions]
    text_len = len(questions[0].split(" "))
    questions_tensor = retriever.generate_question_vectors(questions)
    return questions_tensor, urls, text_len


def get_similar_questions_text(question_tensor, top_k=5, index="faiss"):

    try:
        if index == "es":
            body = {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'dense_vector_dpr') + 1.0",
                            "params": {"query_vector": question_tensor.tolist()[0]},
                        },
                    }
                },
            }
            res = es_client.search(index=es_index_name, body=body)
            question_ids = [
                res["hits"]["hits"][i]["_source"]["question_id"] for i in range(top_k)
            ]
            scores = [res["hits"]["hits"][i]["_score"] for i in range(top_k)]
            return question_ids, scores
        elif index == "faiss":

            faiss.normalize_L2(question_tensor)
            results = faiss_index.search(question_tensor, top_k)
            question_ids = results[1].tolist()[0]
            scores = results[0].tolist()[0]
            question_ids = [txt_index_id_to_question_id[i] for i in question_ids]
            return question_ids, scores
        else:
            return None, None
    except Exception as error:
        logger.exception(str(error))
        return None, None


transforms = TRANSFORM.Compose(
    [TRANSFORM.Resize((PRConfig.IMG_SIZE, PRConfig.IMG_SIZE)), TRANSFORM.ToTensor()]
)

efnet_model = timm.create_model("efficientnet_b0", pretrained=True).as_sequential()[:-2]
efnet_model = efnet_model.eval().to(device)


def get_image_data(image_source: str):
    """Get the image data from give image_source

    Parameters
    ----------
    image_source : str
        [can be a image_fiel URL, or else a redis-key]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ImageRequestError
        [description]
    """
    image_data = None
    if not is_url(image_source):
        redis_client = ImageStoreRedisApp.get_client()
        image_data = redis_client.get(image_source)  # image_data in bytes
        # since image is saved in bytes, to process the image in PIL
        # we need to convert it as file like object
        image_data = io.BytesIO(image_data)
    else:
        image_data = requests.get(image_source, stream=True).raw
    if not image_data:
        raise ImageRequestError("Image not found")
    return image_data


def load_image_tensor(image_path: str, device: str):
    image_tensor = transforms(Image.open(image_path).convert("RGB"))
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return image_tensor


def compute_similar_images(image_paths: list, num_images: int, index: str = "faiss"):
    tensor_list = []
    torch.cuda.empty_cache()
    gc.collect()

    for i in image_paths:
        image_tensor = transforms(Image.open(get_image_data(i)).convert("RGB"))
        image_tensor = image_tensor.unsqueeze(0).to(device)
    tensor_list.append(image_tensor)
    tensor_list = torch.cat(tensor_list)

    with torch.no_grad():
        image_embedding = (
            torch.mean(efnet_model(tensor_list).cpu(), axis=0)
            .detach()
            .numpy()
            .reshape(1, -1)
        )

    del image_tensor
    del tensor_list
    torch.cuda.empty_cache()
    gc.collect()

    faiss.normalize_L2(image_embedding)
    dist, indices = image_index.search(image_embedding, num_images)
    return indices.tolist()[0], dist.tolist()[0]


def get_final_score(text_score: float, img_score: float, text_len: int) -> float:
    if text_len >= 25:
        return 0.7 * text_score + 0.3 * img_score
    else:
        return 0.5 * text_score + 0.5 * img_score


def get_similar_questions(text, text_len, image, k, index="faiss"):
    score = None
    indices_list = None
    if len(image) == 0:
        image = None
    if image is not None and text is not None:
        index = "faiss"
        indices_list_img, dist_img = compute_similar_images(image, k, index)
        indices_list_text, dist_text = get_similar_questions_text(text, k, index)
        img_df = pd.DataFrame({"img_indices": indices_list_img, "img_score": dist_img})
        img_df["question_id"] = img_df.img_indices.map(img_names_to_q_id)
        img_df = (
            img_df.groupby("question_id")
            .agg({"img_indices": lambda x: list(x)[0], "img_score": "mean"})
            .reset_index()
        )

        text_df = pd.DataFrame(
            {"question_id": indices_list_text, "txt_score": dist_text}
        )
        merge_df = text_df.merge(img_df, on="question_id", how="outer")
        merge_df.txt_score = merge_df.txt_score.fillna(
            min(merge_df.txt_score.dropna().tolist()) * 0.8
        )
        merge_df.img_score = merge_df.img_score.fillna(
            min(merge_df.img_score.dropna().tolist()) * 0.8
        )

        merge_df["final_score"] = merge_df.apply(
            lambda x: get_final_score(x["txt_score"], x["img_score"], text_len), axis=1
        )
        merge_df = merge_df.sort_values(by="final_score", ascending=False)
        merge_df = merge_df.drop_duplicates(subset=["question_id"]).reset_index(
            drop=True
        )
        merge_df = merge_df.head(k)
        indices_list = merge_df.question_id.tolist()
        score = merge_df.final_score.tolist()

    elif image is None and text is not None:
        indices_list, score = get_similar_questions_text(text, k, index)
    elif text is None and image is not None:
        index = "faiss"
        indices_list, score = compute_similar_images(image, k, index)
        q_codes = [img_names[x].split("_")[0] for x in indices_list]
        indices_list = q_codes

    response = []
    for i, j in zip(indices_list, score):
        response.append([j, i])
    return response
