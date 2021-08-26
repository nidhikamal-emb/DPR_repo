import os
from pathlib import Path

import torch


class PRConfig:
    # STATIC_DATA_ROOT = "/home/ubuntu/most_similar_questions/darshan/"
    STATIC_DATA_ROOT = os.getenv("DPR_MODELS_PATH")  # path for the model files
    DATA_DIR = str(Path(STATIC_DATA_ROOT, "dataset_dir"))
    MODEL_DIR = str(Path(STATIC_DATA_ROOT, "model_dir"))
    ALL_MODELS = [
        "dpr_biencoder.4.13636",
    ]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 30
    OPENBOOK_CONTEXT = str(Path(DATA_DIR, "all_emb_ctx_5"))
    QUESTION_EMB_FAISS = str(Path(DATA_DIR, "cg_question_index_norm_native.faiss"))
    QUESTION_EMB_ID_MAPPING = str(Path(DATA_DIR, "cg_question_index_norm_native_id_mapping.pkl"))
    IMAGE_INDEX = str(Path(DATA_DIR, "image_index.faiss"))
    IMG_NAMES = str(Path(DATA_DIR, "data_img_name_efnet_vanilla.pt"))
    QUESTION_ID_TO_CODE = str(Path(DATA_DIR, "question_id_to_code_mapping.pkl"))
    IMG_SIZE = 256
