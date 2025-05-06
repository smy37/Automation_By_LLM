from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import os
import pickle
from common.openai_api import embedding_openAI_batch
from common.utils import doc_split
from common import variable
from tqdm import tqdm
import sys

b_save_bm25_index = False
b_save_vector = False
VECTOR_DIMENSION = 3072
KEYWORD_RATIO = 0.3
read_path = "./result/concat"
write_path = "./artifact"

### 1. Make Data for Keyword Search
if b_save_bm25_index:
    tokenized_corpus = [open(os.path.join(read_path, fn), 'r', encoding="utf-8-sig").read().split() for fn in os.listdir(read_path)]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(f"{write_path}/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(f"{write_path}/bm25_title_index.txt", 'w', encoding="utf-8") as wr:
        temp = "\n".join(os.listdir(read_path))
        wr.write(temp)

### 2. Make Data for Vector Search
if b_save_vector:
    vector_candi = [fn for fn in os.listdir(read_path)]
    vector_title = []
    vector_list = []
    vector_text = []
    for i, fn in tqdm(enumerate(vector_candi)):
        text = open(f"{read_path}/{fn}", 'r', encoding='utf-8-sig').read()
        chunks = doc_split(text, variable.EMBEDDING_MODEL, 7500)
        temp_vector = embedding_openAI_batch(chunks, variable.EMBEDDING_MODEL, VECTOR_DIMENSION)

        for j, c in enumerate(chunks):
            vector_title.append(f"{fn}_{j + 1}")
            vector_text.append(c)
        for v in temp_vector:
            vector_list.append(v.embedding)
    index = faiss.IndexFlatIP(VECTOR_DIMENSION)
    vector_array = np.array(vector_list).astype('float32')
    index.add(vector_array)
    faiss.write_index(index, f"{write_path}/vector_index.bin")
    index = faiss.read_index(f"{write_path}/vector_index.bin")

    with open(f"{write_path}/vector_title_index.txt", 'w', encoding="utf-8") as wr:
        temp = "\n".join(vector_title)
        wr.write(temp)


def normalize(scores):
    max_v, min_v = max(scores), min(scores)
    normalize_s = [(s-min_v)/(max_v-min_v+1e-6) for s in scores]
    return normalize_s


def get_bm25_score(query:str):
    result = {}
    with open(f"{write_path}/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(scores)[::-1][:100]

    bm25_index = os.listdir(read_path)
    normalize_score = normalize([scores[idx] for idx in top_bm25_indices])

    for i, idx in enumerate(top_bm25_indices):
        id = bm25_index[idx]
        result[id] = normalize_score[i]
    return result

def get_vector_score(query:str):
    result = {}
    index = faiss.read_index(f"{write_path}/vector_index.bin")
    query_vector = np.array([embedding_openAI_batch([query], variable.EMBEDDING_MODEL, VECTOR_DIMENSION)[0].embedding]).astype('float32')
    distances, indices = index.search(query_vector, 100)
    indices = indices.tolist()[0]
    distances = distances.tolist()[0]

    vector_index = open(f"{write_path}/vector_title_index.txt", 'r', encoding='utf-8').readlines()
    normalize_score = normalize(distances)

    for i, idx in enumerate(indices):
        id = "_".join(vector_index[idx].strip().split("_")[:-1])
        if id not in result:
            result[id] = normalize_score[i]
        else:
            if result[id] < normalize_score[i]:
                result[id] = normalize_score[i]
    return result


def cal_hybrid_score(query: str):
    result_keyword = get_bm25_score(query)
    result_vector = get_vector_score(query)

    hybrid_result = {}

    for id in result_keyword:
        hybrid_result[id] = KEYWORD_RATIO*result_keyword[id]

    for id in result_vector:
        if id in hybrid_result:
            hybrid_result[id] += (1-KEYWORD_RATIO)*result_vector[id]
        else:
            hybrid_result[id] = (1-KEYWORD_RATIO)*result_vector[id]

    return hybrid_result

print(cal_hybrid_score("Airflow에서 postgresql을 연결해서 사용하는 방법을 상세히 알려줄래?"))