# import libraries
import requests
from tqdm import tqdm
from typing import List
import json
import numpy as np
from transformers import MLukeTokenizer, LukeModel
import torch
from sklearn.neighbors import NearestNeighbors
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_narow_data():

    # データ取得
    all_data = []
    loop_cnt = 0

    st_lim_pair = [(1, 499), (500, 500), (1000, 500), (1500, 500), (2000, 500)]
    for st, lim in st_lim_pair:
        print("st:", st)  # st：表示開始位置。max2000
        print("lim", lim)  # lim:最大出力数。max500

        params = {
            "of": "s",
            "order": "hyoka",
            "type": "er",  # 完結済み連載小説に絞る
            "out": "json",
            "lim": lim,
            "st": st,
        }  # nコードとあらすじをjsonで出力

        res = requests.get("https://api.syosetu.com/novelapi/api/", params=params)
        res = res.json()
        new_data = res[1:]

        for data in new_data:
            all_data.append(data["story"])
        print("done")
        print()
    
    return all_data

# sentence-luke
class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, narow_data:List[str], batch_size=8, device=None):
        self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)
        
        #knnをset
        self.knn=self._fit_knn(narow_data, batch_size)
        
        #narow_dataを保持
        self.narow_data=narow_data

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def _fit_knn(self, narow_data:List[str], batch_size=8):
        sentence_embeddings=self.encode(narow_data, batch_size)
        knn = NearestNeighbors(n_neighbors=10, metric="cosine") #距離指標はコサイン類似度
        knn.fit(sentence_embeddings) 
        return knn
        
    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in tqdm(iterator):
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)
    
    def check_novelty(self, sentence: str):
        sentence_embedding=self.encode([sentence]).cpu().detach().numpy().copy()  #sentenceはリスト化して渡す。
        dists, idxs=self.knn.kneighbors(sentence_embedding, n_neighbors=10, return_distance=True)
        
        return np.mean(dists[0]) #distsは(1,近傍数)。そのため、dists[0]をして1次元化してから類似度平均を返す。低い方が新規性がありそう。


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir`."""
    logger.info("start model_fn")
    narow_data=get_narow_data()
    model = SentenceLukeJapanese(model_dir, narow_data, batch_size=8, device=None)
    
    return model

def input_fn(serialized_input_data, content_type):
    """Preprocess input data."""
    logger.info("start input_fn")
    
    if content_type == 'application/json':
        request = json.loads(serialized_input_data)
        return request['text']
    else:
        raise ValueError("The 'content_type' must be 'application/json'.")

def predict_fn(input_data, model):
    """Predicts outputs given the inputs."""
    # Implement your prediction logic and return prediction
    logger.info("start predict_fn")
    prediction = model.check_novelty(input_data)
    prediction={"mean similarity with previous novels": prediction}

    return prediction

def output_fn(prediction, accept='text/plain'):
    logger.info('start output_fn')
    logger.info(f"accept: {accept}")
    if accept == 'text/plain':
        return str(prediction)
    else:
        raise ValueError("The 'accept' must be 'text/plain'.")
