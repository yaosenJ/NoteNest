# coding: utf-8
import os
from typing import List
import difflib
import re
import json
import Levenshtein
import cn2an
from decimal import Decimal, getcontext
from collections import Counter
import numpy as np
import pandas as pd
import dashscope
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma

# 常量定义
STANDARD_DATA = ["白酒", "白酒行业", "房地产", "房地产开发"]
DASHSCOPE_API_KEY = ""


class TextSimilarity:
    """文本相似度计算工具类"""

    @staticmethod
    def find_neighbours(input_data, reference_data, method=None):
        """查找最相似的参考数据"""
        if method == 'edit':
            distances = [Levenshtein.distance(input_data, q) / len(q) for q in reference_data]
            idx = np.argsort(distances)
        else:
            similarity_scores = cosine_similarity([input_data], reference_data)
            idx = np.argsort(similarity_scores[0])[::-1]
        return idx[0]

    @classmethod
    def use_model(cls, method, input_data, reference_data):
        """使用指定模型计算相似度"""
        model_map = {
            'sim': 'IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese',
            'm3e': 'moka-ai/m3e-large',
            'e5': 'intfloat/multilingual-e5-base'
        }

        model = SentenceTransformer(model_map[method])
        standard_data_emb = model.encode(reference_data)
        input_data_emb = model.encode(input_data)
        return cls.find_neighbours(input_data_emb, standard_data_emb, method=method)


class DashscopeEmbeddings(Embeddings):
    """Dashscope 嵌入模型封装"""

    def __init__(self, model: str = 'text-embedding-v1'):
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档嵌入"""
        embeddings = []
        for text in texts:
            response = dashscope.TextEmbedding.call(
                model=self.model,
                input=text
            )
            if response.status_code != 200:
                raise ValueError(f"Embedding request failed: {response.message}")
            embedding = response.output['embeddings'][0]['embedding']
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """生成查询嵌入"""
        response = dashscope.TextEmbedding.call(
            model=self.model,
            input=text
        )
        if response.status_code != 200:
            raise ValueError(f"Embedding request failed: {response.message}")
        return response.output['embeddings'][0]['embedding']


def load_and_split_documents(file_path: str, encoding: str = "gbk") -> List[Document]:
    """加载并分割文档"""
    loader = TextLoader(file_path, encoding=encoding)
    documents = loader.load()
    text = documents[0].page_content
    lines = text.split('\n')
    return [Document(page_content=line.strip()) for line in lines if line.strip()]


def create_vector_store(documents: List[Document], persist_dir: str = "./chroma_db"):
    """创建并持久化向量存储"""
    embedding_model = DashscopeEmbeddings()
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir
    )

def vector_search(query: str, file_path: str = "./data.txt", top_k: int = 1) -> List[Document]:
    """
    使用向量数据库检索最相似的文本块

    Args:
        query (str): 查询文本
        file_path (str): 文本文件路径，默认为"./data.txt"
        top_k (int): 返回最相似的前k个结果，默认为1

    Returns:
        List[Document]: 最相似的文本块列表
    """
    # 加载并分割文档
    documents = load_and_split_documents(file_path)

    # 创建/加载向量数据库
    db = create_vector_store(documents)

    # 执行相似度搜索
    docs = db.similarity_search(query, k=top_k)

    return docs[0].page_content


def main():
    # print(STANDARD_DATA[TextSimilarity.find_neighbours("白酒行业", STANDARD_DATA, method='edit')])
    # print(STANDARD_DATA[TextSimilarity.use_model('sim', "白酒行业", STANDARD_DATA)])
    # print(STANDARD_DATA[TextSimilarity.use_model('m3e', "白酒行业", STANDARD_DATA)])
    # print(STANDARD_DATA[TextSimilarity.use_model('e5', "白酒行业", STANDARD_DATA)])
    query = "白酒行业"
    results = vector_search(query)
    print(results)

if __name__ == "__main__":
    main()