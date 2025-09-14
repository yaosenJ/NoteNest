# coding:utf-8
import os
import json
import hashlib
from typing import List, Dict, Set, Optional
import warnings

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 忽略警告
warnings.filterwarnings('ignore')


class RAGSystem:
    """RAG系统类，封装整个检索增强生成流程"""

    def __init__(self, api_key: str,
                 embedding_model: str = r"C:\Users\jys\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620"):
        """
        初始化RAG系统

        Args:
            api_key: DashScope API密钥
            embedding_model: 嵌入模型名称
        """
        self.api_key = api_key
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_db = None
        self.mapping_dict = {}
        self.all_big_chunks = []
        self.all_small_chunks = []

    def load_and_preprocess_documents(self, document_text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """
        加载和预处理文档

        Args:
            document_text: 文档文本内容
            metadata: 文档元数据

        Returns:
            文档对象列表
        """
        if metadata is None:
            metadata = {"source": "unknown"}

        return [Document(page_content=document_text, metadata=metadata)]

    def split_documents(self, documents: List[Document],
                        big_chunk_size: int = 300, big_chunk_overlap: int = 50,
                        small_chunk_size: int = 100, small_chunk_overlap: int = 20) -> None:
        """
        分割文档并建立映射关系

        Args:
            documents: 文档列表
            big_chunk_size: 大块大小
            big_chunk_overlap: 大块重叠大小
            small_chunk_size: 小块大小
            small_chunk_overlap: 小块重叠大小
        """
        # 创建文本分割器
        big_splitter = RecursiveCharacterTextSplitter(
            chunk_size=big_chunk_size,
            chunk_overlap=big_chunk_overlap,
        )

        small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=small_chunk_size,
            chunk_overlap=small_chunk_overlap,
        )

        # 清空之前的数据
        self.all_small_chunks.clear()
        self.all_big_chunks.clear()
        self.mapping_dict.clear()

        # 首先，将文档切分成"大"块
        big_chunks = big_splitter.split_documents(documents)

        for big_chunk_index, big_chunk in enumerate(big_chunks):
            # 将每个"大"块进一步切分成"小"块
            small_chunks_from_big = small_splitter.split_documents([big_chunk])

            # 为每个"小"块创建唯一ID并存储映射关系
            for small_chunk in small_chunks_from_big:
                # 使用更稳定的哈希方法
                small_chunk_id = self._generate_chunk_id(small_chunk.page_content)
                self.mapping_dict[small_chunk_id] = {
                    "big_chunk_content": big_chunk.page_content,
                    "big_chunk_index": big_chunk_index
                }
                self.all_small_chunks.append(small_chunk)

            self.all_big_chunks.append(big_chunk)

        print(f"切分出 {len(self.all_big_chunks)} 个大块")
        print(f"切分出 {len(self.all_small_chunks)} 个小块")

    def _generate_chunk_id(self, content: str) -> str:
        """生成内容ID"""
        return hashlib.md5(content.encode()).hexdigest()

    def build_vector_index(self) -> None:
        """构建向量索引"""
        if not self.all_small_chunks:
            raise ValueError("没有可用的文档块，请先调用 split_documents 方法")

        self.vector_db = FAISS.from_documents(self.all_small_chunks, self.embeddings)
        print("向量索引构建完成")

    def call_qwen_api(self, prompt: str, model: str = "qwen-plus-2025-04-28", temperature: float = 0.1) -> Optional[str]:
        """
        调用通义千问API

        Args:
            prompt: 输入的提示文本
            model: 使用的模型名称
            temperature: 生成温度，控制创造性

        Returns:
            API响应文本或None（如果出错）
        """
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "temperature": temperature,
                "top_p": 0.8,
                "result_format": "text"
            }
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()
            return result["output"]["text"]
        except requests.exceptions.RequestException as e:
            print(f"API请求出错: {e}")
        except KeyError as e:
            print(f"解析API响应出错: {e}")
        except Exception as e:
            print(f"未知错误: {e}")

        return None

    def rag_query(self, query: str, k: int = 3) -> Optional[str]:
        """
        执行RAG查询

        Args:
            query: 查询问题
            k: 检索的相关块数量

        Returns:
            生成的答案或None（如果出错）
        """
        if self.vector_db is None:
            raise ValueError("向量索引未初始化，请先调用 build_vector_index 方法")

        # a) 使用查询检索最相关的"小"块
        retrieved_small_docs = self.vector_db.similarity_search(query, k=k)

        print("\n--- 检索到的最相关'小'块 ---")
        for i, doc in enumerate(retrieved_small_docs):
            print(f"[Small Chunk {i + 1}]: {doc.page_content}\n")

        # b) 根据映射字典，找到这些"小"块对应的父"大"块
        retrieved_big_contents = set()  # 使用集合自动去重

        for small_doc in retrieved_small_docs:
            small_id = self._generate_chunk_id(small_doc.page_content)
            if small_id in self.mapping_dict:
                retrieved_big_contents.add(self.mapping_dict[small_id]["big_chunk_content"])
            else:
                # 如果找不到映射，使用小块本身
                retrieved_big_contents.add(small_doc.page_content)

        # 将去重后的大块内容合并为上下文
        context = "\n\n---\n\n".join(retrieved_big_contents)

        # c) 构建Prompt，调用QWen API生成答案
        prompt_template = f"""请根据以下上下文信息回答问题。如果上下文不包含答案，请如实告知。

上下文：
{context}

问题：
{query}

请给出准确、简洁的回答："""

        print("\n--- 发送给QWen API的Prompt ---")
        print(prompt_template)

        # 调用API
        answer = self.call_qwen_api(prompt_template)
        return answer


def main():
    """主函数"""
    # 获取API密钥
    API_KEY = "sk-1bc48ff360614aa7a1b14e7e68171afb"
    if not API_KEY:
        print("错误: 请设置DASHSCOPE_API_KEY环境变量")
        return

    # 初始化RAG系统
    rag_system = RAGSystem(api_key=API_KEY)

    # 文档文本
    fake_document_text = """机器学习是人工智能的一个子领域，它使计算机系统能够从数据中学习并改进，而无需显式编程。机器学习算法通常分为三类：监督学习、无监督学习和强化学习。监督学习使用标记数据来训练模型，例如用于图像分类。无监督学习在未标记数据中寻找隐藏模式，例如客户细分。强化学习则通过与环境交互并获得奖励来学习最佳策略，例如AlphaGo。深度学习是机器学习的一个分支，它使用称为神经网络的多层模型。这些网络能够从大量数据中学习复杂的特征层次结构。卷积神经网络（CNN）特别适用于图像处理任务，而循环神经网络（RNN）则擅长处理序列数据，如文本或时间序列。"""

    # 加载和预处理文档
    documents = rag_system.load_and_preprocess_documents(
        fake_document_text,
        metadata={"source": "ml_textbook_chapter1"}
    )

    # 分割文档
    rag_system.split_documents(documents)

    # 构建向量索引
    rag_system.build_vector_index()

    # 查询示例
    query = "CNN神经网络主要用于什么任务？"

    # 执行RAG查询
    result = rag_system.rag_query(query)

    print("\n--- 最终答案 ---")
    print(result)


if __name__ == "__main__":
    main()


# 切分出 1 个大块
# 切分出 4 个小块
# 向量索引构建完成
#
# --- 检索到的最相关'小'块 ---
# [Small Chunk 1]: 学习是机器学习的一个分支，它使用称为神经网络的多层模型。这些网络能够从大量数据中学习复杂的特征层次结构。卷积神经网络（CNN）特别适用于图像处理任务，而循环神经网络（RNN）则擅长处理序列数据，如文本
#
# [Small Chunk 2]: 网络（RNN）则擅长处理序列数据，如文本或时间序列。
#
# [Small Chunk 3]: 数据来训练模型，例如用于图像分类。无监督学习在未标记数据中寻找隐藏模式，例如客户细分。强化学习则通过与环境交互并获得奖励来学习最佳策略，例如AlphaGo。深度学习是机器学习的一个分支，它使用称为神经
#
#
# --- 发送给QWen API的Prompt ---
# 请根据以下上下文信息回答问题。如果上下文不包含答案，请如实告知。
#
# 上下文：
# 机器学习是人工智能的一个子领域，它使计算机系统能够从数据中学习并改进，而无需显式编程。机器学习算法通常分为三类：监督学习、无监督学习和强化学习。监督学习使用标记数据来训练模型，例如用于图像分类。无监督学习在未标记数据中寻找隐藏模式，例如客户细分。强化学习则通过与环境交互并获得奖励来学习最佳策略，例如AlphaGo。深度学习是机器学习的一个分支，它使用称为神经网络的多层模型。这些网络能够从大量数据中学习复杂的特征层次结构。卷积神经网络（CNN）特别适用于图像处理任务，而循环神经网络（RNN）则擅长处理序列数据，如文本或时间序列。
#
# 问题：
# CNN神经网络主要用于什么任务？
#
# 请给出准确、简洁的回答：
#
# --- 最终答案 ---
# CNN神经网络主要用于图像处理任务。