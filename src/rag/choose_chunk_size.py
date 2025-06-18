# coding:utf-8
import random
import time
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import DatasetGenerator, FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SentenceSplitter
import asyncio

# 修复 Windows 事件循环问题
if os.name == 'nt':  # Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = "sk-"

# 初始化 LLM 和 Embedding
llm = OpenAI(
    model="gpt-3.5-turbo",
    api_base="https://api.wlai.vip/v1"
)

# 使用 OpenAI 嵌入模型
embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    api_base="https://api.wlai.vip/v1"
)

# 配置全局设置
Settings.llm = llm
Settings.embed_model = embed_model

# 加载文档
data_dir = "./ch_data"
documents = SimpleDirectoryReader(data_dir).load_data()

# 生成评估问题 
num_eval_questions = 10
eval_documents = documents[0:2]
data_generator = DatasetGenerator.from_documents(eval_documents, llm=llm)
eval_questions = data_generator.generate_questions_from_nodes(num=20)
k_eval_questions = random.sample(eval_questions, num_eval_questions)

# 初始化评估器
faithfulness_evaluator = FaithfulnessEvaluator()
relevancy_evaluator = RelevancyEvaluator()


def evaluate_response_time_and_accuracy(chunk_size):
    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0

    # 创建带有自定义分块大小的节点解析器
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size // 5)

    # 使用自定义分块创建索引
    vector_index = VectorStoreIndex.from_documents(
        eval_documents,
        transformations=[node_parser]
    )

    query_engine = vector_index.as_query_engine()
    num_questions = len(k_eval_questions)  # 使用抽样后的问题数量

    for question in k_eval_questions:  # 使用抽样后的问题
        start_time = time.time()
        response_vector = query_engine.query(question)
        elapsed_time = time.time() - start_time

        faithfulness_result = faithfulness_evaluator.evaluate_response(
            response=response_vector
        ).passing

        relevancy_result = relevancy_evaluator.evaluate_response(
            query=question, response=response_vector
        ).passing

        total_response_time += elapsed_time
        total_faithfulness += 1 if faithfulness_result else 0
        total_relevancy += 1 if relevancy_result else 0

    average_response_time = total_response_time / num_questions
    average_faithfulness = total_faithfulness / num_questions
    average_relevancy = total_relevancy / num_questions

    return average_response_time, average_faithfulness, average_relevancy


# 测试不同分块大小
for chunk_size in [128, 256, 512, 1024, 2048]:
    avg_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(chunk_size)
    print(f"Chunk size {chunk_size} - Average Response time: {avg_time:.2f}s, "
          f"Average Faithfulness: {avg_faithfulness:.2f}, "
          f"Average Relevancy: {avg_relevancy:.2f}")