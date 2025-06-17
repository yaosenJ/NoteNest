import random
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import DatasetGenerator
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="gpt-3.5-turbo",
    api_key="sk-",
    api_base="https://api.wlai.vip/v1")

data_dir = "./ch_data"

documents = SimpleDirectoryReader(data_dir).load_data()

num_eval_questions = 25

eval_documents = documents[0:20]
data_generator = DatasetGenerator.from_documents(eval_documents,llm=llm)
eval_questions = data_generator.generate_questions_from_nodes()
k_eval_questions = random.sample(eval_questions, num_eval_questions)

print(len(eval_questions))

print(k_eval_questions)
