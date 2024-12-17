import sqlite3
from typing import List
import time
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_models import ChatZhipuAI
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.document_loaders import TextLoader
load_dotenv()

def setup_database():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute(""" 
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            npc_response TEXT
        )
    """)
    conn.commit()
    return conn

def store_chat_history(conn, user_message, npc_response):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (user_message, npc_response) VALUES (?, ?)", (user_message, npc_response))
    conn.commit()

def retrieve_chat_history(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT user_message, npc_response FROM chat_history")
    rows = cursor.fetchall()
    history = ChatMessageHistory()
    for user_message, npc_response in rows:
        history.add_message(HumanMessage(content=user_message))
        history.add_message(AIMessage(content=npc_response))
    return history

# Initialize the model
chat = ChatZhipuAI(
    model="GLM-4-Flash",
    temperature=0.5,
)

def get_contextualize_question_prompt():

    contextualize_q_system_prompt = """
    你是一个问题重写助手。你的任务是根据聊天历史和用户最后提出的问题，改写该问题以使其具有独立性。
    规则如下：
    1. 你必须仅返回改写后的问题。
    2. 严禁回答问题或输出任何与问题无关的内容。
    3. 如果没有聊天历史，直接返回用户的问题。
    输出格式：只包含改写后的问题内容，不能包含其他信息。
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return contextualize_q_prompt


def get_answer_prompt():
    qa_system_prompt = """
    你是一个问答任务的助手，请依据以下检索出来的信息去回答问题：
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return qa_prompt

def clean_contextualized_question(output: str) -> str:
    """
    解析模型输出，确保输出只有改写后的问题，剔除其他内容。
    """
    # 简单检查，保留输出的第一行作为改写后问题
    return output.strip().split("\n")[0]


# Embedding generator for Chroma
class EmbeddingGenerator(Embeddings):
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = ZhipuAI()

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(model=self.model_name, input=text)
            if hasattr(response, 'data') and response.data:
                embeddings.append(response.data[0].embedding)
            else:
                embeddings.append([0] * 1024)  # Default zero vector if embedding fails
        return embeddings

    def embed_query(self, query):
        response = self.client.embeddings.create(model=self.model_name, input=query)
        if hasattr(response, 'data') and response.data:
            return response.data[0].embedding
        return [0] * 1024  # Default zero vector if embedding fails


embedding_generator = EmbeddingGenerator(model_name="embedding-2")


# chroma_store = Chroma(
#     collection_name="example_collection",
#     embedding_function=embedding_generator,
#     create_collection_if_not_exists=True
# )


loader = TextLoader("./sidamingzhu.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)



def get_retriever(docs: List[Document]):
    db = Chroma.from_documents(docs, embedding_generator)
    return db.as_retriever(search_kwargs={"k": 5})


retriever = get_retriever(docs)

question_prompt = get_contextualize_question_prompt()

history_aware_retriever = create_history_aware_retriever(chat, retriever, question_prompt)


qa_prompt_template = get_answer_prompt()
qa_chain = create_stuff_documents_chain(chat, qa_prompt_template)


rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)


conn = setup_database()
#chat_history = retrieve_chat_history(conn)
# print(chat_history)
def get_session_history():
    return retrieve_chat_history(conn)

# Conversational RAG chain with message history
conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,  # Pass the callable function here
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

# 改写用户内容部分
contextualize_question_chain = RunnableWithMessageHistory(
        question_prompt | chat,
        get_session_history,  # Pass the callable function here
        input_messages_key="input",
        history_messages_key="chat_history"
    )

print("Welcome to the NPC interaction system!")

while True:
    user_input = input("请输入: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break
    # Get the reformulated question
    res = contextualize_question_chain.invoke({
        "input": user_input
    })
    revised_question = clean_contextualized_question(res.content)
    print("改写后内容：\n" + revised_question)

    # Get the response from the conversational chain
    res = conversational_rag_chain.invoke({
        "input": user_input
    })
    npc_response = res["answer"]
    print("回答：\n" + npc_response)
    store_chat_history(conn, user_input, npc_response)
