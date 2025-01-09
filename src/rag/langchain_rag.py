import sqlite3
from typing import List
import time

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
load_dotenv()

# Database setup
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

# Store chat history in the database
def store_chat_history(conn, user_message, npc_response):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (user_message, npc_response) VALUES (?, ?)", (user_message, npc_response))
    conn.commit()

# Retrieve chat history from the database
def retrieve_chat_history(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT user_message, npc_response FROM chat_history")
    rows = cursor.fetchall()
    history = []
    for user_message, npc_response in rows:
        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=npc_response))
    return history

# Initialize the model
chat = ChatZhipuAI(
    model="GLM-4-Flash",
    temperature=0.8,
)

# Prompt templates
def get_contextualize_question_prompt():
    contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the 
    chat history, formulate a standalone question which can be understood without the chat history. 
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

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
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    {context}"
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return qa_prompt

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

# Chroma setup
chroma_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding_generator,
    create_collection_if_not_exists=True
)

# Document loader and splitter
from langchain.document_loaders import TextLoader
loader = TextLoader("./sidamingzhu.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create retriever
def get_retriever(docs: List[Document]):
    db = Chroma.from_documents(docs, embedding_generator)
    return db.as_retriever(search_kwargs={"k": 5})

retriever = get_retriever(docs)

question_prompt = get_contextualize_question_prompt()
history_aware_retriever = create_history_aware_retriever(chat, retriever, question_prompt)

# qa chain
qa_prompt_template = get_answer_prompt()
qa_chain = create_stuff_documents_chain(chat, qa_prompt_template)
# Retrieve function
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Main function to chat with the NPC

print("Welcome to the NPC interaction system!")
# Combine history with the latest user question

from langchain_community.chat_message_histories import ChatMessageHistory

def get_session_history() -> ChatMessageHistory:
    conn = setup_database()
    chat_history = retrieve_chat_history(conn)  # Get the list of messages
    history = ChatMessageHistory(messages=chat_history)  # Wrap it in ChatMessageHistory
    return history

while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    # Contextualize the question to stand alone
    contextualize_question_chain = RunnableWithMessageHistory(
        question_prompt | chat,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    res = contextualize_question_chain.invoke({
        "input": user_input
    }, config={
        "configurable": {"session_id": "test456"}
    })
    print("改写后内容：\n" + res.content)

    res = conversational_rag_chain.invoke({
        "input": user_input
    }, config={
        "configurable": {"session_id": "test123"}
    })
    print("回答：\n" + res["answer"])
    conn = sqlite3.connect("chat_history.db")
    store_chat_history(conn, user_input, res["answer"])
