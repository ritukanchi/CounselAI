import os 
import json 
import qdrant_client
import warnings 
import time 
import requests
import re
from dotenv import load_dotenv

from qdrant_client import QdrantClient 
from langchain_core.output_parsers.string import StrOutputParser
from langchain.chains import create_retrieval_chain, create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Qdrant 
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')
qdrant_cloud_url = os.getenv("QDRANT_CLOUD_URL")
qdrant_cloud_api_key = os.getenv("QDRANT_CLOUD_API_KEY")


config_file = "./artifacts/config.json"
with open(config_file, "r") as file:
    config = json.load(file)

collection_name = config['COLLECTION_NAME']['bns-001']

llm = config['LLAMA']
embedding_model = config['EMBEDDING_MODEL']

def vector_store(embeddings, collection_name, to_cloud=False):
    client = QdrantClient(url="http://localhost:6333/") if not to_cloud else QdrantClient(url=qdrant_cloud_url, api_key=qdrant_cloud_api_key,)
        
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    return vector_store


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def memory_chain(query: str, chat_history: list, embedding_model, llm, client_url, collection_name):
    
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, 
                                            google_api_key=gemini_api_key)

    llm = ChatGroq(groq_api_key=groq_api_key,
                model=llm,
                temperature=0.2)

    vector_db = vector_store(embeddings, client_url, collection_name)
    retriever = vector_db.as_retriever(top_k=5)


    # Define system-prompt
    qa_system_prompt = (
        """You are legal advisor assistant for question-answering tasks.
        Use the following piece of retrieved context to answer the following query.
        
        **Don't answer without referring the context.**
        
        Find Bharatiya Nyaya Sanhita (BNS) Section which is criminal code of India from the context. 
        Focus on entities present in the information. Only add relevant information.
        
        USE BEAUTIFUL MARKDOWN FORMAT and ANSWER like chatbot. 
        
        Provide information in below format:
        A. BNS sections:
                List all sections in below format:
                Section (section number or name): Description \n
        
        B. Punishments:
                Define Punishments in detail for associate section name
        
        C. Legal Advice:
            Define legal advice including police and medical if emergency. Also prvoide
            help line number for India.
        
        Context:
        {context}
        
        """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )


    # question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # rag_chain = create_retrieval_chain(retriever, question_answer_chain)


    #  System Prompt
    contexualize_q_system_prompt = """ 
        Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history, Do Not answer the question, \
        just reformulate it if needed and otherwise return it as it.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contexualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # Hisotry aware retiever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    # Define QA prompts
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    responae = rag_chain.invoke({"input": query, "chat_history": chat_history})
    
    return responae

if __name__ == '__main__':
    chat_history = []
    i=0
    
    while i < 3:
        query = input("Ask a query ('exit' to quit): ")
        response = memory_chain(query, chat_history, embedding_model, llm, client, collection_name)
        chat_history.extend([HumanMessage(content=query), response["answer"]])
        print(response["answer"])
            
        print("\n\n")
        i += 1
    # # Embedding model
    # embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, 
    #                                         google_api_key=gemini_api_key)

    # # llm = ChatGroq(groq_api_key=groq_api_key,
    # #             model=llm,
    # #             temperature=0.2)

    # vector_db = vector_store(embeddings, client, collection_name)
    # retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k":3})
    # print(retriever)
    # result = retriever.invoke("Uncle rapes 22 years old. Body found at railway station.")
    # print(result)