from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature=0, 
    groq_api_key='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', 
    model_name="llama-3.1-70b-versatile"
)



import os
import streamlit as st
import pickle
import time
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

from langchain_community.embeddings import OpenAIEmbeddings



loaders = UnstructuredURLLoader(urls=[
    "https://www.moneycontrol.com/news/business/ipo/some-strong-2024-ipos-see-erosion-in-gains-others-rebound-after-tepid-debut-12804673.html",
    "https://www.moneycontrol.com/news/business/markets/why-market-experts-are-most-bullish-on-metals-after-trumps-win-here-are-the-key-reasons-12860593.html"
])
data = loaders.load() 
len(data)



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(data)

len(docs)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorindex_huggingface = FAISS.from_documents(docs, embeddings)




file_path="vector_index.pkl"
with open(file_path, "wb") as f:
    pickle.dump(vectorindex_huggingface, f)


if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorIndex = pickle.load(f)

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
print(chain)


query = "how many IPO where launched in 2024"
langchain.debug=True

chain({"question": query}, return_only_outputs=True)
