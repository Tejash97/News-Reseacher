import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


groq_api_key = "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"
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama-3.1-70b-versatile"
)

st.title("Tejash: News Researcher Bot 📈")
st.sidebar.title("News Article URLs")
urls = st.sidebar.text_input("Enter URL")
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_llama.pkl"


main_placeholder = st.empty()

if process_url_clicked:
    if urls:
        loader = UnstructuredURLLoader(urls=[urls])
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        if not data:
            main_placeholder.error("Data loading failed. Check the URL or internet connection.")
            st.stop()


        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        if not docs:
            main_placeholder.error("Text splitting failed. No content to process.")
            st.stop()

        # Create embeddings and save them to FAISS index
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorindex_huggingface = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)


        with open(file_path, "wb") as f:
            pickle.dump(vectorindex_huggingface, f)
        st.success("Vector database created and saved successfully!")
    else:
        st.sidebar.error("Please enter a valid URL.")


query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            answer = result.get("answer", "").strip()
            if answer:
                st.write(answer)
            else:
                st.write("No relevant information found in the processed documents.")


            sources = result.get("sources", "").strip()
            if not sources:
                st.write("No sources available for this answer.")
            else:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
    else:
        st.error("FAISS index not found. Please process URLs first.")
