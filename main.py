import os
import streamlit as st
import pickle
import time

# LangChain Core
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms import OpenAI
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain

# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document Loader (URL-based)
from langchain_community.document_loaders import UnstructuredURLLoader

# Embeddings
from langchain_community.embeddings import OpenAIEmbeddings

# Vector Store
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAI

from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("FactCheck 📈")
st.sidebar.title("News Article URLs")

# defile the no of urls
urls=[]
for i in range(2):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Process URLs")

file_path="faiss_store_openai" # a folder for vector database

#configure the embeddings
main_placefolder=st.empty()
llm=OpenAI(
    temperature=0,max_tokens=500)

if process_url_clicked:

    # Loading the data (url)
    loader=UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Loading the data...⏱️")
    data=loader.load()

    # Splitting the data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placefolder.text("Splitting the data into chunks...🖖")
    docs=text_splitter.split_documents(data)

    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai=FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embedding vector started building...👾")
    time.sleep(2)

    # Save the faiss index into a pickle file
    vectorstore_openai.save_local(file_path)

# section for questrion answering
query=main_placefolder.text_input("Ask your question...🤔")

# Load the FAISS index from the pickle file
if query:
    if os.path.exists(file_path):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            file_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        chain=RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result=chain({"question": query}, return_only_outputs=True)
        st.header("Answer:")
        st.write(result['answer'])

