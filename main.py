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
