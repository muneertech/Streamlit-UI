import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

load_dotenv()

st.set_page_config(page_title="RAG Document Chat")
st.title("📄 Chat With Your Documents")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    else:
        loader = TextLoader(temp_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(docs, embeddings)
    st.session_state.vectorstore = vectorstore

    st.success(f"Document processed into {len(docs)} chunks.")

query = st.text_input("Ask a question about the document")

if query and st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-4o-mini")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    answer = qa_chain.run(query)

    st.subheader("Answer")
    st.write(answer)