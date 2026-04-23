import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from openai import RateLimitError
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings   
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def show_openai_quota_message() -> None:
    st.error(
        "OpenAI rejected the request because the API quota for this key is exhausted. "
        "Add billing or use a key with available quota, then try again."
    )

st.set_page_config(page_title="RAG Document Chat")
st.title("Chat With Your Documents")

st.markdown(
    """
    **Important steps**

    1. step1: Upload a PDF or TXT file.
    2. step2: The app extracts and splits the document text.
    3. step3: OpenAI generates embeddings for the document chunks.
    4. step4: FAISS stores the embeddings and enables retrieval.
    5. step5: Ask a question and get an answer from the document context.
    """
)

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file:
    # step1: save the uploaded file to a temporary local path
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    try:
        # step2: load text from the uploaded document
        if file_ext == ".pdf":
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path, encoding="utf-8")

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        if not docs:
            st.session_state.vectorstore = None
            st.error(
                "No readable text was found in that file. Try a document with selectable text "
                "or paste the content into a .txt file."
            )
        else:
            try:
                # step3: generate embeddings for chunks and store them in FAISS
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(docs, embeddings)
                st.session_state.vectorstore = vectorstore
                st.success(f"Document processed into {len(docs)} chunks.")
            except RateLimitError:
                st.session_state.vectorstore = None
                show_openai_quota_message()
    except Exception as exc:
        st.session_state.vectorstore = None
        st.error(f"Unable to process the uploaded file: {exc}")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

query = st.text_input("Ask a question about the document")

if query and st.session_state.vectorstore:
    # step4: retrieve relevant chunks from the vector store and answer the query
    retriever = st.session_state.vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant answering questions based on the provided context.

Context:
{context}

Question: {input}

Answer:"""
    )

    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(retriever, document_chain)

        result = qa_chain.invoke({"input": query})
        answer = result.get("answer", "")

        st.subheader("Answer")
        st.write(answer)
    except RateLimitError:
        show_openai_quota_message()
    except Exception as exc:
        st.error(f"Unable to answer the question: {exc}")
