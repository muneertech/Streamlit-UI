import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from huggingface_utils import HuggingFaceEmbeddings, generate_answer

load_dotenv()


def show_model_error_message() -> None:
    st.error(
        "The Hugging Face model failed to generate a response. "
        "Try again or choose a smaller model if needed."
    )

st.set_page_config(page_title="RAG Document Chat - Hugging Face")
st.title("Chat With Your Documents (Hugging Face)")

st.markdown(
    """
    **Important steps**

    1. step1: Upload a PDF or TXT file.
    2. step2: The app extracts and splits the document text.
    3. step3: Hugging Face generates embeddings for the document chunks.
    4. step4: FAISS stores the embeddings and enables retrieval.
    5. step5: Ask a question and get an answer from the document context.
    """
)

## step1) Upload document and load text
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file:
    # step2: save the uploaded file to a temporary local path
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    try:
        # step3: load text from the uploaded document
        if file_ext == ".pdf":
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path, encoding="utf-8")

        documents = loader.load()
        # step4: split the document text into smaller chunks for embedding
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
                # step5: generate embeddings for chunks and store them in FAISS using Hugging Face
                embeddings = HuggingFaceEmbeddings()
                vectorstore = FAISS.from_documents(docs, embeddings)
                st.session_state.vectorstore = vectorstore
                st.success(f"Document processed into {len(docs)} chunks.")
            except Exception as exc:
                st.session_state.vectorstore = None
                st.error(f"Unable to generate embeddings: {exc}")
    except Exception as exc:
        st.session_state.vectorstore = None
        st.error(f"Unable to process the uploaded file: {exc}")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

query = st.text_input("Ask a question about the document")

if query and st.session_state.vectorstore:
    # step7: retrieve relevant chunks from the vector store and answer the query with Hugging Face
    docs = st.session_state.vectorstore.similarity_search(query, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)

    try:
        answer = generate_answer(context, query)

        st.subheader("Answer")
        st.write(answer)
    except Exception as exc:
        show_model_error_message()
        st.error(f"Unable to answer the question: {exc}")
