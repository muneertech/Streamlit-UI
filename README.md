# Streamlit RAG Document Chat

A small Streamlit app for chatting with the contents of an uploaded `.pdf` or `.txt` document using a retrieval-augmented generation (RAG) workflow.

## What It Does

- Uploads a PDF or text file
- Extracts and splits the document into chunks
- Builds a FAISS vector index with OpenAI embeddings
- Answers questions about the uploaded document with `gpt-4o-mini`

## Tech Stack

- Streamlit
- LangChain
- OpenAI
- FAISS
- PyPDF

## Project Files

- `app.py`: main Streamlit app
- `requirements.txt`: Python dependencies
- `.env`: local environment variables such as your OpenAI API key
- `temp.py`: older experimental version kept for reference

## Requirements

- Python 3.10+
- An OpenAI API key with available quota

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Add your OpenAI API key to `.env`.

```env
OPENAI_API_KEY=your_api_key_here
```

## Run The App

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## How It Works

1. The user uploads a `.pdf` or `.txt` file.
2. The app stores the upload temporarily on disk.
3. LangChain loaders extract the document text.
4. The text is split into chunks with overlap.
5. OpenAI embeddings are generated for each chunk.
6. FAISS stores those embeddings for retrieval.
7. User questions are answered with retrieved context plus `gpt-4o-mini`.

## Error Handling Included

- Empty or non-readable documents are rejected with a friendly message.
- Temporary uploaded files are deleted after processing.
- OpenAI quota errors are caught and displayed clearly.
- General upload and question-answering failures are surfaced in the UI.

## Notes

- Scanned PDFs without selectable text may not produce readable chunks.
- The app currently supports one uploaded document at a time per session.
- If your OpenAI account has no quota, document processing and answers will fail until you update billing or use another key.

## Future Improvements

- Support multiple documents
- Add local embeddings to avoid OpenAI quota dependency
- Persist vector indexes between sessions
- Add source citations in answers
- Improve UI feedback during long-running operations
