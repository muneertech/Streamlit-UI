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

step1: Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

step2: Install the Python dependencies.

```powershell
pip install -r requirements.txt
```

step3: Add your OpenAI API key to a `.env` file.

```env
OPENAI_API_KEY=your_api_key_here
```

## Run The App

step4: Start Streamlit.

```powershell
streamlit run app.py
```

step5: Open the local URL shown by Streamlit in your browser.

## Important Steps

step1: Upload a `.pdf` or `.txt` file in the Streamlit UI.

step2: The app saves the upload temporarily and detects the file type.

step3: The document loader extracts text from the file.

step4: The text is split into overlapping chunks for more reliable retrieval.

step5: OpenAI generates embeddings for each chunk and FAISS builds the vector index.

step6: Enter a question about the document in the text input.

step7: The app retrieves relevant chunks and uses `gpt-4o-mini` to answer based on context.

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
