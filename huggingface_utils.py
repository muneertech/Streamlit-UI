import re
from typing import TYPE_CHECKING, List

import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL_NAME = "google/flan-t5-small"

_embedding_model = None
_generator_model = None
_generator_tokenizer = None

URL_PATTERN = re.compile(r"https?://[^\s)>\]]+|www\.[^\s)>\]]+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'sentence-transformers'. Activate the project virtual "
                "environment and run 'pip install -r requirements.txt'."
            ) from exc

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embedding_model


def get_generator_components():
    global _generator_model, _generator_tokenizer

    if _generator_model is None or _generator_tokenizer is None:
        _generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
        _generator_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_NAME)

    return _generator_model, _generator_tokenizer


class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model: "SentenceTransformer" | None = None):
        self.model = model or get_embedding_model()

    def embed_documents(self, texts: List[str]):
        return [self.model.encode(text, convert_to_numpy=True).tolist() for text in texts]

    def embed_query(self, text: str):
        return self.model.encode(text, convert_to_numpy=True).tolist()


def extract_direct_answer(context: str, question: str) -> str | None:
    question_lower = question.lower()
    urls = URL_PATTERN.findall(context)
    emails = EMAIL_PATTERN.findall(context)
    phones = PHONE_PATTERN.findall(context)

    if "linkedin" in question_lower:
        for url in urls:
            if "linkedin.com" in url.lower():
                return url

    if "url" in question_lower or "link" in question_lower or "website" in question_lower:
        if urls:
            return urls[0]

    if "email" in question_lower or "mail" in question_lower:
        if emails:
            return emails[0]

    if "phone" in question_lower or "mobile" in question_lower or "contact number" in question_lower:
        if phones:
            return phones[0].strip()

    return None


def generate_answer(context: str, question: str, max_new_tokens: int = 256) -> str:
    direct_answer = extract_direct_answer(context, question)
    if direct_answer:
        return direct_answer

    model, tokenizer = get_generator_components()
    prompt = (
        "Answer using only the provided context.\n"
        "If the answer is present verbatim, copy it exactly.\n"
        "If the answer is not in the context, reply with: I could not find that in the document.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.1,
        )
    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if answer.lower().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    return answer or "I could not find that in the document."
