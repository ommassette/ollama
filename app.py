import logging
import os
import re
from pathlib import Path
from typing import List

import chromadb
import ollama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL = "Saimon:latest"
EMBED_MODEL = "nomic-embed-text-v2-moe"

chroma_client = chromadb.PersistentClient(path="CHROMA")
collection_research = chroma_client.get_or_create_collection(name="CHROMA")


def clean_text(text: str) -> str:
    text = re.sub(r'-\n(\w)', r'\1', text)
    text = text.replace('•', '\n- ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def create_embedding(text: str) -> List[float]:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response.embeddings[0]


def process_and_ingest_files(file_paths: List[str]) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            logging.error(f"File not found: {file_path}")
            continue

        logging.info(f"Processing file: {file_path}")
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        pages = loader.load()
        for page in pages:
            page.page_content = clean_text(page.page_content)

        chunks = splitter.split_documents(pages)
        
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{path.stem}_{idx}"
            embedding = create_embedding(chunk.page_content)

            ids.append(chunk_id)
            documents.append(chunk.page_content)
            metadatas.append({"source": str(path)})
            embeddings.append(embedding)

        if ids:
            collection_research.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            logging.info(f"Successfully indexed {len(ids)} chunks from {file_path}")


def retrieve_from_vector_db(query: str, k: int = 4) -> List[str]:
    query_emb = create_embedding(query)
    response = collection_research.query(
        query_texts=[query],
        n_results=k,
        query_embeddings=[query_emb],
    )
    return response["documents"][0]


def answer(question: str, k: int = 4) -> str:
    chunks = retrieve_from_vector_db(question, k=k)
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""Answer the question using only the context below.
If the context does not contain the answer, say "I don't know based on the document."

Context:
{context}

Question: {question}"""

    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return resp.message.content


if __name__ == "__main__":
    raw_paths = input("Enter file paths (separated by commas): ")
    docs_to_upload = [p.strip() for p in raw_paths.split(",") if p.strip()]

    if docs_to_upload:
        process_and_ingest_files(docs_to_upload)

    user_query = input("Enter your question: ").strip()
    if user_query:
        logging.info(f"Querying: {user_query}")
        result = answer(user_query)
        logging.info(f"Response:\n{result}")