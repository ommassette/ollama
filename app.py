import os
import sys
import warnings
import numpy as np
import ollama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

EMBEDDING_MODEL = "nomic-embed-text-v2-moe"


def load_document_chunks(file_path: str):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        print("Unsupported format. Use .pdf or .txt")
        return []

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=50)
    return splitter.split_documents(docs)

def clean_document_text(text):
    # Remove extra whitespace and newlines
    cleaned_text = ' '.join(text.split())
    return cleaned_text

def get_embedding(text):
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return np.array(response.embeddings[0])


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def search_documents(query_text, document_embeddings, top_k=3):
    if not document_embeddings:
        print("\nNo document embeddings found. Please add a document first.")
        return

    print("\nGenerating query embedding...")
    query_vec = get_embedding(query_text)

    scores = []
    for item in document_embeddings:
        sim = cosine_similarity(query_vec, item["embedding"])
        scores.append((sim, item["text"], item["source"]))

    scores.sort(key=lambda x: x[0], reverse=True)

    print(f"\n--- Top {min(top_k, len(scores))} Results ---")
    for idx, (score, text, source) in enumerate(scores[:top_k], start=1):
        print(f"\nResult {idx} (Score: {score:.4f}) [{source}]:")
        print(f"\"{text.strip()}\"")


def main():
    document_embeddings = []

    while True:
        print("\n" + "=" * 40)
        print("1. Would you like to add a document?")
        print("2. Enter your query based on the document")
        print("3. Close program")
        print("=" * 40)

        choice = input("Select an option (1-3): ").strip()

        if choice == "1":
            file_path = input("Enter path to PDF or TXT file: ").strip().strip('"\'')
            chunks = load_document_chunks(file_path)

            if not chunks:
                continue

            print(f"Processing and embedding {len(chunks)} chunks...")
            for chunk in chunks:
                cleaned_text = clean_document_text(chunk.page_content)
                emb = get_embedding(cleaned_text)
                document_embeddings.append({
                    "embedding": emb,
                    "text": cleaned_text,
                    "source": os.path.basename(file_path)
                })
            print(f"Successfully added {len(chunks)} chunks to embedding list. Total store size: {len(document_embeddings)}.")

        elif choice == "2":
            if not document_embeddings:
                print("\nDocument list is empty! Please add a document first.")
                continue

            query = input("Enter your query: ").strip()
            if query:
                search_documents(query, document_embeddings, top_k=3)

        elif choice == "3":
            print("Exiting program.")
            sys.exit(0)

        else:
            print("Invalid option. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()