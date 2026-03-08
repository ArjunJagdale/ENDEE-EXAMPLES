import os
import warnings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from endee import Endee, Precision
import re

load_dotenv()
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

# --- Setup ---
embeddings = SentenceTransformer("all-MiniLM-L6-v2")

client = Endee()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.7
)

# --- Cleaning Function ---
"""
WebBaseLoader scrapes raw HTML - result contains leftover whitespace,
newlines, and navigation/footer noise. We clean that up before upserting.
"""
def clean_text(text: str) -> str:
    text = re.sub(r'\[\d+\]', '', text)   # remove wikipedia-style references [1][2]
    text = re.sub(r'\s+', ' ', text)      # collapse whitespace and newlines
    return text.strip()


# --- Index Creation ---
"""Check if index exists, reuse if it does, create if it doesn't."""
def make_index(name: str):
    existing = [idx["name"] for idx in client.list_indexes()["indexes"]]
    if name not in existing:
        client.create_index(
            name=name,
            dimension=384,
            space_type="cosine",
            precision=Precision.INT8D
        )
        print(f"[+] INDEX CREATED: {name}")
    else:
        print(f"[!] REUSING INDEX: {name}")
    return client.get_index(name=name)


# --- Upserting Docs ---
def upsert_docs(index, chunks):
    vectors = []
    for i, chunk in enumerate(chunks):
        vector = embeddings.encode(chunk.page_content).tolist()
        vectors.append({
            "id": f"web_chunk{i}",
            "vector": vector,
            "meta": {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),      # page URL
                "title": chunk.metadata.get("title", "unknown"),        # <title> tag
                "description": chunk.metadata.get("description", ""),  # <meta description>
                "language": chunk.metadata.get("language", "unknown"),  # html lang attribute
            },
            "filter": {"language": chunk.metadata.get("language", "unknown")}
        })
    index.upsert(vectors)
    print(f"[+] Upserted {len(vectors)} vectors")


# --- Ask Function ---
def ask(index, question: str, top_k: int = 5):
    query_vector = embeddings.encode(question).tolist()
    results = index.query(query_vector, top_k=top_k, ef=1024)

    print(f"{'='*60}")
    print(f"\n Query: {question}")
    print(f"{'='*60}")

    for i, r in enumerate(results, 1):
        meta = r['meta']
        print(f" [{i}] sim={r['similarity']:.4f} | title='{meta['title']}' | source={meta['source']}")
        print(f"{'-'*60}")
        print(f" {meta['text']}")

    context = "\n\n".join([r['meta']['text'] for r in results])
    prompt = f"""You are a helpful AI assistant. Answer based on the context below.
If you don't know, say "I don't know".
Context: {context}
Question: {question}
Answer:"""
    response = llm.invoke(prompt)
    print(f"{'-'*60}")
    print(f"\n LLM Answer: {response.content}")
    print(f"{'-'*60}")


# --- Main Function ---
if __name__ == "__main__":
    URL = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

    loader = WebBaseLoader(URL)
    documents = loader.load()
    print(f"Loaded {len(documents)} doc(s) from: {URL}")
    print(f"Title: {documents[0].metadata.get('title')}")
    print(f"Description: {documents[0].metadata.get('description')}")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", "", "."],
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    # Clean the data first
    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)

    print(f"Split into {len(chunks)} chunks")

    index = make_index("WEBLOADER")
    upsert_docs(index, chunks)
    ask(index, "What is RAG and how does it work?")
