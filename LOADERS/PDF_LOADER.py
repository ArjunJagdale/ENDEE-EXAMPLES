import os
import warnings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
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
We should clean what we upsert. As in our PDF we do have "[70][71] reference numbers" like reference numbers(we have made the PDF out of the WIKIPEDIA!)
"""
def clean_text(text: str) -> str:
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', " ", text)
    return text.strip()


# --- Index creation ---
"""What we will check here is, if the index exists or not, if not, then only create an index"""
def make_index(name: str):
    existing = [idx["name"] for idx in client.list_indexes()["indexes"]] # to return list of strings not a dict!!!
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
            "id": f"pdf_page{chunk.metadata.get('page', i)}_chunk{i}",
            "vector": vector,
            "meta": {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", -1) + 1, # PyPDFLoader uses 0-indexed pages
                "total_pages": chunk.metadata.get("total_pages", -1)
            },
            "filter": { "page": chunk.metadata.get("page", -1) + 1 }
        })
    index.upsert(vectors)
    print(f"[+] Upserted {len(vectors)} vectors")

# --- Ask function ---
def ask(index, question: str, top_k: int=5):
    query_vector = embeddings.encode(question).tolist()
    results = index.query(query_vector, top_k=top_k, ef=1024)

    print(f"{'='*60}")
    print(f"\n Query: {question}")
    print(f"{'='*60}")

    for i, r in enumerate(results, 1):
        meta = r['meta']
        print(f" [{i}] sim={r['similarity']: .4f} | page={meta['page']}")
        print(f"{'-'*60}")
        print(f" {meta['text']}")
    
    context = "\n\n".join([r['meta']['text'] for r in results])
    prompt = f"""You are a helpful AI assistant. Answer based on the context below.
if you don't know, say "I don't know".
Context: {context}
Question: {question}
Answer:"""
    response = llm.invoke(prompt)
    print(f"{'-'*60}")
    print(f"\n LLM Answer: {response.content}")
    print(f"{'-'*60}")

# --- Main Function --- 
if __name__ == "__main__":
    PDF_PATH = "IBM_HISTORY.pdf"

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Loader {len(documents)} pages from: {PDF_PATH}")

    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", "", "."] ,chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # Clean the data first
    """
    It iterates every chunk and **overwrites** `chunk.page_content` with the cleaned version in place. 
    So by the time `upsert_docs(index, chunks)` is called, the `chunks` list already contains cleaned text in `page_content`.
    """
    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)

    # Debugging where we see "Jeopardy" (should be page 7)
    for i, chunk in enumerate(chunks):
        if "Watson" in chunk.page_content or "Jeopardy" in chunk.page_content:
            print(f"*** FOUND at chunk {i}, page {chunk.metadata.get('page')}")
            print(f"    {chunk.page_content[:300]}")

    print(f"Split into {len(chunks)} chunks")

    index = make_index("PDFLOADER")
    upsert_docs(index, chunks)
    ask(index, "In 2011, how IBM gained worldwide attention? What was the reason?")

