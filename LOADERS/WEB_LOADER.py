from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from endee import Endee, Precision
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
import warnings

load_dotenv()
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

# Load using WebBaseLoader
loader = WebBaseLoader("https://en.wikipedia.org/wiki/IBM")
documents = loader.load()

# Split into chunks

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", "", "."],
    chunk_size = 800,
    chunk_overlap = 150
)

chunks = splitter.split_documents(documents)
print(f"created {len(chunks)} chunks")

# Embedding creation

embeddings = SentenceTransformer('all-MiniLM-L6-v2')

# embeddings = OpenAIEmbeddings(
#     base_url = "https://openrouter.ai/api/v1",
#     api_key = os.getenv("OPENROUTER_API_KEY"),
#     model="openai/text-embedding-3-small"
# )

# init Endee
client = Endee()
client.create_index(
    name="YT_rag",
    dimension=384,
    space_type="cosine",
    precision = Precision.INT8D
)

index = client.get_index("YT_rag")

# store chunks as vectors
text_vectors = []
for i, chunk in enumerate(chunks):
    vector = embeddings.encode(chunk.page_content).tolist()
    text_vectors.append({
        "id": f"chunk_{i}",
        "vector": vector,
        "meta": {
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", "unknown")
        }
    })

index.upsert(text_vectors)


# init LLM model
llm = ChatOpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.7
)

# RAG system function
def ask_question(question, top_k=3):
    # 1. Encode question
    query_vector = embeddings.encode(question).tolist()

    # 2. Retrieve relevant chunks
    results = index.query(query_vector, top_k=top_k, ef=256)
    print("retrived context")
    for i, r in enumerate(results, 1):
        print(f"[{i}] Similarity: {r['similarity']:.4f}")
        print(f"result: {r['meta']['text']}...\n")

    # 3. Build context
    context = "\n\n".join([r["meta"]["text"] for r in results])

    # 4. Generate answer
    prompt = f"""You are a helpful AI assistant. Answer the question based on the given context. If you don't know the answer, just say "I don't know".

Context: {context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    return response.content


print(ask_question("What is IBM"))