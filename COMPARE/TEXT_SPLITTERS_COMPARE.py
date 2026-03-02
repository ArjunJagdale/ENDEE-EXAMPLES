from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from endee import Endee, Precision
from dotenv import load_dotenv
import os
import time

load_dotenv()

text = """
Artificial Intelligence has become one of the most transformative technologies of the 21st century. Machine learning, a core subset of AI, enables systems to learn patterns from data without being explicitly programmed for every scenario. This capability has revolutionized fields ranging from healthcare to finance.
Deep learning represents a more advanced form of machine learning, utilizing neural networks with multiple layers to process information in increasingly abstract ways. These networks can recognize complex patterns in images, speech, and text with remarkable accuracy. Convolutional Neural Networks excel at computer vision tasks, while Recurrent Neural Networks are particularly effective for sequential data like time series and natural language.
Natural Language Processing has made tremendous strides in recent years. Large Language Models like GPT and BERT have demonstrated unprecedented abilities in understanding and generating human language. These models can perform tasks such as translation, summarization, question answering, and even creative writing with impressive fluency.
Computer vision technology enables machines to interpret and understand visual information from the world. Applications range from facial recognition systems to autonomous vehicles that can navigate complex environments. Object detection, image segmentation, and image classification are fundamental tasks that have been revolutionized by deep learning approaches.
Reinforcement Learning represents a different paradigm where agents learn by interacting with an environment and receiving rewards or penalties. This approach has achieved superhuman performance in games like Chess, Go, and various video games. More importantly, it's being applied to real-world problems like robotics, resource management, and personalized recommendations.
Transfer learning has emerged as a powerful technique that allows models trained on one task to be adapted for related tasks with minimal additional training. This approach significantly reduces the data and computational requirements for developing new AI systems, making advanced AI more accessible to organizations with limited resources.
The field of Explainable AI addresses the black-box nature of many machine learning models. As AI systems are deployed in critical domains like healthcare and criminal justice, understanding how these systems make decisions becomes crucial for trust, accountability, and regulatory compliance.
Edge AI brings artificial intelligence capabilities directly to devices like smartphones, IoT sensors, and embedded systems. By processing data locally rather than in the cloud, edge AI reduces latency, enhances privacy, and enables AI applications in environments with limited connectivity.
AutoML systems automate the process of selecting algorithms, tuning hyperparameters, and engineering features. This democratization of machine learning allows domain experts without deep AI expertise to build effective models for their specific problems.
Few-shot and zero-shot learning techniques enable models to perform tasks with minimal or no task-specific training examples. These approaches are particularly valuable when labeled data is scarce or expensive to obtain, pushing AI systems closer to human-like learning capabilities.
Generative AI has captured public imagination with its ability to create novel content including text, images, music, and code. Models like DALL-E, Stable Diffusion, and ChatGPT demonstrate creativity and versatility that were previously thought to be uniquely human capabilities.
The ethical implications of AI development cannot be overlooked. Issues of bias, fairness, privacy, and the societal impact of automation require careful consideration. Responsible AI development involves not just technical excellence but also thoughtful engagement with these broader concerns to ensure AI benefits humanity as a whole.
"""

# init embeddings
embeddings = OpenAIEmbeddings(
    base_url = "https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/text-embedding-3-small"
)

# split text
char_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=450,
    chunk_overlap=90
)

chunks = char_splitter.split_text(text)

print(f"total chunks: {len(chunks)}\n")

### === Endee Setup === ###
print("Setting up Endee...")

client = Endee()

# Create index
start_time = time.time()
client.create_index(
    name = "endee_compare",
    dimension = 1536,
    space_type = "cosine",
    precision=Precision.FLOAT16
)
endee_index = client.get_index(name="endee_compare")
index_time = time.time() - start_time

# Upsert vectors to Endee
start_time = time.time()
endee_vectors = []

for i, chunk in enumerate(chunks):
    vector = embeddings.embed_query(chunk)
    endee_vectors.append({
        "id": f"endee_{i}",
        "vector": vector,
        "meta": {"text": chunk, "chunk_num": i}
    })

endee_index.upsert(endee_vectors)
upsert_time = time.time() - start_time

print(f"Index creation: {index_time:.4f}s")
print(f"Upsert {len(chunks)} vectors: {upsert_time:.4f}s")

### === FAISS Setup === ###

print("Setting up FAISS...")

# Create FAISS index
start_time = time.time()
faiss_index = FAISS.from_texts(chunks, embeddings)

### === Query comparison === ###
query = "What is deep learning? how it is related to machine learning?"
query_vector = embeddings.embed_query(query)

# Endee Query
start_time = time.time()
endee_results = endee_index.query(query_vector, top_k=2)
endee_time = (time.time() - start_time) * 1000

print("ENDEE - ")
for i, item in enumerate(endee_results, 1):
    print(f"{i} Similarity: {item['similarity']:.4f}")
    print(f"{item['meta']['text'][:120]}...\n")

# FAISS Query
start_time = time.time()
faiss_results = faiss_index.similarity_search_with_score(query, k=2)
faiss_time = (time.time() - start_time) * 1000

print("FAISS - ")
for i, (doc, score) in enumerate(faiss_results, 1):
    print(f" {i}. Similarity: {1- score:.4f}")
    print(f" {doc.page_content[:120]}...\n")

print(f"Endee query time: {endee_time:.2f} ms")
print(f"FAISS query time: {faiss_time:.2f} ms")



