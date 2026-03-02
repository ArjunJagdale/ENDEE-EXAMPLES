from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from endee import Endee, Precision
from dotenv import load_dotenv
import os

load_dotenv()

# Longer, complex text
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

embeddings =OpenAIEmbeddings(
    base_url = "https://openrouter.ai/api/v1",
    api_key = os.getenv("OPENROUTER_API_KEY"),
    model = "openai/text-embedding-3-small"
)

# Connect to client
client = Endee()

### === Character Text Splitter ### ===
char_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=450,
    chunk_overlap=90
)

char_chunks = char_splitter.split_text(text)
print(f"\nTotal chunks: {len(char_chunks)}\n")

for i, chunk in enumerate(char_chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")

# create index for char split
client.create_index(
    name="char_splitter_index",
    dimension=1536,
    space_type="cosine",
    precision=Precision.FLOAT32
)

char_index = client.get_index(name="char_splitter_index")

# upsert chunks
char_vectors = []

for i, chunk in enumerate(char_chunks):
    vector = embeddings.embed_query(chunk)
    char_vectors.append({
        "id": f"char_{i}",
        "vector": vector,
        "meta": {"text": chunk, "chunk_num": i}
    })

char_index.upsert(char_vectors)
print(f"Upserted {len(char_vectors)} vectors to index '{char_index.name}'")

### === Recursive Character Text Splitter ### ===
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=450,
    chunk_overlap=110
)

rec_chunks = recursive_splitter.split_text(text)
print(f'\nTotal chunks: {len(rec_chunks)}\n')

for i, chunk in enumerate(rec_chunks):
    print(f'Chunk {i+1}: \n{chunk}\n')

# create index for recursive char split
client.create_index(
    name="recursive_splitter_index",
    dimension=1536,
    space_type="cosine",
    precision=Precision.FLOAT32
)

rec_index = client.get_index(name="recursive_splitter_index")

# upsert chunks
rec_vectors = []
for i, chunk in enumerate(rec_chunks):
    vector = embeddings.embed_query(chunk)
    rec_vectors.append({
        "id": f"rec_{i}",
        "vector":vector,
        "meta": {"text": chunk, "chunk_num": i}
    })


rec_index.upsert(rec_vectors)
print(f"Upserted {len(rec_vectors)} vectors to index '{rec_index.name}'")

### === Query both indexes ### ===
query = "What is Deep learning?"
query_vector = embeddings.embed_query(query)

print("\nChar splitter Output-")
char_results = char_index.query(vector=query_vector, top_k=2) 
for item in char_results:
    print(f"Similiarity: {item['similarity']:.4f}")
    print(f'Text: {item["meta"]["text"]}\n')

print("\nRecursive splitter Output-")
rec_results = rec_index.query(vector=query_vector, top_k=2) 
for item in rec_results:
    print(f"Similiarity: {item['similarity']:.4f}")
    print(f'Text: {item["meta"]["text"]}\n')
