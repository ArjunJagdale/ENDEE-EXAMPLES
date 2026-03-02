from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from endee import Endee, Precision
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
import warnings
import json

# -------------------- setup --------------------
load_dotenv()
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

# -------------------- load JSON --------------------
loader = JSONLoader(
    file_path="data/motors.json",
    jq_schema=".[]",          # each car = one document
    text_content=False       # keep as dict
)

documents = loader.load()
print(f"loaded {len(documents)} car records")

# -------------------- embeddings --------------------
embeddings = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- init Endee --------------------
client = Endee()
client.create_index(
    name="MOTORS_json_rag",
    dimension=384,
    space_type="cosine",
    precision=Precision.INT8D
)

index = client.get_index("MOTORS_json_rag")

# -------------------- upsert vectors --------------------
text_vectors = []

for i, doc in enumerate(documents):
    car = json.loads(doc.page_content)  # dict

    text = (
        f"Model: {car['model']}. "
        f"MPG: {car['mpg']}. "
        f"Cylinders: {car['cyl']}. "
        f"Displacement: {car['disp']}. "
        f"Horsepower: {car['hp']}. "
        f"Rear Axle Ratio: {car['drat']}. "
        f"Weight: {car['wt']}. "
        f"Quarter Mile Time: {car['qsec']}. "
        f"Engine Type: {'V-shaped' if car['vs'] == 0 else 'Straight'}. "
        f"Transmission: {'Manual' if car['am'] == 1 else 'Automatic'}. "
        f"Gears: {car['gear']}. "
        f"Carburetors: {car['carb']}."
    )

    vector = embeddings.encode(text).tolist()

    text_vectors.append({
        "id": f"car_{i}",
        "vector": vector,
        "meta": {
            "text": text,
            "model": car["model"]
        }
    })

index.upsert(text_vectors)
print("vectors upserted")

# -------------------- LLM --------------------
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0
)

# -------------------- RAG function --------------------
def ask_question(question, top_k=5):
    query_vector = embeddings.encode(question).tolist()

    results = index.query(query_vector, top_k=top_k, ef=256)

    if not results:
        return "I don't know."

    context = "\n".join([r["meta"]["text"] for r in results])

    prompt = f"""
You are an automotive data analyst.

Field definitions:
- mpg: Miles per gallon (fuel efficiency)
- cyl: Number of engine cylinders
- disp: Engine displacement
- hp: Horsepower
- drat: Rear axle ratio
- wt: Vehicle weight
- qsec: Quarter mile time
- vs: Engine type (0 = V-shaped, 1 = straight)
- am: Transmission (0 = automatic, 1 = manual)
- gear: Number of forward gears
- carb: Number of carburetors

Answer the question using ONLY the provided dataset.
You may compare numbers and reason logically.
If the answer is not present in the data, say "I don't know".

Dataset:
{context}

Question:
{question}

Answer (concise and factual):
"""

    response = llm.invoke(prompt)
    return response.content.strip()

# -------------------- test --------------------
print(ask_question("Which car has the highest horsepower?"))
print(ask_question("Best mpg among 8-cylinder cars"))
