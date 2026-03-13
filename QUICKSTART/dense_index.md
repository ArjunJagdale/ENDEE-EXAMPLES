## Imports

```python
from endee import Endee, Precision
from sentence_transformers import SentenceTransformer
```

- **`Endee`** - the client class for talking to an Endee vector database server.
- **`Precision`** - an enum (a set of named constants) that controls how vectors are stored in memory. e.g. `INT8`, `FLOAT32`, etc.
- **`SentenceTransformer`** - a library that loads pre-trained models which convert text → numerical vectors (embeddings).

---

## Setup

```python
embeddings = SentenceTransformer("all-MiniLM-L6-v2")
```

- Downloads (or loads from cache) the `all-MiniLM-L6-v2` model.
- This model takes a sentence and outputs a **384-dimensional vector** - a list of 384 float numbers that numerically represent the meaning of that sentence.

```python
client = Endee()
```

- Creates a client that connects to an **Endee server running locally** (default: `localhost`).
- Think of it like creating a database connection object.

---

## `make_index(name)`

```python
existing = [idx["name"] for idx in client.list_indexes()["indexes"]]
```

- `client.list_indexes()` - asks the server: *"what indexes exist?"* Returns a dict like `{"indexes": [{"name": "X"}, ...]}`.
- The list comprehension extracts just the names into a plain list like `["DENSEINDEX", "OTHER"]`.

```python
if name not in existing:
    client.create_index(
        name = name,
        dimension = 384,
        space_type="cosine",
        M=16,
        ef_con=128,
        precision=Precision.INT8D
    )
```

- Only creates the index if it doesn't already exist (avoids duplicate errors).
- **`dimension=384`** - vectors stored here must be 384-dimensional. Matches the model output exactly.
- **`space_type="cosine"`** - the similarity metric used. Cosine similarity measures the *angle* between two vectors. Values range from 0 (unrelated) to 1 (identical meaning).
- **`M=16`** - an HNSW graph parameter. Each node connects to 16 neighbors. Higher = better recall, more memory.
- **`ef_con=128`** - controls accuracy during index *building*. Higher = more accurate graph construction, slower inserts.
- **`Precision.INT8D`** - stores vectors as 8-bit integers instead of 32-bit floats. Uses **4x less memory**, slight accuracy tradeoff.

```python
return client.get_index(name=name)
```

- Returns an **index object** you can call `.upsert()` and `.query()` on. Like getting a reference to a specific database table.

---

## `upsert_docs(index)`

```python
docs = [ {"id": "doc1", "text": "...", "category": "tech"}, ... ]
```

- Plain Python list of dictionaries. Just raw data - nothing special yet.

```python
vectors = [
    {
        'id': doc['id'],
        'vector': embeddings.encode(doc['text']).tolist(),
        'meta': {'title': doc['id'], 'text': doc['text']},
        'filter': {'category': doc['category']}
    }
    for doc in docs
]
```

This is a **list comprehension** that transforms each `doc` into a vector-ready dict:

- **`'id'`** - unique identifier for this vector. Used to update or delete it later.
- **`embeddings.encode(doc['text'])`** - runs the text through the model, outputs a NumPy array of 384 floats.
- **`.tolist()`** - converts NumPy array → plain Python list (required for JSON serialization).
- **`'meta'`** - arbitrary key-value data stored alongside the vector. Not used for search, but returned with results. Here it stores the original text so you can display it later.
- **`'filter'`** - metadata used for **pre-filtering** searches. e.g. *"only search docs where category = tech"*.

```python
index.upsert(vectors)
```

- **Upsert** = insert + update. If the ID exists → update it. If not → insert it.
- Sends all 5 vectors to the Endee server in one call.

---

## `query(index, question, top_k)`

```python
query_vector = embeddings.encode(question).tolist()
```

- Encodes the user's question into a 384-dim vector using the **same model**.
- Critical: you must use the same model for both storing and querying, otherwise the vector spaces won't match.

```python
results = index.query(query_vector, top_k=top_k)
```

- Sends the query vector to Endee.
- Endee uses **HNSW (Hierarchical Navigable Small World)** - a graph-based approximate nearest neighbor algorithm - to find the `top_k` most similar vectors fast.
- Returns a list of result dicts with `id`, `similarity`, and `meta`.

```python
for item in results:
    print(f"ID: {item['id']} | Similarity: {item['similarity']:.4f}")
    print(f"TEXT: {item['meta']['text']}\n")
```

- `:.4f` - formats the float to 4 decimal places. e.g. `0.8734`.
- Prints the matched document's ID, how similar it was, and the original text retrieved from `meta`.

---

## `__main__` block

```python
if __name__ == "__main__":
```

- This block **only runs when you execute this file directly** (`python file.py`).
- If someone imports this file as a module, this block is skipped.

```python
index = make_index(INDEX_NAME)   # Create or reuse index
upsert_docs(index)               # Store 5 documents as vectors
query(index, "What is Langchain?", top_k=3)  # Find 3 most similar docs
```

- Full pipeline: setup → store → search.
- The query *"What is Langchain?"* will match `doc1` with high similarity because the model understands semantic meaning, not just keywords.

```python
print("\n ALL INDEXES:", [i['name'] for i in client.list_indexes()['indexes']])
```

- Lists all indexes on the server. Same pattern as before - extract names from the response dict.

---

## The Big Picture

```
Text → SentenceTransformer → 384-dim vector → Endee (HNSW index)
                                                      ↑
Query text → same model → query vector → cosine similarity search → top_k results
```

This is a **semantic search system**. It finds documents by *meaning*, not exact word matches. That's why querying *"What is Langchain?"* can match *"LangChain is a framework..."* even with slightly different wording.
