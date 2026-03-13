## The Vectors

```
ID  Vector
1   [1, 1]
2   [1, 3]
3   [3, 2]
4   [6, 1]
5   [6, 3]
6   [8, 2]
```

With cosine, **magnitude doesn't matter** - only direction. So let's feel that immediately:

```
[1, 1]  → points at 45°
[2, 2]  → also points at 45° → cosine = 1.0 with [1,1]
[6, 6]  → also points at 45° → cosine = 1.0 with [1,1]
```

Same direction = cosine 1.0, regardless of how long the vector is.

---

## Cosine Formula (one more time, clean)

```
          V1 · V2
cos θ = -----------
         |V1| × |V2|
```

Let's build a **full distance table** for all our vectors upfront, so search is easy later.

---

## Pre-computing All Cosine Similarities

**Node 1 [1,1] vs Node 2 [1,3]:**
```
dot     = (1×1) + (1×3) = 4
|V1|    = √(1+1) = 1.414
|V2|    = √(1+9) = 3.162
cosine  = 4 / (1.414 × 3.162) = 4 / 4.471 = 0.894
```

**Node 1 [1,1] vs Node 3 [3,2]:**
```
dot     = (1×3) + (1×2) = 5
|V1|    = 1.414
|V3|    = √(9+4) = 3.606
cosine  = 5 / (1.414 × 3.606) = 5 / 5.099 = 0.981
```

**Node 1 [1,1] vs Node 4 [6,1]:**
```
dot     = (1×6) + (1×1) = 7
|V1|    = 1.414
|V4|    = √(36+1) = 6.083
cosine  = 7 / (1.414 × 6.083) = 7 / 8.601 = 0.814
```

**Node 1 [1,1] vs Node 5 [6,3]:**
```
dot     = (1×6) + (1×3) = 9
|V1|    = 1.414
|V5|    = √(36+9) = 6.708
cosine  = 9 / (1.414 × 6.708) = 9 / 9.485 = 0.949
```

**Node 1 [1,1] vs Node 6 [8,2]:**
```
dot     = (1×8) + (1×2) = 10
|V1|    = 1.414
|V6|    = √(64+4) = 8.246
cosine  = 10 / (1.414 × 8.246) = 10 / 11.660 = 0.858
```

**Node 2 [1,3] vs Node 3 [3,2]:**
```
dot     = (1×3) + (3×2) = 9
|V2|    = 3.162
|V3|    = 3.606
cosine  = 9 / (3.162 × 3.606) = 9 / 11.402 = 0.789
```

**Node 2 [1,3] vs Node 4 [6,1]:**
```
dot     = (1×6) + (3×1) = 9
|V2|    = 3.162
|V4|    = 6.083
cosine  = 9 / (3.162 × 6.083) = 9 / 19.234 = 0.468
```

**Node 2 [1,3] vs Node 5 [6,3]:**
```
dot     = (1×6) + (3×3) = 15
|V2|    = 3.162
|V5|    = 6.708
cosine  = 15 / (3.162 × 6.708) = 15 / 21.211 = 0.707
```

**Node 2 [1,3] vs Node 6 [8,2]:**
```
dot     = (1×8) + (3×2) = 14
|V2|    = 3.162
|V6|    = 8.246
cosine  = 14 / (3.162 × 8.246) = 14 / 26.073 = 0.537
```

**Node 3 [3,2] vs Node 4 [6,1]:**
```
dot     = (3×6) + (2×1) = 20
|V3|    = 3.606
|V4|    = 6.083
cosine  = 20 / (3.606 × 6.083) = 20 / 21.931 = 0.912
```

**Node 3 [3,2] vs Node 5 [6,3]:**
```
dot     = (3×6) + (2×3) = 24
|V3|    = 3.606
|V5|    = 6.708
cosine  = 24 / (3.606 × 6.708) = 24 / 24.190 = 0.992
```

**Node 3 [3,2] vs Node 6 [8,2]:**
```
dot     = (3×8) + (2×2) = 28
|V3|    = 3.606
|V6|    = 8.246
cosine  = 28 / (3.606 × 8.246) = 28 / 29.735 = 0.942
```

**Node 4 [6,1] vs Node 5 [6,3]:**
```
dot     = (6×6) + (1×3) = 39
|V4|    = 6.083
|V5|    = 6.708
cosine  = 39 / (6.083 × 6.708) = 39 / 40.804 = 0.956
```

**Node 4 [6,1] vs Node 6 [8,2]:**
```
dot     = (6×8) + (1×2) = 50
|V4|    = 6.083
|V6|    = 8.246
cosine  = 50 / (6.083 × 8.246) = 50 / 50.160 = 0.997
```

**Node 5 [6,3] vs Node 6 [8,2]:**
```
dot     = (6×8) + (3×2) = 54
|V5|    = 6.708
|V6|    = 8.246
cosine  = 54 / (6.708 × 8.246) = 54 / 55.313 = 0.976
```

---

## Full Similarity Table

```
     1      2      3      4      5      6
1    -     0.894  0.981  0.814  0.949  0.858
2   0.894   -     0.789  0.468  0.707  0.537
3   0.981  0.789   -     0.912  0.992  0.942
4   0.814  0.468  0.912   -     0.956  0.997
5   0.949  0.707  0.992  0.956   -     0.976
6   0.858  0.537  0.942  0.997  0.976   -
```

With cosine, **higher = more similar**. So when building the graph, each node connects to its **highest scoring neighbors**.

---

## Building the Graph - M=2

Each node picks its **top 2 most similar** neighbors.

**Node 1 → top 2 similarities:**
```
vs 3 → 0.981  ✓
vs 5 → 0.949  ✓
vs 6 → 0.858
vs 2 → 0.894
```
Node 1 connects to: **3, 5**

**Node 2 → top 2:**
```
vs 1 → 0.894  ✓
vs 3 → 0.789  ✓
vs 5 → 0.707
```
Node 2 connects to: **1, 3**

**Node 3 → top 2:**
```
vs 5 → 0.992  ✓
vs 1 → 0.981  ✓
vs 6 → 0.942
```
Node 3 connects to: **5, 1**

**Node 4 → top 2:**
```
vs 6 → 0.997  ✓
vs 5 → 0.956  ✓
vs 3 → 0.912
```
Node 4 connects to: **6, 5**

**Node 5 → top 2:**
```
vs 3 → 0.992  ✓
vs 6 → 0.976  ✓
vs 4 → 0.956
```
Node 5 connects to: **3, 6**

**Node 6 → top 2:**
```
vs 4 → 0.997  ✓
vs 5 → 0.976  ✓
vs 3 → 0.942
```
Node 6 connects to: **4, 5**

---

## Layer 0 - Final Graph

```
1-> 3, 6
2-> 1, 3
3-> 1, 5
4-> 5, 6
5-> 3, 6
6-> 4, 5
```

---

## Layer 1 - Promoted Nodes: 1, 3, 6 (assumption!)

From the table, top neighbor pairs among {1, 3, 6}:

```
1 to 3 → 0.981
3 to 6 → 0.942  
1 to 6 → 0.858
```

```
So the Layer 1 is - 
Layer 1:   1 --- 3 --- 6
```

---

## Layer 2 - Entry Point: Node 1 (assume node 1 was promoted to layer 2)

```
Layer 2:   1
```

---

## Search - Query = [5, 2]

Find the most similar node to `[5, 2]`.

First compute query similarities against all nodes:

```
Q=[5,2],  |Q| = √(25+4) = 5.385

vs 1 [1,1]: dot=7,  cos = 7/(5.385×1.414) = 7/7.615  = 0.919
vs 2 [1,3]: dot=11, cos = 11/(5.385×3.162) = 11/17.027 = 0.646
vs 3 [3,2]: dot=19, cos = 19/(5.385×3.606) = 19/19.418 = 0.978
vs 4 [6,1]: dot=32, cos = 32/(5.385×6.083) = 32/32.757 = 0.977
vs 5 [6,3]: dot=36, cos = 36/(5.385×6.708) = 36/36.123 = 0.997
vs 6 [8,2]: dot=44, cos = 44/(5.385×8.246) = 44/44.404 = 0.991
```

Best match is **node 5 at 0.997**. Let's see if HNSW finds it.

---

**Layer 2 - Entry: Node 1**

```
Current best: node 1,  similarity = 0.919
```

Only node at this layer. Drop to Layer 1.

---

**Layer 1 - Start at Node 1, neighbors: 3**

```
node 3 → 0.978   better than 0.919 → move to node 3
```

At node 3, neighbors: 1, 6

```
node 1 → 0.919   worse
node 6 → 0.991   better than 0.978 → move to node 6
```

At node 6, neighbors: 3

```
node 3 → 0.978   worse than 0.991
```

No improvement. Best at Layer 1 = **node 6 at 0.991**. Drop to Layer 0.

---

**Layer 0 - Start at Node 6, neighbors: 4, 5**

```
node 4 → 0.977   worse than 0.991
node 5 → 0.997   BETTER → move to node 5
```

At node 5, neighbors: 3, 6

```
node 3 → 0.978   worse than 0.997
node 6 → 0.991   worse than 0.997
```

No improvement. **Search terminates.**

---

## Result

```
Closest node to query [5,2] → Node 5 at similarity 0.997
```

Correct. And we only computed **8 similarities during search** instead of computing all 6 at every layer.

---
