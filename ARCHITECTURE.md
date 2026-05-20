# askdocs — 적용 기술 정리

RAG 기반 문서 Q&A 시스템. 문서를 업로드하면 자연어로 질문할 수 있다.

---

## 전체 파이프라인

```
문서 업로드
    └→ DocumentLoader → Chunker (Parent-Child) → Embedder → PostgreSQL + pgvector

사용자 질문
    └→ Router (simple/complex 분기)
           ├─ simple → embed_query
           └─ complex → HyDE → embed_query
                              ↓
                       Hybrid Search (BM25 + KNN, RRF)   top_k=50
                              ↓
                       Cross-Encoder Reranker             top_k=5
                              ↓
                       Generate (Ollama)  ← parent 청크 전문 사용
                              ↓
                           답변 반환
```

---

## 1. 인제스트 파이프라인

### 문서 로딩 — `src/askdocs/ingestion/loader.py`

PDF와 텍스트 파일을 비동기로 로드.

```python
async def load(self, file_path: str) -> str:
    if suffix == ".pdf":
        return await self._load_pdf(file_path)
    elif suffix in (".md", ".txt"):
        return await self._load_text(file_path)
```

### Parent-Child 청킹 — `src/askdocs/ingestion/chunker.py`

작은 child 청크로 검색 정확도를 높이고, 큰 parent 청크를 LLM에 전달해 문맥을 보존.

```python
def chunk_parent_child(self, text: str) -> List[Tuple[str, List[str]]]:
    parents = self._parent_splitter.split_text(text)   # 500 토큰
    return [
        (parent, self._child_splitter.split_text(parent))  # child: 100 토큰
        for parent in parents
    ]
```

| 구분 | 크기 | 용도 |
|------|------|------|
| Parent 청크 | 500 토큰 (`chunk_size`) | LLM 컨텍스트 |
| Child 청크 | 100 토큰 (`child_chunk_size`) | 벡터 검색 |

설정값: `src/askdocs/core/config.py`
```python
chunk_strategy: str = Field(default="parent_child")
chunk_size: int = Field(default=500)
child_chunk_size: int = Field(default=100)
```

### 인제스트 서비스 — `src/askdocs/ingestion/service.py`

Parent를 먼저 저장해 ID를 확보한 뒤, Child에 `parent_id`를 연결.

```python
async def _ingest_parent_child(self, ...):
    for parent_idx, (parent_text, child_texts) in enumerate(pairs):
        parent_chunk = Chunk(content=parent_text, embedding=None, ...)
        db.add(parent_chunk)
        await db.flush()  # parent.id 확보

        vectors = await self._embedder.embed(child_texts)
        db.add_all([
            Chunk(..., parent_id=parent_chunk.id)
            for child_text, vector in zip(child_texts, vectors)
        ])
```

---

## 2. 텍스트 임베딩

**모델**: `BAAI/bge-m3` (다국어, 1024차원)  
**라이브러리**: `sentence-transformers`

`src/askdocs/ingestion/embedder.py`
```python
class Embedder:
    def __init__(self):
        self._model = SentenceTransformer(settings.embedding_model)  # BAAI/bge-m3

    async def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,  # Cosine 유사도용 L2 정규화
        )
        return embeddings.tolist()
```

---

## 3. 데이터 모델 — `src/askdocs/models/chunk.py`

Parent-Child 관계를 자기참조 FK로 표현. `parent_content` 컬럼 없이 JOIN으로 가져옴.

```python
class Chunk(Base):
    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content     = Column(Text, nullable=False)
    embedding   = Column(Vector(1024), nullable=True)  # parent는 None → 검색 제외
    chunk_index = Column(Integer, nullable=False)

    doc_id    = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    parent_id = Column(UUID(as_uuid=True), ForeignKey("chunks.id"), nullable=True)

    parent = relationship("Chunk", remote_side=[id])  # 자기참조
```

---

## 4. Hybrid Search — `src/askdocs/retrieval/`

Lexical(BM25) + Semantic(KNN) 결과를 RRF로 결합.

### BM25 — `bm25_store.py`

```python
tokenized = [chunk.split() for chunk in chunks]
self._bm25 = BM25Okapi(tokenized)
```

### 벡터 검색 — `vector_store.py`

Child 청크만 검색 (embedding IS NOT NULL), parent 즉시 로드.

```python
result = await db.execute(
    select(Chunk)
    .where(Chunk.embedding.is_not(None))           # parent 청크 제외
    .order_by(Chunk.embedding.cosine_distance(query_vector))
    .limit(top_k)
    .options(joinedload(Chunk.parent))             # N+1 방지
)
```

### RRF 결합 — `hybrid.py`

```python
def _rrf(self, vector_results, bm25_results, top_k, k=60):
    scores = {}
    for rank, chunk in enumerate(vector_results):
        scores[chunk.id] += 1 / (k + rank + 1)   # rank 기반 → 정규화 불필요
    for rank, content in enumerate(bm25_results):
        scores[matched_id] += 1 / (k + rank + 1)
    return sorted_top_k(scores)
```

---

## 5. Two-Stage Retrieval

1단계에서 많이 가져와 Recall을 확보하고, 2단계 Re-ranker로 정밀하게 줄여 LLM 비용을 절감.

```
Hybrid Search → top_k=50    (Recall 확보)
      ↓
Cross-Encoder  → top_k=5    (LLM 전달)
```

설정값: `src/askdocs/core/config.py`
```python
top_k: int = Field(default=50)       # 1차: 넓게
rerank_top_k: int = Field(default=5) # 2차: 좁게
```

### Cross-Encoder Reranker — `src/askdocs/reranking/cross_encoder.py`

BiEncoder(임베딩)와 달리 쿼리-문서 쌍을 함께 입력해 정밀한 관련성 점수 산출.

```python
class Reranker:
    def __init__(self):
        self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    def rerank(self, query, chunks, top_k=None):
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]
```

---

## 6. LangGraph 파이프라인 — `src/askdocs/pipeline/`

### 상태 머신 — `graph.py`

```python
graph.add_conditional_edges(
    "분기",
    lambda state: "complex" if state["is_complex"] else "simple",
    {"simple": "embed_query", "complex": "hyde"},
)
graph.add_edge("hyde",       "embed_query")
graph.add_edge("embed_query", "retrieve")
graph.add_edge("retrieve",    "rerank")
graph.add_edge("rerank",      "generate")
```

### 상태 타입 — `state.py`

```python
class GraphState(TypedDict):
    query:           str
    expanded_query:  Optional[str]        # HyDE 결과
    query_vector:    Optional[List[float]]
    chunks:          Optional[List[Chunk]]
    reranked_chunks: Optional[List[Chunk]]
    answer:          Optional[str]
    is_complex:      Optional[bool]
```

---

## 7. Query Expansion — HyDE

`src/askdocs/pipeline/nodes/hyde.py`

쿼리를 임베딩하는 대신, LLM이 가상의 답변 문서를 먼저 생성하고 그 문서를 임베딩해 검색. 복잡한 쿼리에서 관련 문서를 더 잘 찾는다.

```python
async def hyde_node(state: GraphState) -> GraphState:
    prompt = f"다음 질문에 대한 답변이 담긴 문서를 작성해줘.\n질문: {state['query']}"
    response = await ollama.chat(model=settings.ollama_model, ...)
    expanded_query = response["message"]["content"]  # 가상 문서 → embed_query로 전달
    return {**state, "expanded_query": expanded_query}
```

---

## 8. Query Router

`src/askdocs/pipeline/nodes/route.py`

질문 복잡도를 LLM으로 판단해 HyDE 적용 여부를 결정.

```python
async def router_node(state: GraphState) -> GraphState:
    prompt = """
    단순(simple): 단일 사실 조회  → "연차 신청 기한은?"
    복잡(complex): 비교/분석/다단계 추론 필요 → "연차랑 병가 차이점은?"
    simple 또는 complex 중 하나만 답해:"""

    is_complex = "complex" in response["message"]["content"].lower()
    return {**state, "is_complex": is_complex}
```

---

## 9. 생성 — Parent 청크 우선 사용

`src/askdocs/pipeline/nodes/generate.py`

Child로 검색됐더라도 LLM엔 Parent 전문을 전달해 문맥 단절을 방지.

```python
def _context_text(chunk) -> str:
    return chunk.parent.content if chunk.parent else chunk.content

async def generate_node(state: GraphState) -> GraphState:
    context = "\n\n".join([
        f"[{idx+1}] {_context_text(chunk)}"
        for idx, chunk in enumerate(state["reranked_chunks"])
    ])
```

---

## 10. API — `src/askdocs/api/`

FastAPI + 비동기 SQLAlchemy. 스트리밍 응답 지원.

```python
# chat.py
@router.post("/stream")
async def chat_stream(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    async def generate():
        state = await router_node(state)
        state = await retrieve_node(state, db)
        state = await rerank_node(state)
        async for token in generate_stream_node(state):
            yield token
    return StreamingResponse(generate(), media_type="text/plain")
```

```python
# database.py
engine = create_async_engine(
    settings.db_url,
    pool_size=10,
    max_overflow=20,
)
```

---

## 11. RAGAS 평가 — `eval/evaluate.py`

```python
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
```

| 메트릭 | 의미 |
|--------|------|
| `faithfulness` | 답변이 문서에 충실한가 (환각 감지) |
| `answer_relevancy` | 답변이 질문에 관련 있는가 |
| `context_recall` | 필요한 문서를 잘 찾았는가 |
| `context_precision` | 찾은 문서가 실제로 관련 있는가 |

설치: `pip install -e ".[eval]"`

---

## 기술 스택 요약

| 영역 | 기술 |
|------|------|
| API | FastAPI, Uvicorn |
| DB | PostgreSQL, pgvector, SQLAlchemy (async) |
| 임베딩 | sentence-transformers (`BAAI/bge-m3`, 1024dim) |
| Lexical 검색 | rank-bm25 (BM25Okapi) |
| Semantic 검색 | pgvector cosine distance |
| 점수 결합 | RRF (Reciprocal Rank Fusion) |
| Reranking | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| 청킹 | langchain-text-splitters (Recursive + Parent-Child) |
| 파이프라인 | LangGraph (StateGraph) |
| LLM | Ollama (gemma3, 로컬) |
| 평가 | RAGAS |