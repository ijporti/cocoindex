"""RAG (Retrieval-Augmented Generation) pipeline example using cocoindex.

This pipeline demonstrates a complete RAG setup:
1. Ingest documents from a local directory
2. Chunk text into overlapping segments
3. Generate embeddings for each chunk
4. Store in a vector database
5. Provide a query interface that retrieves context and generates answers
"""

import cocoindex
from dataclasses import dataclass
from typing import Optional


@dataclass
class RagChunk:
    """A single chunk ready for RAG retrieval."""
    text: str
    embedding: list[float]
    source_file: str
    chunk_index: int
    metadata: dict


@cocoindex.transform_flow()
def rag_pipeline(data_scope: cocoindex.DataScope) -> None:
    """Full RAG indexing pipeline.

    Reads documents, splits them into overlapping chunks, embeds each chunk,
    and exports everything to a Postgres vector store for retrieval.
    """
    # --- Source: local markdown / text files ---
    data_scope["documents"] = cocoindex.sources.LocalFile(
        path="docs/",
        included_patterns=["**/*.md", "**/*.txt"],
        binary=False,
    )

    with data_scope["documents"].row() as doc:
        # Keep the raw filename for provenance tracking
        doc["filename"] = doc["metadata"]["filename"]

        # --- Step 1: split into overlapping chunks ---
        doc["chunks"] = cocoindex.functions.SplitRecursively(
            language=None,
            chunk_size=512,
            chunk_overlap=64,
        )(doc["text"])

        with doc["chunks"].row() as chunk:
            # --- Step 2: embed each chunk ---
            chunk["embedding"] = cocoindex.functions.SentenceTransformerEmbed(
                model="sentence-transformers/all-MiniLM-L6-v2",
            )(chunk["text"])

    # --- Sink: Postgres with pgvector ---
    data_scope["documents"].export(
        "rag_chunks",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename", "chunks.index"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="chunks.embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _embed_query(query: str) -> list[float]:
    """Embed a user query with the same model used during indexing."""
    embedder = cocoindex.functions.SentenceTransformerEmbed(
        model="sentence-transformers/all-MiniLM-L6-v2",
    )
    return embedder(query)


def retrieve(
    query: str,
    top_k: int = 5,
    score_threshold: Optional[float] = 0.35,
) -> list[RagChunk]:
    """Retrieve the most relevant chunks for *query*.

    Args:
        query: Natural-language question or search string.
        top_k: Maximum number of chunks to return.
        score_threshold: Minimum cosine similarity score (0-1). Chunks below
            this threshold are discarded.

    Returns:
        List of :class:`RagChunk` objects ordered by relevance (best first).
    """
    query_embedding = _embed_query(query)

    results = cocoindex.query.search(
        index_name="rag_chunks",
        vector_field="chunks.embedding",
        query_vector=query_embedding,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    return [
        RagChunk(
            text=r["chunks.text"],
            embedding=r["chunks.embedding"],
            source_file=r["filename"],
            chunk_index=r["chunks.index"],
            metadata={"score": r["_score"]},
        )
        for r in results
    ]


def run_pipeline() -> None:
    """Entry point: update the index and run a sample retrieval."""
    cocoindex.init()

    # Incrementally update the vector store
    rag_pipeline.update()

    # Demo retrieval
    sample_query = "How does cocoindex handle incremental updates?"
    chunks = retrieve(sample_query, top_k=3)

    print(f"Query: {sample_query}")
    for i, chunk in enumerate(chunks, 1):
        score = chunk.metadata.get("score", 0.0)
        print(f"  [{i}] (score={score:.3f}) {chunk.source_file} "
              f"chunk#{chunk.chunk_index}: {chunk.text[:120]}...")


if __name__ == "__main__":
    run_pipeline()
