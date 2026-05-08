"""Hybrid search pipeline example for cocoindex.

Demonstrates combining dense vector search with sparse keyword (BM25-style)
search to produce more robust retrieval results.
"""

import cocoindex
from cocoindex import flow, sources, transforms, embeddings, storages


@cocoindex.flow_def(name="HybridSearchPipeline")
def hybrid_search_pipeline(flow_builder: flow.FlowBuilder, data_scope: flow.DataScope) -> None:
    """Build a hybrid search pipeline combining dense and sparse retrieval.

    This pipeline:
    1. Loads documents from a local directory
    2. Chunks text for granular retrieval
    3. Generates dense embeddings for semantic search
    4. Extracts sparse keyword features for lexical search
    5. Stores both representations in a vector store
    """
    # Source: read markdown and text documents
    data_scope["documents"] = flow_builder.add_source(
        sources.LocalFile(
            path="docs/",
            included_patterns=["*.md", "*.txt"],
        )
    )

    doc_embeddings = data_scope["documents"].transform(
        transforms.SplitRecursively(
            language="markdown",
            chunk_size=512,
            chunk_overlap=64,
        ),
        output_field="chunks",
    )

    # Dense embeddings via a sentence-transformer model
    doc_embeddings["dense_embedding"] = doc_embeddings["chunks"]["text"].transform(
        embeddings.SentenceTransformerEmbedding(
            model="all-MiniLM-L6-v2",
        )
    )

    # Sparse keyword vector (TF-IDF / BM25 approximation)
    doc_embeddings["sparse_embedding"] = doc_embeddings["chunks"]["text"].transform(
        transforms.SparseEmbedding(
            method="bm25",
            vocab_size=30_000,
        )
    )

    # Persist to a hybrid-capable vector store
    doc_embeddings.export(
        storages.QdrantStorage(
            collection_name="hybrid_docs",
            dense_vector_field="dense_embedding",
            sparse_vector_field="sparse_embedding",
            payload_fields=["text", "filename", "chunk_index"],
        )
    )


def search(
    query: str,
    top_k: int = 5,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[dict]:
    """Run a hybrid search query against the indexed documents.

    Args:
        query: Natural language search query.
        top_k: Number of results to return.
        dense_weight: Weight applied to the dense (semantic) score.
        sparse_weight: Weight applied to the sparse (keyword) score.

    Returns:
        List of result dicts with 'text', 'filename', 'score' keys.
    """
    pipeline = cocoindex.get_flow("HybridSearchPipeline")

    results = pipeline.query(
        query=query,
        top_k=top_k,
        fusion={
            "method": "rrf",  # Reciprocal Rank Fusion
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
        },
    )

    return [
        {
            "text": r.payload["text"],
            "filename": r.payload["filename"],
            "score": r.score,
        }
        for r in results
    ]


def run_pipeline() -> None:
    """Index documents and demonstrate a hybrid search query."""
    cocoindex.init()

    print("Building hybrid search index...")
    pipeline = hybrid_search_pipeline()
    pipeline.run()
    print("Indexing complete.")

    query = "how to configure authentication"
    print(f"\nHybrid search query: '{query}'")
    results = search(query, top_k=3)

    for rank, result in enumerate(results, start=1):
        print(f"  [{rank}] score={result['score']:.4f} | {result['filename']}")
        print(f"       {result['text'][:120].strip()}...")


if __name__ == "__main__":
    run_pipeline()
