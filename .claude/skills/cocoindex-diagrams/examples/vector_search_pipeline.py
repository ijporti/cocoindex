"""Vector search pipeline example for cocoindex.

Demonstrates building a pipeline that indexes documents with embeddings
and supports semantic vector search queries.
"""

import cocoindex
from typing import Annotated


@cocoindex.flow_def(name="VectorSearchPipeline")
def vector_search_pipeline(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """Build a pipeline that creates a searchable vector index from documents.

    This pipeline:
    1. Reads documents from a local directory
    2. Splits text into overlapping chunks
    3. Generates embeddings for each chunk
    4. Stores results in a vector store for similarity search
    """
    # Source: read markdown and text files from the docs directory
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="docs/",
            included_patterns=["*.md", "*.txt"],
        )
    )

    doc_embeddings = data_scope["documents"].row_handler()

    # Split documents into overlapping chunks for better retrieval
    doc_embeddings["chunks"] = doc_embeddings["content"].transform(
        cocoindex.functions.SplitRecursively(
            language="markdown",
            chunk_size=512,
            chunk_overlap=64,
        )
    )

    chunk_scope = doc_embeddings["chunks"].row_handler()

    # Generate embeddings using a sentence transformer model
    chunk_scope["embedding"] = chunk_scope["text"].transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2",
        ),
        result_type=Annotated[list[float], cocoindex.Vector(dim=384)],
    )

    # Export to a vector store target for ANN search
    doc_embeddings["chunks"].export(
        "doc_chunks",
        cocoindex.storages.Qdrant(
            collection_name="cocoindex_docs",
            url="http://localhost:6333",
        ),
        primary_key_fields=["filename", "location"],
        vector_fields=["embedding"],
    )


def search(query: str, top_k: int = 5) -> list[dict]:
    """Perform a semantic similarity search against the indexed documents.

    Args:
        query: Natural language search query.
        top_k: Number of top results to return.

    Returns:
        List of matching document chunks with scores.
    """
    # Embed the query using the same model used during indexing
    query_handler = cocoindex.query.SimpleSemanticsQueryHandler(
        flow_def=vector_search_pipeline,
        target_name="doc_chunks",
        query_transform_flow=cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2",
        ),
        default_similarity_metric=cocoindex.SimilarityMetric.COSINE,
    )

    results = query_handler.search(query, top_k=top_k)
    return [
        {
            "filename": r.data["filename"],
            "text": r.data["text"],
            "score": r.score,
        }
        for r in results
    ]


def run_pipeline():
    """Update the vector index and run an example search query."""
    print("Updating vector search index...")
    cocoindex.update(vector_search_pipeline)
    print("Index update complete.")

    example_query = "How does cocoindex handle incremental updates?"
    print(f"\nSearching for: '{example_query}'")
    results = search(example_query, top_k=3)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['filename']} (score: {result['score']:.4f})")
        print(f"    {result['text'][:200].strip()}...")


if __name__ == "__main__":
    cocoindex.init()
    run_pipeline()
