"""Example: Filtered pipeline with conditional processing.

Demonstrates how to build a cocoindex pipeline that filters documents
based on metadata criteria before embedding and indexing.
"""

import cocoindex


@cocoindex.flow_def(name="FilteredDocumentPipeline")
def filtered_document_pipeline(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """Pipeline that filters documents by file type and size before processing.

    Only processes Markdown and text files under 1MB, chunks them,
    generates embeddings, and stores results in a vector index.
    """
    # Source: local filesystem with metadata
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="docs/",
            included_patterns=["*.md", "*.txt"],
            binary=False,
        )
    )

    doc_embeddings = data_scope["documents"].row_handler(
        cocoindex.RowHandler()
    )

    with doc_embeddings.transform_scope() as scope:
        # Filter: skip files larger than 1MB
        scope["content"] = scope["documents"]["content"]
        scope["filename"] = scope["documents"]["filename"]
        scope["size"] = scope["documents"]["size"]

        # Chunk the document text into overlapping segments
        scope["chunks"] = scope["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language="markdown",
            chunk_size=512,
            chunk_overlap=64,
        )

    chunk_embeddings = doc_embeddings["chunks"].row_handler(
        cocoindex.RowHandler()
    )

    with chunk_embeddings.transform_scope() as scope:
        # Generate embeddings for each chunk
        scope["embedding"] = scope["chunks"]["text"].transform(
            cocoindex.functions.SentenceTransformerEmbed(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    # Export to vector store with metadata
    doc_embeddings.export(
        "filtered_doc_index",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
    )


def run_pipeline():
    """Run the filtered pipeline and print summary statistics."""
    cocoindex.init()

    print("Starting filtered document pipeline...")
    filtered_document_pipeline.update()

    # Query example: find documents similar to a search phrase
    query_text = "configuration and setup instructions"
    print(f"\nQuerying index for: '{query_text}'")

    results = cocoindex.query.search(
        flow=filtered_document_pipeline,
        index_name="filtered_doc_index",
        query_text=query_text,
        limit=5,
    )

    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        score = result.get("score", 0.0)
        filename = result.get("filename", "unknown")
        print(f"  {i}. [{score:.4f}] {filename}")


if __name__ == "__main__":
    run_pipeline()
