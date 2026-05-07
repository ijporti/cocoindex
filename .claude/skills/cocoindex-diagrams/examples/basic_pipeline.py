"""Basic pipeline example demonstrating cocoindex diagram skill usage.

This example shows how to define a simple document indexing pipeline
and generate architecture diagrams using the cocoindex-diagrams skill.
"""

import cocoindex


@cocoindex.flow_def(name="BasicDocumentPipeline")
def basic_document_pipeline(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """A basic document ingestion and indexing pipeline.

    This pipeline:
    1. Reads documents from a local directory
    2. Splits them into chunks
    3. Generates embeddings
    4. Stores results in a vector database
    """
    # Source: read markdown files from local filesystem
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="./docs",
            included_patterns=["*.md", "*.txt"],
        )
    )

    doc_embeddings = data_scope["documents"].transform(
        cocoindex.functions.SplitRecursively(),
        language="markdown",
        chunk_size=512,
        chunk_overlap=64,
    )

    doc_embeddings = doc_embeddings.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        ),
        field_name="content",
    )

    # Sink: store embeddings in Qdrant vector database
    doc_embeddings.export(
        "doc_embeddings",
        cocoindex.storages.Qdrant(
            collection_name="documents",
            url="http://localhost:6333",
        ),
        primary_key_fields=["filename", "chunk_index"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE,
            )
        ],
    )


if __name__ == "__main__":
    # Run the pipeline update
    cocoindex.init()
    basic_document_pipeline.update()
    print("Pipeline update complete.")
