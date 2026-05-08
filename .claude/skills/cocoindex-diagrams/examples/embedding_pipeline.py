"""Example pipeline demonstrating embedding generation and vector storage with CocoIndex.

This example shows how to:
- Load documents from a directory
- Split text into chunks
- Generate embeddings using a sentence transformer model
- Store embeddings in a vector store for similarity search
"""

import cocoindex
from pathlib import Path


@cocoindex.flow_def(name="EmbeddingPipeline")
def embedding_pipeline(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """Pipeline that generates embeddings from text documents and stores them for retrieval.

    Args:
        flow_builder: CocoIndex flow builder for constructing the pipeline DAG.
        data_scope: Data scope for managing intermediate and output data.
    """
    # Source: Read markdown and text files from a docs directory
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="docs/",
            included_patterns=["*.md", "*.txt"],
            binary=False,
        )
    )

    doc_embeddings = data_scope["documents"].row_handler(
        "doc_embeddings",
        cocoindex.DataType.KTable(
            cocoindex.DataType.Record({
                "chunk_id": cocoindex.DataType.Str,
                "text": cocoindex.DataType.Str,
                "embedding": cocoindex.DataType.Vector(dim=384),
                "source_file": cocoindex.DataType.Str,
                "chunk_index": cocoindex.DataType.Int64,
            })
        ),
    )

    @doc_embeddings.transform
    def process_document(doc, collector, context):
        """Split document into chunks and generate embeddings for each chunk."""
        filename = context.key
        content = doc["content"]

        # Split content into overlapping chunks
        chunks = cocoindex.functions.SentenceChunker(
            chunk_size=256,
            chunk_overlap=32,
        )(content)

        for idx, chunk_text in enumerate(chunks):
            # Generate embedding for the chunk
            embedding = cocoindex.functions.SentenceTransformerEmbed(
                model="all-MiniLM-L6-v2",
            )(chunk_text)

            collector.collect({
                "chunk_id": f"{filename}::{idx}",
                "text": chunk_text,
                "embedding": embedding,
                "source_file": filename,
                "chunk_index": idx,
            })

    # Export embeddings to a vector store (Qdrant)
    doc_embeddings.export(
        "vector_store",
        cocoindex.targets.Qdrant(
            collection_name="document_embeddings",
            vector_field="embedding",
            payload_fields=["text", "source_file", "chunk_index"],
        ),
        primary_key_fields=["chunk_id"],
    )


def run_pipeline():
    """Run the embedding pipeline and report statistics."""
    cocoindex.init()

    print("Starting embedding pipeline...")
    stats = cocoindex.update(embedding_pipeline)

    print(f"Pipeline complete.")
    print(f"  Documents processed: {stats.get('documents_processed', 0)}")
    print(f"  Chunks embedded:     {stats.get('rows_exported', 0)}")
    print(f"  Duration:            {stats.get('duration_ms', 0):.1f}ms")


if __name__ == "__main__":
    run_pipeline()
