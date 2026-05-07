"""Example pipeline demonstrating text chunking and embedding with cocoindex.

This example shows how to build a pipeline that:
1. Reads text documents from a source
2. Splits them into semantic chunks
3. Generates embeddings for each chunk
4. Stores results in a vector store
"""

import cocoindex
from cocoindex import (
    DataSource,
    Pipeline,
    PipelineContext,
    transformers,
)


@cocoindex.flow_def(name="TextChunkingPipeline")
def text_chunking_pipeline(
    flow_builder: cocoindex.FlowBuilder,
    data_scope: cocoindex.DataScope,
) -> None:
    """Define a text chunking and embedding pipeline.

    Args:
        flow_builder: Builder object for constructing the pipeline flow.
        data_scope: Scope containing data sources and sinks.
    """
    # Define the document source
    documents = flow_builder.add_source(
        cocoindex.sources.LocalFileSource(
            path="./data/documents",
            file_extensions=[".txt", ".md", ".rst"],
        )
    )

    # Extract text content from each document
    text_content = documents.transform(
        transformers.ExtractText(
            encoding="utf-8",
            fallback_encoding="latin-1",
        )
    )

    # Split text into overlapping chunks for better context preservation
    chunks = text_content.transform(
        transformers.SplitRecursively(
            language="markdown",
            chunk_size=512,
            chunk_overlap=64,
        )
    )

    # Generate embeddings for each chunk using a local or remote model
    embeddings = chunks.transform(
        transformers.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
        )
    )

    # Export results to a vector store
    embeddings.export(
        cocoindex.targets.QdrantTarget(
            collection_name="document_chunks",
            vector_size=384,
            distance="Cosine",
        ),
        primary_key_fields=["filename", "chunk_index"],
        vector_field="embedding",
    )


def run_pipeline() -> None:
    """Run the text chunking pipeline with default settings."""
    # Initialize cocoindex with local settings
    cocoindex.init(
        cocoindex.Settings(
            database_url="postgresql://localhost/cocoindex",
        )
    )

    # Execute the pipeline
    cocoindex.utils.simple_indexing_main(text_chunking_pipeline)


if __name__ == "__main__":
    run_pipeline()
