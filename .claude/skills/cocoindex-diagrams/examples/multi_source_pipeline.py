"""Multi-source pipeline example for cocoindex.

Demonstrates how to ingest documents from multiple sources (local files,
S3, and a database), apply transformations, and index into a vector store.
"""

import cocoindex
from cocoindex.sources import LocalFileSource, S3Source, PostgresSource
from cocoindex.transforms import (
    TextSplitter,
    EmbeddingTransform,
    MetadataExtractor,
)
from cocoindex.sinks import PineconeVectorSink


@cocoindex.pipeline
def multi_source_pipeline(
    local_dir: str = "./docs",
    s3_bucket: str = "my-docs-bucket",
    s3_prefix: str = "documents/",
    db_table: str = "articles",
    index_name: str = "multi-source-index",
) -> None:
    """Pipeline that ingests from local files, S3, and Postgres.

    Args:
        local_dir: Path to local directory containing documents.
        s3_bucket: Name of the S3 bucket to read from.
        s3_prefix: Key prefix to filter S3 objects.
        db_table: Postgres table name containing article text.
        index_name: Target Pinecone index name.
    """
    # --- Sources ---
    local_docs = cocoindex.source(
        LocalFileSource(
            path=local_dir,
            glob="**/*.{md,txt,pdf}",
            recursive=True,
        ),
        name="local_files",
    )

    s3_docs = cocoindex.source(
        S3Source(
            bucket=s3_bucket,
            prefix=s3_prefix,
            file_types=["pdf", "docx", "txt"],
        ),
        name="s3_files",
    )

    db_docs = cocoindex.source(
        PostgresSource(
            table=db_table,
            text_column="body",
            id_column="article_id",
            metadata_columns=["title", "author", "published_at"],
        ),
        name="postgres_articles",
    )

    # --- Merge all sources into a unified stream ---
    all_docs = cocoindex.merge(
        local_docs,
        s3_docs,
        db_docs,
        on_conflict="latest",  # keep most recently updated version
    )

    # --- Transformations ---
    # Extract metadata (title, author, date) from raw content where missing
    enriched = cocoindex.transform(
        all_docs,
        MetadataExtractor(
            fields=["title", "author", "source_type"],
            fallback_to_filename=True,
        ),
    )

    # Split into chunks suitable for embedding
    chunks = cocoindex.transform(
        enriched,
        TextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            split_by="sentence",
        ),
    )

    # Generate dense embeddings
    embedded = cocoindex.transform(
        chunks,
        EmbeddingTransform(
            model="text-embedding-3-small",
            batch_size=64,
            dimensions=1536,
        ),
    )

    # --- Sink ---
    cocoindex.sink(
        embedded,
        PineconeVectorSink(
            index_name=index_name,
            namespace="multi-source",
            metadata_fields=["title", "author", "source_type", "chunk_index"],
        ),
    )


def run_pipeline() -> None:
    """Entry point for running the multi-source pipeline locally."""
    multi_source_pipeline(
        local_dir="./sample_docs",
        s3_bucket="cocoindex-demo",
        s3_prefix="examples/",
        db_table="demo_articles",
        index_name="cocoindex-multi-source-demo",
    )


if __name__ == "__main__":
    run_pipeline()
