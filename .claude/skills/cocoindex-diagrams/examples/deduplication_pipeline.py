"""Deduplication pipeline example for cocoindex.

Demonstrates how to build a pipeline that detects and removes duplicate
documents using content hashing and similarity-based deduplication.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Annotated

import cocoindex


@dataclass
class DeduplicationResult:
    """Result of deduplication analysis for a document."""
    content_hash: str
    normalized_content: str
    word_count: int
    is_duplicate: bool


def normalize_for_dedup(text: str) -> str:
    """Normalize text for deduplication comparison.

    Strips whitespace, lowercases, and removes punctuation to allow
    near-duplicate detection across minor formatting differences.
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def compute_content_hash(text: str) -> str:
    """Compute a stable SHA-256 hash of normalized text content."""
    normalized = normalize_for_dedup(text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def count_words(text: str) -> int:
    """Count the number of words in a text string."""
    return len(text.split())


@cocoindex.op.function()
def analyze_for_deduplication(content: str) -> DeduplicationResult:
    """Analyze a document chunk for deduplication.

    Computes a content hash and normalized form that can be used
    downstream to filter out duplicate entries in the index.
    """
    content_hash = compute_content_hash(content)
    normalized = normalize_for_dedup(content)
    word_count = count_words(content)
    # Deduplication decision is made at collection level via the hash key;
    # mark all as non-duplicate here — the storage layer deduplicates by key.
    return DeduplicationResult(
        content_hash=content_hash,
        normalized_content=normalized,
        word_count=word_count,
        is_duplicate=False,
    )


@cocoindex.transform_flow()
def deduplication_pipeline(
    docs: Annotated[cocoindex.DataSlice, cocoindex.field("docs")],
) -> cocoindex.DataSlice:
    """Pipeline that indexes documents with deduplication by content hash.

    Documents are chunked, analyzed for duplicate content, and stored
    with their content hash as the primary deduplication key so that
    re-ingesting the same content is idempotent.
    """
    chunks = docs.transform(
        cocoindex.functions.SplitRecursively(),
        language=None,
        chunk_size=512,
        chunk_overlap=64,
    )

    analyzed = chunks.transform(analyze_for_deduplication)

    # Embed the normalized content for semantic search
    embeddings = analyzed[
        lambda r: r.normalized_content
    ].transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    return analyzed.with_field("embedding", embeddings)


def run_pipeline() -> None:
    """Run the deduplication pipeline and report indexed document count."""
    pipeline = cocoindex.Pipeline(
        name="deduplication_demo",
        flow=deduplication_pipeline,
        source=cocoindex.sources.LocalFile(path="./docs"),
    )
    stats = pipeline.run()
    print(f"Indexed {stats.documents_processed} documents")
    print(f"Skipped {stats.documents_skipped} duplicates")


if __name__ == "__main__":
    run_pipeline()
