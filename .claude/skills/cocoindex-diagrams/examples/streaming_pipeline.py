"""Streaming pipeline example for cocoindex.

Demonstrates how to build a pipeline that processes documents
in a streaming fashion, useful for large datasets or real-time
ingestion from queues and event streams.
"""

import cocoindex
from dataclasses import dataclass
from typing import Generator


@dataclass
class StreamRecord:
    """Represents a single record from a stream source."""
    id: str
    content: str
    source: str
    timestamp: str


@cocoindex.transform_flow()
def parse_stream_record(record: cocoindex.DataSlice) -> cocoindex.DataSlice:
    """Parse and validate an incoming stream record."""
    return record.transform(
        cocoindex.functions.JsonParse(),
        output_type=StreamRecord,
    )


@cocoindex.transform_flow()
def enrich_record(record: cocoindex.DataSlice) -> cocoindex.DataSlice:
    """Enrich a stream record with additional metadata."""
    return record.transform(
        cocoindex.functions.ExtractByLine(),
    )


@cocoindex.flow_def(name="StreamingPipeline")
def streaming_pipeline(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """Streaming pipeline that ingests, processes, and indexes records in real time.

    Pipeline stages:
      1. Source  - reads from a streaming source (e.g. Kafka, file tail)
      2. Parse   - deserialises raw bytes into structured records
      3. Chunk   - splits long content into overlapping text chunks
      4. Embed   - generates vector embeddings for each chunk
      5. Export  - writes chunks + vectors to a vector store
    """
    # ---------------------------------------------------------------------------
    # Stage 1: Source
    # ---------------------------------------------------------------------------
    data_scope["stream_records"] = flow_builder.add_source(
        cocoindex.sources.FileSource(
            path="data/stream/",
            # In a real deployment this would be a KafkaSource or similar.
            # FileSource is used here so the example runs without extra deps.
            binary=False,
        )
    )

    # ---------------------------------------------------------------------------
    # Stage 2: Parse raw content into structured chunks
    # ---------------------------------------------------------------------------
    records = data_scope["stream_records"]

    records["chunks"] = records["content"].transform(
        cocoindex.functions.SplitRecursively(),
        language="markdown",
        chunk_size=512,
        chunk_overlap=64,
    )

    # ---------------------------------------------------------------------------
    # Stage 3: Embed each chunk
    # ---------------------------------------------------------------------------
    with records["chunks"].row() as chunk:
        chunk["embedding"] = chunk["text"].transform(
            cocoindex.functions.SentenceTransformerEmbed(
                model="sentence-transformers/all-MiniLM-L6-v2",
            )
        )

    # ---------------------------------------------------------------------------
    # Stage 4: Export to vector store
    # ---------------------------------------------------------------------------
    records["chunks"].export(
        "stream_chunks",
        cocoindex.storages.QdrantStorage(
            collection_name="streaming_pipeline",
            vector_field="embedding",
        ),
        primary_key_fields=["filename", "location"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE,
            )
        ],
    )


def run_pipeline():
    """Run the streaming pipeline with live update mode."""
    cocoindex.init()

    # update() processes new/changed files incrementally.
    # In production, pair this with a scheduler or event trigger.
    with cocoindex.FlowLiveUpdater(streaming_pipeline) as updater:
        print("Streaming pipeline running — press Ctrl+C to stop.")
        updater.wait()


if __name__ == "__main__":
    run_pipeline()
