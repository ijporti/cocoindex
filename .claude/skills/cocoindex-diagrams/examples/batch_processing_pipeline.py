"""Batch processing pipeline example for cocoindex.

Demonstrates how to process large document collections in batches,
with progress tracking, error handling, and result aggregation.
"""

from dataclasses import dataclass, field
from typing import Iterator
import cocoindex


@dataclass
class BatchResult:
    """Result from processing a single batch."""
    batch_id: int
    processed: int
    failed: int
    errors: list[str] = field(default_factory=list)


@cocoindex.op.function()
def split_into_batches(documents: list[str], batch_size: int = 32) -> list[list[str]]:
    """Split a flat list of documents into fixed-size batches."""
    return [
        documents[i : i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]


@cocoindex.op.function()
def preprocess_document(text: str) -> str:
    """Normalise whitespace and strip leading/trailing space."""
    import re
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@cocoindex.op.function()
def process_batch(batch: list[str]) -> BatchResult:
    """Process a single batch of documents, collecting per-item errors."""
    result = BatchResult(batch_id=id(batch), processed=0, failed=0)
    for doc in batch:
        try:
            if not doc or not doc.strip():
                raise ValueError("Empty document")
            # Simulate per-document work (e.g. extraction, enrichment).
            _ = preprocess_document(doc)
            result.processed += 1
        except Exception as exc:  # noqa: BLE001
            result.failed += 1
            result.errors.append(str(exc))
    return result


@cocoindex.op.function()
def aggregate_results(results: list[BatchResult]) -> dict:
    """Merge per-batch results into a single summary dict."""
    total_processed = sum(r.processed for r in results)
    total_failed = sum(r.failed for r in results)
    all_errors = [err for r in results for err in r.errors]
    return {
        "batches": len(results),
        "total_processed": total_processed,
        "total_failed": total_failed,
        "success_rate": (
            total_processed / (total_processed + total_failed)
            if (total_processed + total_failed) > 0
            else 0.0
        ),
        "errors": all_errors[:10],  # cap error list for readability
    }


@cocoindex.flow_def(name="BatchProcessingPipeline")
def batch_processing_pipeline(
    flow_builder: cocoindex.FlowBuilder,
    data_scope: cocoindex.DataScope,
) -> None:
    """Define a batch-processing pipeline.

    Steps:
    1. Ingest raw documents from a local directory source.
    2. Collect them into fixed-size batches.
    3. Process each batch independently.
    4. Aggregate all batch results into a summary.
    """
    # --- Source ---
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(path="data/documents", binary=False)
    )

    # --- Batch splitting ---
    data_scope["batches"] = data_scope["documents"].transform(
        split_into_batches,
        cocoindex.Field("content"),
        batch_size=32,
    )

    # --- Per-batch processing ---
    data_scope["batch_results"] = data_scope["batches"].transform(
        process_batch,
        cocoindex.Field("batches"),
    )

    # --- Aggregation ---
    data_scope["summary"] = data_scope["batch_results"].transform(
        aggregate_results,
        cocoindex.Field("batch_results"),
    )

    # --- Export summary to a key-value store ---
    data_scope["summary"].export(
        "batch_summary",
        cocoindex.storages.Postgres(),
        primary_key_fields=["batches"],
    )


def run_pipeline() -> None:
    """Entry point: update the pipeline and print the run summary."""
    cocoindex.init()
    stats = cocoindex.flow_run_stats(batch_processing_pipeline)
    print(f"Pipeline run complete: {stats}")


if __name__ == "__main__":
    run_pipeline()
