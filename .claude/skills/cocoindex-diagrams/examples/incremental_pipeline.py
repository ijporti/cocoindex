"""Example: Incremental update pipeline with change detection.

Demonstrates how to build a pipeline that efficiently processes
only new or modified documents using CocoIndex's incremental
update capabilities.
"""

import cocoindex
from datetime import datetime
from pathlib import Path


@cocoindex.flow_def(name="IncrementalDocumentPipeline")
def incremental_document_pipeline(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """Pipeline that tracks document modifications and processes incrementally.

    Only re-processes documents that have changed since the last run,
    making it efficient for large document collections that update frequently.
    """
    # Source: watch a local directory for file changes
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="docs/",
            included_patterns=["*.md", "*.txt", "*.rst"],
            binary=False,
        )
    )

    doc_embeddings = data_scope["documents"].row_handler(
        cocoindex.RowHandler()
    )

    with doc_embeddings.fixed_field_handler("filename", "content") as doc:
        # Split document into chunks for embedding
        doc["chunks"] = doc["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language="markdown",
            chunk_size=512,
            chunk_overlap=64,
        )

        with doc["chunks"].row_handler() as chunk:
            # Generate embeddings for each chunk
            chunk["embedding"] = chunk["text"].transform(
                cocoindex.functions.SentenceTransformerEmbed(
                    model="sentence-transformers/all-MiniLM-L6-v2"
                )
            )

    # Export to a vector store with upsert semantics
    doc_embeddings.export(
        "doc_embeddings",
        cocoindex.storages.Qdrant(
            collection_name="incremental_docs",
            url="http://localhost:6333",
        ),
        primary_key_fields=["filename", "location"],
        vector_fields=["embedding"],
    )


def run_pipeline(update_mode: str = "update"):
    """Run the incremental pipeline.

    Args:
        update_mode: One of 'update' (incremental), 'full' (reprocess all),
                     or 'live' (continuous watching mode).
    """
    cocoindex.init()

    flow = incremental_document_pipeline()

    if update_mode == "full":
        print("Running full reindex...")
        flow.run()
        print("Full reindex complete.")

    elif update_mode == "update":
        print("Running incremental update...")
        start = datetime.now()
        stats = flow.update()
        elapsed = (datetime.now() - start).total_seconds()
        print(f"Incremental update complete in {elapsed:.2f}s")
        if stats:
            print(f"  Added:   {stats.added}")
            print(f"  Updated: {stats.updated}")
            print(f"  Removed: {stats.removed}")

    elif update_mode == "live":
        print("Starting live update mode (Ctrl+C to stop)...")
        flow.live_update(interval_seconds=30)

    else:
        raise ValueError(f"Unknown update_mode: {update_mode!r}")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "update"
    run_pipeline(update_mode=mode)
