"""Example pipeline demonstrating custom transformation functions in cocoindex.

This example shows how to define and use custom transform functions
to process documents with domain-specific logic before indexing.
"""

import re
import cocoindex


@cocoindex.transform_flow()
def extract_metadata(text: str) -> dict:
    """Extract structured metadata from document text.

    Parses common metadata patterns like author, date, and tags
    from document headers or frontmatter.

    Args:
        text: Raw document text content.

    Returns:
        Dictionary containing extracted metadata fields.
    """
    metadata = {
        "author": None,
        "date": None,
        "tags": [],
        "word_count": len(text.split()),
    }

    # Extract author from common patterns like "Author: Name"
    author_match = re.search(r"(?i)author:\s*(.+)", text)
    if author_match:
        metadata["author"] = author_match.group(1).strip()

    # Extract date in ISO format
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if date_match:
        metadata["date"] = date_match.group(0)

    # Extract hashtags as tags
    metadata["tags"] = re.findall(r"#(\w+)", text)

    return metadata


@cocoindex.transform_flow()
def normalize_text(text: str) -> str:
    """Normalize text by cleaning whitespace and standardizing punctuation.

    Args:
        text: Raw input text.

    Returns:
        Cleaned and normalized text string.
    """
    # Collapse multiple whitespace characters
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Normalize unicode dashes to ASCII hyphen
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


@cocoindex.flow_def(name="CustomTransformPipeline")
def custom_transform_pipeline(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """Pipeline that applies custom transformations before indexing.

    Reads documents from a local directory, applies metadata extraction
    and text normalization, then stores results in a vector store.

    Args:
        flow_builder: CocoIndex flow builder instance.
        data_scope: Shared data scope for the pipeline.
    """
    # Source: read markdown documents from a local directory
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(path="docs/", included_patterns=["*.md", "*.txt"])
    )

    doc_embeddings = data_scope["documents"].row_handler(
        lambda doc_scope: _process_document(flow_builder, doc_scope)
    )

    # Sink: store processed documents with embeddings
    doc_embeddings.export(
        "custom_transform_index",
        cocoindex.storages.Postgres(),
        primary_key_fields=["filename"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
    )


def _process_document(flow_builder: cocoindex.FlowBuilder, doc_scope: cocoindex.DataScope):
    """Apply transformations to a single document within the pipeline."""
    # Normalize the raw text first
    doc_scope["normalized_text"] = doc_scope["content"].transform(
        normalize_text
    )

    # Extract structured metadata from the normalized text
    doc_scope["metadata"] = doc_scope["normalized_text"].transform(
        extract_metadata
    )

    # Generate embedding from normalized text
    doc_scope["embedding"] = doc_scope["normalized_text"].transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )


def run_pipeline():
    """Run the custom transform pipeline update."""
    cocoindex.init()
    custom_transform_pipeline.update()
    print("Custom transform pipeline update complete.")


if __name__ == "__main__":
    run_pipeline()
