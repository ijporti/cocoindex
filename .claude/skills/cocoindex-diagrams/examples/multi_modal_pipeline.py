"""Multi-modal pipeline example for cocoindex.

Demonstrates processing both text and image content from documents,
generating embeddings for each modality and storing them in a unified index.
"""

import cocoindex
from pathlib import Path


@cocoindex.op.function()
def extract_text_content(raw_bytes: bytes, filename: str) -> str:
    """Extract text content from a document based on file type."""
    suffix = Path(filename).suffix.lower()
    if suffix in (".txt", ".md"):
        return raw_bytes.decode("utf-8", errors="replace")
    # For other types, return empty string (would use real parsers in production)
    return ""


@cocoindex.op.function()
def extract_image_paths(raw_bytes: bytes, filename: str) -> list[str]:
    """Extract image references from a document.

    In a real implementation this would parse HTML/Markdown/PDF
    and return embedded image paths or base64 blobs.
    """
    import re

    text = raw_bytes.decode("utf-8", errors="replace")
    # Simple regex to find markdown image references
    pattern = r"!\[.*?\]\((.+?)\)"
    return re.findall(pattern, text)


@cocoindex.op.function()
def classify_modality(filename: str) -> str:
    """Classify document as text, image, or mixed based on extension."""
    suffix = Path(filename).suffix.lower()
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
    text_extensions = {".txt", ".md", ".rst", ".html", ".htm"}

    if suffix in image_extensions:
        return "image"
    elif suffix in text_extensions:
        return "text"
    return "mixed"


@cocoindex.flow_def(name="MultiModalPipeline")
def multi_modal_pipeline(
    flow_builder: cocoindex.FlowBuilder,
    data_scope: cocoindex.DataScope,
) -> None:
    """Pipeline that handles both text and image modalities.

    Sources:
        - Local filesystem directory with mixed document types

    Transformations:
        - Classify each file by modality
        - Extract text content for text/mixed files
        - Chunk text into segments
        - Generate text embeddings for text segments
        - Track image references for downstream processing

    Targets:
        - Vector store with text embeddings
        - Metadata store with modality classification
    """
    # Ingest files from a local directory
    data_scope["files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="docs/",
            included_patterns=["*.md", "*.txt", "*.png", "*.jpg"],
        )
    )

    files = data_scope["files"]

    # Classify each file by modality
    files["modality"] = files["filename"].transform(
        classify_modality,
        cocoindex.Field(name="filename"),
    )

    # Extract text content
    files["text_content"] = files.transform(
        extract_text_content,
        cocoindex.Field(name="content"),
        cocoindex.Field(name="filename"),
    )

    # Extract image references from documents
    files["image_refs"] = files.transform(
        extract_image_paths,
        cocoindex.Field(name="content"),
        cocoindex.Field(name="filename"),
    )

    # Chunk text content for embedding
    files["chunks"] = files["text_content"].transform(
        cocoindex.functions.SentenceSplitter(chunk_size=512, chunk_overlap=64),
    )

    with files["chunks"].row() as chunk:
        # Generate embeddings for each text chunk
        chunk["embedding"] = chunk["text"].transform(
            cocoindex.functions.SentenceTransformerEmbed(
                model="sentence-transformers/all-MiniLM-L6-v2"
            ),
        )

        # Export to vector store
        chunk.export(
            "text_embeddings",
            cocoindex.storages.Qdrant(
                collection_name="multi_modal_docs",
                vector_field="embedding",
            ),
            primary_key_fields=["filename", "chunk_index"],
        )


def run_pipeline() -> None:
    """Run the multi-modal pipeline update."""
    cocoindex.init()
    multi_modal_pipeline.update()
    print("Multi-modal pipeline update complete.")


if __name__ == "__main__":
    run_pipeline()
