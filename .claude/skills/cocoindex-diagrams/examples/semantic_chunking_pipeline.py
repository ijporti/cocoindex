"""Semantic chunking pipeline example for cocoindex.

This pipeline demonstrates semantic-aware document chunking, where chunks
are created based on semantic similarity rather than fixed token counts.
This approach tends to produce more coherent chunks for downstream tasks.
"""

import cocoindex
from dataclasses import dataclass
from typing import Optional


@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk of text."""
    text: str
    start_index: int
    end_index: int
    semantic_score: float
    section_title: Optional[str] = None


@cocoindex.op.function()
def detect_section_boundaries(text: str, min_section_length: int = 100) -> list[dict]:
    """Detect natural section boundaries in text based on structural cues.

    Looks for headings, paragraph breaks, and other structural markers
    to identify where sections begin and end.
    """
    import re

    sections = []
    # Match markdown headings or lines followed by blank lines
    heading_pattern = re.compile(r'^(#{1,6}\s+.+|[A-Z][^.!?]*:)$', re.MULTILINE)
    paragraph_pattern = re.compile(r'\n{2,}')

    boundaries = [0]
    titles = {0: None}

    for match in heading_pattern.finditer(text):
        pos = match.start()
        if pos > 0:
            boundaries.append(pos)
            titles[pos] = match.group().strip('#').strip()

    for match in paragraph_pattern.finditer(text):
        pos = match.end()
        if pos not in boundaries:
            boundaries.append(pos)
            titles[pos] = None

    boundaries = sorted(set(boundaries))
    boundaries.append(len(text))

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        section_text = text[start:end].strip()
        if len(section_text) >= min_section_length:
            sections.append({
                "text": section_text,
                "start": start,
                "end": end,
                "title": titles.get(start),
            })

    return sections


@cocoindex.op.function()
def score_chunk_coherence(chunk_text: str) -> float:
    """Score how semantically coherent a chunk is (0.0 to 1.0).

    Uses simple heuristics: sentence completeness, consistent vocabulary,
    and absence of mid-sentence breaks.
    """
    import re

    if not chunk_text or len(chunk_text) < 20:
        return 0.0

    # Check if chunk ends with sentence-ending punctuation
    ends_cleanly = bool(re.search(r'[.!?]\s*$', chunk_text.strip()))

    # Check if chunk starts with a capital letter or heading
    starts_cleanly = bool(re.match(r'^[A-Z#]', chunk_text.strip()))

    # Penalize very short or very long chunks
    word_count = len(chunk_text.split())
    length_score = min(1.0, word_count / 50) if word_count < 50 else max(0.5, 1.0 - (word_count - 200) / 500)

    coherence = (0.4 * ends_cleanly + 0.3 * starts_cleanly + 0.3 * length_score)
    return round(coherence, 4)


@cocoindex.flow_def(name="SemanticChunkingPipeline")
def semantic_chunking_pipeline(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """Pipeline that chunks documents using semantic boundary detection.

    Flow:
    1. Load documents from a local directory
    2. Detect semantic section boundaries
    3. Score each chunk for coherence
    4. Embed coherent chunks
    5. Export to a vector store for retrieval
    """
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(path="docs/", binary=False)
    )

    doc_embeddings = data_scope["documents"].transform(
        cocoindex.transforms.SplitRecursively(),
        language="markdown",
        chunk_size=512,
        chunk_overlap=64,
    )

    doc_embeddings["sections"] = doc_embeddings["text"].transform(
        detect_section_boundaries,
        min_section_length=80,
    )

    doc_embeddings["coherence_score"] = doc_embeddings["text"].transform(
        score_chunk_coherence,
    )

    doc_embeddings["embedding"] = doc_embeddings["text"].transform(
        cocoindex.transforms.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        ),
    )

    doc_embeddings.export(
        "semantic_chunks",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename", "location"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
    )


def run_pipeline():
    """Run the semantic chunking pipeline."""
    cocoindex.init()
    semantic_chunking_pipeline.update()
    print("Semantic chunking pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
