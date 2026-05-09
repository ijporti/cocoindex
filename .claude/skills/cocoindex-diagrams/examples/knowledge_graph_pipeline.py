"""Knowledge graph pipeline example for cocoindex.

Demonstrates building a knowledge graph from documents by extracting
entities and relationships, then storing them in a graph-compatible format.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

import cocoindex


@dataclass
class Entity:
    """Represents a named entity extracted from text."""
    name: str
    entity_type: str  # PERSON, ORG, LOCATION, CONCEPT
    context: str


@dataclass
class Relationship:
    """Represents a relationship between two entities."""
    source: str
    relation: str
    target: str
    confidence: float


@cocoindex.op.function()
def extract_entities(text: str) -> List[Entity]:
    """Extract named entities from text using simple heuristics.

    In production, replace with an NLP model (spaCy, HuggingFace NER, etc.).
    """
    entities: List[Entity] = []

    # Simple pattern-based extraction as a placeholder for real NER
    # Matches capitalized phrases as potential entities
    pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
    matches = pattern.finditer(text)

    for match in matches:
        name = match.group(1)
        start = max(0, match.start() - 40)
        end = min(len(text), match.end() + 40)
        context = text[start:end].strip()

        # Naive type classification based on common patterns
        if any(title in name for title in ["Inc", "Corp", "Ltd", "LLC"]):
            entity_type = "ORG"
        elif name.split()[0] in ["Dr", "Mr", "Ms", "Prof"]:
            entity_type = "PERSON"
        else:
            entity_type = "CONCEPT"

        entities.append(Entity(name=name, entity_type=entity_type, context=context))

    # Deduplicate by name
    seen: set = set()
    unique: List[Entity] = []
    for e in entities:
        if e.name not in seen:
            seen.add(e.name)
            unique.append(e)

    return unique[:20]  # Limit to top 20 entities per document


@cocoindex.op.function()
def extract_relationships(
    text: str, entities: List[Entity]
) -> List[Relationship]:
    """Extract relationships between entities found in the same sentence."""
    relationships: List[Relationship] = []
    entity_names = {e.name for e in entities}

    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        found_in_sentence = [n for n in entity_names if n in sentence]
        if len(found_in_sentence) >= 2:
            # Create co-occurrence relationships
            for i, src in enumerate(found_in_sentence):
                for tgt in found_in_sentence[i + 1:]:
                    relationships.append(
                        Relationship(
                            source=src,
                            relation="CO_OCCURS_WITH",
                            target=tgt,
                            confidence=0.7,
                        )
                    )

    return relationships


@cocoindex.flow_def(name="KnowledgeGraphPipeline")
def knowledge_graph_pipeline(
    flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
) -> None:
    """Build a knowledge graph from a document corpus.

    Reads documents, extracts entities and relationships, and stores
    them in a structured format suitable for graph databases.
    """
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="docs/",
            included_patterns=["*.txt", "*.md"],
        )
    )

    docs = data_scope["documents"]

    with docs.row() as doc:
        doc["entities"] = doc["content"].transform(
            extract_entities,
        )
        doc["relationships"] = doc["content"].transform(
            extract_relationships,
            doc["entities"],
        )

        # Export entity nodes
        doc["entities"].save(
            cocoindex.storages.Postgres(
                table_name="kg_entities",
                primary_key_fields=["name", "entity_type"],
            )
        )

        # Export relationship edges
        doc["relationships"].save(
            cocoindex.storages.Postgres(
                table_name="kg_relationships",
                primary_key_fields=["source", "relation", "target"],
            )
        )


def run_pipeline() -> None:
    """Run the knowledge graph pipeline."""
    cocoindex.init()
    knowledge_graph_pipeline.run()
    print("Knowledge graph pipeline complete.")
    print("Entities and relationships stored in Postgres tables.")


if __name__ == "__main__":
    run_pipeline()
