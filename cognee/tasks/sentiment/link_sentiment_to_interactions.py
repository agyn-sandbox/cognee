from __future__ import annotations

from typing import List, Tuple

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage import index_graph_edges

from .models import InteractionSentiment


logger = get_logger("link_sentiment_to_interactions")

RELATIONSHIP_NAME = "has_sentiment"


def _build_edge(sentiment: InteractionSentiment) -> Tuple:
    return (
        sentiment.id,
        sentiment.interaction_id,
        RELATIONSHIP_NAME,
        {
            "relationship_name": RELATIONSHIP_NAME,
            "source_node_id": sentiment.id,
            "target_node_id": sentiment.interaction_id,
            "ontology_valid": False,
        },
    )


async def link_sentiment_to_interactions(
    sentiments: List[InteractionSentiment],
) -> List[InteractionSentiment]:
    """Create has_sentiment edges between InteractionSentiment and CogneeUserInteraction nodes."""

    if not sentiments:
        logger.info("No sentiment datapoints provided for linking")
        return []

    edges = [_build_edge(sentiment) for sentiment in sentiments]

    graph_engine = await get_graph_engine()
    await graph_engine.add_edges(edges)
    await index_graph_edges(edges)

    logger.info("Linked sentiments to interactions", edge_count=len(edges))
    return sentiments
