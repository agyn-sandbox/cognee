from __future__ import annotations

from typing import Any, List, Optional
from uuid import UUID

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.utils.models import CogneeUserInteraction
from cognee.shared.logging_utils import get_logger


logger = get_logger("extract_recent_interactions")

DEFAULT_LAST_K = 20


def _normalize_limit(last_k: Optional[int]) -> int:
    if last_k is None:
        return DEFAULT_LAST_K
    if last_k <= 0:
        raise ValueError("last_k must be a positive integer")
    return last_k


async def extract_recent_interactions(
    data: Any, last_k: Optional[int] = DEFAULT_LAST_K
) -> List[CogneeUserInteraction]:
    """Fetch the most recent CogneeUserInteraction records from the graph."""

    limit = _normalize_limit(last_k)
    graph_engine = await get_graph_engine()

    interaction_ids = await graph_engine.get_last_user_interaction_ids(limit=limit)
    if not interaction_ids:
        logger.info("No recent interactions found", limit=limit)
        return []

    raw_nodes = await graph_engine.get_nodes(interaction_ids)
    nodes_by_id = {str(node.get("id")): node for node in raw_nodes if node.get("id")}

    interactions: List[CogneeUserInteraction] = []

    for interaction_id in interaction_ids:
        node = nodes_by_id.get(str(interaction_id))
        if not node:
            logger.warning("Interaction node missing", interaction_id=interaction_id)
            continue

        question = node.get("question") or ""
        answer = node.get("answer") or ""
        context = node.get("context") or ""

        try:
            interactions.append(
                CogneeUserInteraction(
                    id=UUID(str(interaction_id)),
                    question=str(question),
                    answer=str(answer),
                    context=str(context),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to hydrate interaction", interaction_id=interaction_id, error=str(exc)
            )

    logger.info("Loaded recent interactions", fetched=len(interactions), requested=limit)
    return interactions
