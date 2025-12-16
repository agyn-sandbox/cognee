from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import Field, confloat

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import NodeSet


class SentimentLabel(str, Enum):
    """Canonical sentiment labels supported by the interaction classifier."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class InteractionSentiment(DataPoint):
    """Represents the sentiment associated with a user interaction."""

    interaction_id: UUID = Field(..., description="Source CogneeUserInteraction identifier")
    sentiment: SentimentLabel
    confidence: confloat(ge=0, le=1)
    summary: str = ""
    belongs_to_set: Optional[NodeSet] = None
    metadata: dict = {"index_fields": []}
