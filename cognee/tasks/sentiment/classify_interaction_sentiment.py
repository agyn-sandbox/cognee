from __future__ import annotations

import asyncio
from typing import List
from uuid import NAMESPACE_OID, uuid5

from pydantic import BaseModel, Field, confloat, field_validator

from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts.read_query_prompt import read_query_prompt
from cognee.modules.retrieval.utils.models import CogneeUserInteraction
from cognee.shared.logging_utils import get_logger

from .models import InteractionSentiment, SentimentLabel


logger = get_logger("classify_interaction_sentiment")

SYSTEM_PROMPT = read_query_prompt("interaction_sentiment_system_prompt.txt")
USER_PROMPT_TEMPLATE = read_query_prompt("interaction_sentiment_user_prompt.txt")


class InteractionSentimentResponse(BaseModel):
    """Structured response expected from the LLM classifier."""

    sentiment: SentimentLabel
    confidence: confloat(ge=0, le=1)
    summary: str = Field(..., min_length=1)

    @field_validator("sentiment", mode="before")
    @classmethod
    def _normalize_sentiment(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value):
        if isinstance(value, str):
            return float(value.strip())
        return value

    @field_validator("summary", mode="before")
    @classmethod
    def _trim_summary(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value


def _render_prompt(interaction: CogneeUserInteraction) -> str:
    return USER_PROMPT_TEMPLATE.format(
        question=interaction.question,
        answer=interaction.answer,
        context=interaction.context,
    )


def _create_sentiment_datapoint(
    interaction: CogneeUserInteraction, response: InteractionSentimentResponse
) -> InteractionSentiment:
    sentiment_id = uuid5(NAMESPACE_OID, f"interaction-sentiment-{interaction.id}")
    return InteractionSentiment(
        id=sentiment_id,
        interaction_id=interaction.id,
        sentiment=response.sentiment,
        confidence=float(response.confidence),
        summary=response.summary,
    )


async def _classify_single_interaction(
    interaction: CogneeUserInteraction,
) -> InteractionSentiment | None:
    prompt = _render_prompt(interaction)
    try:
        response = await LLMGateway.acreate_structured_output(
            text_input=prompt,
            system_prompt=SYSTEM_PROMPT,
            response_model=InteractionSentimentResponse,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Sentiment classification failed",
            interaction_id=str(interaction.id),
            error=str(exc),
        )
        return None

    try:
        return _create_sentiment_datapoint(interaction, response)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to build InteractionSentiment datapoint",
            interaction_id=str(interaction.id),
            error=str(exc),
        )
        return None


async def classify_interaction_sentiment(
    interactions: List[CogneeUserInteraction],
) -> List[InteractionSentiment]:
    """Classify sentiment for recent interactions using the configured LLM."""

    if not interactions:
        logger.info("No interactions supplied for sentiment analysis")
        return []

    tasks = [_classify_single_interaction(interaction) for interaction in interactions]
    results = await asyncio.gather(*tasks)

    sentiments = [result for result in results if result is not None]
    logger.info(
        "Classified interaction sentiment", total=len(interactions), classified=len(sentiments)
    )
    return sentiments
