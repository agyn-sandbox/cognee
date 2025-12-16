import importlib
import uuid

import pytest

from cognee.modules.retrieval.utils.models import CogneeUserInteraction
from cognee.tasks.sentiment.classify_interaction_sentiment import InteractionSentimentResponse
from cognee.tasks.sentiment.models import InteractionSentiment, SentimentLabel


class _FakeGraphEngine:
    def __init__(self, nodes):
        self._nodes = nodes
        self.requested_limit = None
        self.requested_ids = None

    async def get_last_user_interaction_ids(self, limit: int):
        self.requested_limit = limit
        return [node["id"] for node in self._nodes]

    async def get_nodes(self, node_ids):
        self.requested_ids = node_ids
        return [{"id": item["id"], **item} for item in self._nodes]


@pytest.mark.asyncio
async def test_extract_recent_interactions_returns_hydrated_models(monkeypatch):
    interaction_id = str(uuid.uuid4())
    fake_engine = _FakeGraphEngine(
        [
            {
                "id": interaction_id,
                "question": "What is Cognee?",
                "answer": "Cognee builds AI memory.",
                "context": "Some detailed context.",
            }
        ]
    )

    async def _fake_get_graph_engine():
        return fake_engine

    extract_module = importlib.import_module(
        "cognee.tasks.sentiment.extract_recent_interactions"
    )
    monkeypatch.setattr(extract_module, "get_graph_engine", _fake_get_graph_engine)

    interactions = await extract_module.extract_recent_interactions(data=None, last_k=5)

    assert len(interactions) == 1
    assert interactions[0].question == "What is Cognee?"
    assert interactions[0].answer == "Cognee builds AI memory."
    assert str(interactions[0].id) == interaction_id
    assert fake_engine.requested_limit == 5
    assert fake_engine.requested_ids == [interaction_id]


@pytest.mark.asyncio
async def test_classify_interaction_sentiment_returns_datapoints(monkeypatch):
    interaction = CogneeUserInteraction(
        id=uuid.uuid4(),
        question="How did the session go?",
        answer="Great!",
        context="The user successfully completed the task.",
    )

    calls = []

    async def _fake_acreate_structured_output(text_input, system_prompt, response_model):
        calls.append((text_input, system_prompt, response_model))
        return InteractionSentimentResponse(
            sentiment="positive", confidence=0.92, summary="The user was satisfied."
        )

    classify_module = importlib.import_module(
        "cognee.tasks.sentiment.classify_interaction_sentiment"
    )
    classify_module.SYSTEM_PROMPT = "system"
    classify_module.USER_PROMPT_TEMPLATE = (
        "Question: {question}\nAnswer: {answer}\nContext: {context}"
    )
    monkeypatch.setattr(
        classify_module.LLMGateway,
        "acreate_structured_output",
        _fake_acreate_structured_output,
    )

    sentiments = await classify_module.classify_interaction_sentiment([interaction])

    assert len(sentiments) == 1
    sentiment_dp = sentiments[0]
    assert sentiment_dp.interaction_id == interaction.id
    assert sentiment_dp.sentiment == SentimentLabel.POSITIVE
    assert sentiment_dp.confidence == pytest.approx(0.92)
    assert sentiment_dp.summary == "The user was satisfied."
    assert calls[0][2] is InteractionSentimentResponse


@pytest.mark.asyncio
async def test_link_sentiment_to_interactions_creates_edges(monkeypatch):
    sentiment = InteractionSentiment(
        id=uuid.uuid4(),
        interaction_id=uuid.uuid4(),
        sentiment=SentimentLabel.NEUTRAL,
        confidence=0.5,
        summary="Neutral experience.",
    )

    recorded_edges = []

    class _EdgeGraphEngine:
        async def add_edges(self, edges):
            recorded_edges.extend(edges)

    async def _fake_index_graph_edges(edges):
        recorded_edges.append(("indexed", edges))

    async def _fake_get_graph_engine():
        return _EdgeGraphEngine()

    link_module = importlib.import_module(
        "cognee.tasks.sentiment.link_sentiment_to_interactions"
    )
    monkeypatch.setattr(link_module, "get_graph_engine", _fake_get_graph_engine)
    monkeypatch.setattr(link_module, "index_graph_edges", _fake_index_graph_edges)

    result = await link_module.link_sentiment_to_interactions([sentiment])

    assert result == [sentiment]
    created_edge = recorded_edges[0]
    assert created_edge[0] == sentiment.id
    assert created_edge[1] == sentiment.interaction_id
    assert created_edge[2] == "has_sentiment"
