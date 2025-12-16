import os
import pathlib
from enum import Enum
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic_core import PydanticUndefined

import pytest

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.retrieval.utils.models import CogneeUserInteraction
from cognee.tasks.sentiment import (
    classify_interaction_sentiment,
    extract_recent_interactions,
    link_sentiment_to_interactions,
)
from cognee.tasks.sentiment.classify_interaction_sentiment import (
    InteractionSentimentResponse,
)
from cognee.tasks.storage import add_data_points


@pytest.mark.asyncio
async def test_memify_pipeline_creates_sentiment_nodes(monkeypatch):
    data_directory_path = str(
        pathlib.Path(
            os.path.join(
                pathlib.Path(__file__).parent,
                ".data_storage/test_interaction_sentiment",
            )
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(
                pathlib.Path(__file__).parent,
                ".cognee_system/test_interaction_sentiment",
            )
        ).resolve()
    )

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    monkeypatch.setenv("REQUIRE_AUTHENTICATION", "false")
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    recorded_models = []

    class _FakeLLMClient:
        @staticmethod
        def _default_value(annotation: Any) -> Any:
            origin = get_origin(annotation)

            if origin is not None:
                args = get_args(annotation)
                if origin is Annotated:
                    return _FakeLLMClient._default_value(args[0])
                if origin is list:
                    return []
                if origin is dict:
                    return {}
                if origin is tuple:
                    return tuple()
                if origin is set:
                    return set()
                if origin is bool:
                    return False
                if origin is str:
                    return "mock"
                if origin is int:
                    return 0
                if origin is float:
                    return 0.0
                if origin is type(None):
                    return None
                if origin is Union:
                    for arg in args:
                        if arg is not type(None):  # noqa: E721
                            return _FakeLLMClient._default_value(arg)
                    return None
                if origin is Literal:
                    return args[0]

            if isinstance(annotation, type):
                if issubclass(annotation, Enum):
                    member = next(iter(annotation))
                    return getattr(member, "value", member)
                if issubclass(annotation, bool):
                    return False
                if issubclass(annotation, int):
                    return 0
                if issubclass(annotation, float):
                    return 0.0
                if issubclass(annotation, str):
                    return "mock"

            return "mock"

        def _build_response(self, response_model):
            field_values = {}
            model_fields = getattr(response_model, "model_fields", {})

            for name, field_info in model_fields.items():
                if field_info.default is not PydanticUndefined:
                    field_values[name] = field_info.default
                elif getattr(field_info, "default_factory", None) is not None:
                    field_values[name] = field_info.default_factory()
                else:
                    field_values[name] = self._default_value(field_info.annotation)

            if {"sentiment", "confidence", "summary"}.issubset(model_fields.keys()):
                field_values.update(
                    {
                        "sentiment": "negative",
                        "confidence": 0.4,
                        "summary": "The user was dissatisfied with the answer.",
                    }
                )

            if {"nodes", "edges"}.issubset(model_fields.keys()):
                field_values["nodes"] = []
                field_values["edges"] = []

            recorded_models.append(response_model)

            try:
                return response_model(**field_values)
            except Exception:  # noqa: BLE001
                return response_model.model_construct(**field_values)

        async def acreate_structured_output(self, text_input, system_prompt, response_model):
            return self._build_response(response_model)

        def create_structured_output(self, text_input, system_prompt, response_model):
            return self._build_response(response_model)

        async def create_transcript(self, *args, **kwargs):
            return ""

        async def transcribe_image(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client.get_llm_client",
        lambda: _FakeLLMClient(),
    )

    dataset_name = "interaction_sentiment_dataset"

    await cognee.add("Cognee turns documents into AI memory.", dataset_name)
    print("[test] Starting cognify", flush=True)
    await cognee.cognify([dataset_name])

    interaction = CogneeUserInteraction(
        question="How can Cognee help with document understanding?",
        answer="Cognee builds AI-native memory from your documents.",
        context="User asked about Cognee capabilities.",
    )

    await add_data_points([interaction])

    extraction_tasks = [Task(extract_recent_interactions, last_k=10)]
    enrichment_tasks = [
        Task(classify_interaction_sentiment),
        Task(add_data_points, task_config={"batch_size": 10}),
        Task(link_sentiment_to_interactions),
    ]

    print("[test] Starting memify", flush=True)
    await cognee.memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        dataset=dataset_name,
        data=[{}],
    )

    graph_engine = await get_graph_engine()
    nodes_after, edges_after = await graph_engine.get_graph_data()

    sentiment_nodes = [
        (node_id, props)
        for node_id, props in nodes_after
        if props.get("type") == "InteractionSentiment"
    ]

    assert sentiment_nodes, "Expected InteractionSentiment nodes to be created"

    sentiment_ids = {node_id for node_id, _ in sentiment_nodes}
    has_sentiment_edges = [
        edge for edge in edges_after if edge[0] in sentiment_ids and edge[2] == "has_sentiment"
    ]

    assert has_sentiment_edges, "Expected has_sentiment edges linking to interactions"
    assert InteractionSentimentResponse in recorded_models

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
