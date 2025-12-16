# Interaction Sentiment Memify Pipeline

The interaction sentiment pipeline classifies the sentiment of recent
`CogneeUserInteraction` nodes and stores the results as
`InteractionSentiment` datapoints that are linked back to the original
interaction via a `has_sentiment` edge.

## Tasks

The pipeline is composed of the following tasks:

1. `extract_recent_interactions(last_k=20)` – loads the most recent
   interactions directly from the graph database.
2. `classify_interaction_sentiment` – calls the local Ollama LLM via the
   Instructor integration to obtain sentiment, confidence, and a short
   summary for each interaction.
3. `link_sentiment_to_interactions` – persists the
   `InteractionSentiment` datapoints and creates the `has_sentiment`
   edges to the interaction nodes.

## Running the Pipeline

```python
from cognee import memify
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.sentiment import (
    extract_recent_interactions,
    classify_interaction_sentiment,
    link_sentiment_to_interactions,
)
from cognee.tasks.storage import add_data_points


await memify(
    extraction_tasks=[Task(extract_recent_interactions, last_k=20)],
    enrichment_tasks=[
        Task(classify_interaction_sentiment),
        Task(add_data_points, task_config={"batch_size": 10}),
        Task(link_sentiment_to_interactions),
    ],
    dataset="main_dataset",
    data=[{}],
)
```

The classifier relies on the Ollama adapter and expects a local Ollama
instance to be available through the configured endpoint. Output is
returned as instance-only JSON by Instructor, ensuring that only the
sentiment datapoint is emitted without schema definitions.
