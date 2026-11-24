# Structured output

A Codex agent that returns a typed result. `output_schema` makes Codex ask
the model for native structured output in that shape, and the runtime
validates the result against it.

```python
class Joke(BaseModel):
    setup: str = Field(description="The setup line of the joke.")
    punchline: str = Field(description="The punchline.")
    rating: int = Field(description="How funny, 1-10.")


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    prompt="Tell me a short, clean joke about {topic}.",
    input_schema=JokeInput,
    output_schema=Joke,
)
```

Both schemas become the entry point's input and output schemas
(`uipath init` writes them to `entry-points.json`).

## Run locally

```bash
cd samples/structured-output-agent
uv sync
uv run uipath auth
uv run uipath run agent --file input.json   # {"topic": "robots"}
```

The output is `{"setup": "...", "punchline": "...", "rating": 7}`.

## Evaluate

`evaluations/eval-sets/evaluation-set-default.json` scores three topics with
a JSON similarity evaluator (against an expected joke) and a contains
evaluator (the joke mentions its topic):

```bash
uv run uipath eval agent evaluations/eval-sets/evaluation-set-default.json --no-report
```
