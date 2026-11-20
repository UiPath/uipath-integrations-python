"""Durable process tool: the agent calls an Orchestrator process and suspends.

The model calls ``a_plus_b`` (an Orchestrator process with integer inputs
``a`` and ``b``). Each call starts a job and suspends the run; when the jobs
finish, the run resumes and the model gets each job's output as the result of
its original call.
"""


from pydantic import BaseModel, Field

from uipath_codex_sdk import UiPathCodexAgent, uipath_process_tool


class Numbers(BaseModel):
    x1: int = Field(description="First addend of the first sum")
    y1: int = Field(description="Second addend of the first sum")
    x2: int = Field(description="First addend of the second sum")
    y2: int = Field(description="Second addend of the second sum")


class AddInput(BaseModel):
    a: int
    b: int


class AddOutput(BaseModel):
    """The job's output arguments. It also shapes a mocked result in evals."""

    c: int = Field(description="The sum a + b")


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    input_schema=Numbers,
    prompt=(
        "Use the a_plus_b tool to compute {x1} + {y1} and {x2} + {y2}. "
        "Make both calls together, in parallel, in a single response. "
        "When you have both results, report both sums."
    ),
    tools=[
        uipath_process_tool(
            "a_plus_b",
            # Orchestrator folder of the a_plus_b release. Omit it to use the
            # folder the agent runs in (Orchestrator sets UIPATH_FOLDER_PATH
            # for a deployed job). At deploy time bindings.json lets an
            # overwrite point this tool at another folder.
            folder_path="Coded/a_plus_b",
            description="Adds two integers a and b in a UiPath process.",
            input_schema=AddInput,
            output_schema=AddOutput,
        )
    ],
)
