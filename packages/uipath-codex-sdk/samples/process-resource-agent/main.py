"""Process tool from an agent.json resource, with pre-filled arguments.

``resources/add_shipping.json`` is a process tool resource as a low-code
agent's ``agent.json`` declares it. ``process_tool_from_resource`` turns it
into a durable process tool, as LangChain does for the same resource:

* ``a`` comes from the agent input (``argumentProperties`` variant
  ``argument``, path ``$['order']['items']``). The model sees it as optional
  and fills it only when the input has no value.
* ``b`` is fixed to 1000 (variant ``static``) and hidden from the model.

The job gets ``{"a": <items>, "b": 1000}`` and the run suspends until it
finishes, as with any durable process tool.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from uipath_codex_sdk import UiPathCodexAgent, process_tool_from_resource

#: Orchestrator folder of the a_plus_b release. ``properties.folderPath`` in
#: the resource is ignored, as in low-code (the job runs in the folder the
#: agent runs in); pass it here to point the tool at another folder.
PROCESS_FOLDER = "Coded/a_plus_b"

RESOURCE = json.loads(
    (Path(__file__).parent / "resources" / "add_shipping.json").read_text()
)


class Order(BaseModel):
    items: int = Field(description="Number of items in the order")


class Request(BaseModel):
    order: Order


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    input_schema=Request,
    prompt="Call Add_Shipping once and report the result c.",
    tools=[process_tool_from_resource(RESOURCE, folder_path=PROCESS_FOLDER)],
)
