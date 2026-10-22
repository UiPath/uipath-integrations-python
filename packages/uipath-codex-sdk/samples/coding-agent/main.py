"""Codex coding agent: uses built-in shell + file-edit tools in a workspace."""

from pydantic import BaseModel, Field

from uipath_codex_sdk import UiPathCodexAgent


class CodingTask(BaseModel):
    task: str = Field(description="The coding task for Codex to carry out.")


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    prompt="{task}",
    input_schema=CodingTask,
    sandbox="full-access",
    approval_mode="deny_all",
    base_instructions=(
        "You are a coding assistant. Keep every file you create in the "
        "./workspace folder (create it if needed). Create and run code as "
        "needed, then report what you did and the result."
    ),
)
