"""Human-in-the-loop (HITL) support for Agent Framework.

Provides the ``requires_approval`` decorator for marking tool functions
that need human approval before execution.

Example::

    from uipath_agent_framework.chat import requires_approval

    @requires_approval
    def transfer_funds(from_account: str, to_account: str, amount: float) -> str:
        \"\"\"Transfer funds between accounts.\"\"\"
        return f"Transferred ${amount:.2f} from {from_account} to {to_account}"
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, overload

from agent_framework import FunctionTool, tool


@overload
def requires_approval(func: Callable[..., Any]) -> FunctionTool[Any]: ...


@overload
def requires_approval(
    func: None = None,
) -> Callable[[Callable[..., Any]], FunctionTool[Any]]: ...


def requires_approval(
    func: Callable[..., Any] | None = None,
) -> FunctionTool[Any] | Callable[[Callable[..., Any]], FunctionTool[Any]]:
    """Decorator that marks a tool function as requiring human approval.

    When the agent calls a tool decorated with ``@requires_approval``,
    execution suspends and waits for a human to approve or reject
    the tool call before proceeding.

    Can be used with or without parentheses::

        @requires_approval
        def my_tool(arg: str) -> str: ...

        @requires_approval()
        def my_tool(arg: str) -> str: ...

    Under the hood, this sets ``approval_mode="always_require"`` on the
    resulting ``FunctionTool``. The workflow runtime then intercepts the
    call via ``request_info`` and suspends execution for human approval.

    Args:
        func: The tool function to wrap. If None, returns a decorator.

    Returns:
        A FunctionTool with approval_mode set to "always_require".
    """
    if func is not None:
        return tool(func, approval_mode="always_require")

    def decorator(fn: Callable[..., Any]) -> FunctionTool[Any]:
        return tool(fn, approval_mode="always_require")

    return decorator


__all__ = ["requires_approval"]
