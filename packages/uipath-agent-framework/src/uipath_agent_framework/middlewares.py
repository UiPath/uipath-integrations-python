from uipath._cli.middlewares import Middlewares

from ._cli.cli_new import agent_framework_new_middleware


def register_middleware():
    """This function will be called by the entry point system when uipath-agent-framework is installed"""
    Middlewares.register("new", agent_framework_new_middleware)
