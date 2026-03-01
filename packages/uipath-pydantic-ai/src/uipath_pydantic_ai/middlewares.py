from uipath._cli.middlewares import Middlewares

from ._cli.cli_new import pydantic_ai_new_middleware


def register_middleware():
    """This function will be called by the entry point system when uipath-pydantic-ai is installed"""
    Middlewares.register("new", pydantic_ai_new_middleware)
