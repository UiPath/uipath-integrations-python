from uipath._cli.middlewares import Middlewares

from ._cli.cli_new import codex_new_middleware


def register_middleware():
    """Called by the entry point system when uipath-codex-sdk is installed."""
    Middlewares.register("new", codex_new_middleware)
