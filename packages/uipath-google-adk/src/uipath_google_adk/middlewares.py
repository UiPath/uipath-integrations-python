from uipath._cli.middlewares import Middlewares

from ._cli.cli_new import google_adk_new_middleware


def register_middleware():
    """This function will be called by the entry point system when uipath-google-adk is installed"""
    Middlewares.register("new", google_adk_new_middleware)
