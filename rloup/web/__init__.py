"""Web interface for interactive Coup play, model testing, and live tournaments."""

from .server import WebApp, create_server, main

__all__ = ["WebApp", "create_server", "main"]
