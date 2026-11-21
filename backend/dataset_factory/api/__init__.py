"""FastAPI backend: start, watch, stop and resume curation jobs over HTTP."""

from .app import create_app

__all__ = ["create_app"]
