"""HTTP routes. Controllers should be thin — delegate to services."""

from app.api.routes import api_router

__all__ = ["api_router"]
