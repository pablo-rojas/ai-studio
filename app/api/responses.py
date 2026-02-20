from __future__ import annotations

from typing import Any

from fastapi import Request


def ok_response(data: Any) -> dict[str, Any]:
    """Wrap a successful payload in the standard API envelope."""
    return {"status": "ok", "data": data}


def is_hx_request(request: Request) -> bool:
    """Return whether the incoming request originates from HTMX."""
    return request.headers.get("HX-Request") is not None
