"""Read-only remote corpus connectors."""

from .synology import (
    ALLOWED_API_METHODS,
    N4S4ReadTransportV091,
    SynologyApiError,
    SynologyFileStationReader,
    SynologyProtocolError,
    SynologyRangeError,
    SynologyStreamResponse,
)
from .synology_transport import (
    RequestsResponseV091,
    RequestsStreamAdapter,
    SynologyApiSessionV091,
    SynologyApiTransportV091,
    SynologyTransportError,
)

__all__ = [
    "ALLOWED_API_METHODS",
    "N4S4ReadTransportV091",
    "RequestsResponseV091",
    "RequestsStreamAdapter",
    "SynologyApiSessionV091",
    "SynologyApiTransportV091",
    "SynologyApiError",
    "SynologyFileStationReader",
    "SynologyProtocolError",
    "SynologyRangeError",
    "SynologyStreamResponse",
    "SynologyTransportError",
]
