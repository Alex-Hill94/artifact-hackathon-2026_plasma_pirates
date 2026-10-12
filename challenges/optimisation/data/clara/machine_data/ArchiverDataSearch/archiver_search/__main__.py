"""Run the API server: ``python -m archiver_search``."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable

from .config import get_settings


def main(argv: Iterable[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )
    # h5pyd logs every HTTP request at INFO; one line per hyperslab read is far
    # too much for a running server.
    logging.getLogger("h5pyd").setLevel(logging.WARNING)

    import uvicorn

    uvicorn.run(
        "archiver_search.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
