"""Start the local pass-and-play FastAPI server."""

from __future__ import annotations

import argparse


def main() -> None:
    """Run the local play server with uvicorn."""
    parser = argparse.ArgumentParser(description="Run High Society local play server.")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port number (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "highsociety.server.local_api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
