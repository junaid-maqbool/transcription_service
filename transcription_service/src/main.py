import asyncio

import uvicorn

from src.app import app


async def main():

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
