"""
Simple CLI for launching a standalone worker server
"""

import uvicorn
import fire
from cascade.executor.multihost.worker_server import build_app

def run_one(port: int, executor: str):
    instance: Executor
    if executor == "instant":
        instance = InstantExecutor(1)

    app = build_app(instance)
    uvicorn.run(app, host="0.0.0.0", port=port)

