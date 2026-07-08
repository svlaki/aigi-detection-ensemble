"""Modal web endpoint for the AIGI detection ensemble FastAPI backend.

Serves the same FastAPI app from demo.backend.main as a Modal ASGI endpoint.
Model files are bundled into the image for fast cold starts. D3QE weights are
included when available locally (enables full mode).

Deploy (creates a persistent URL):
  modal deploy demo/modal_backend.py

Dev server (temporary URL, hot-reloads):
  modal serve demo/modal_backend.py
"""
from __future__ import annotations

import os
import modal

app = modal.App("aigi-demo-backend")

# Build the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "open_clip_torch",
        "scikit-learn",
        "joblib",
        "scipy",
        "numpy",
        "pandas",
        "pillow",
        "fastapi[standard]",
        "python-multipart",
        "ftfy",
        "regex",
        "tqdm",
        "setuptools",
    )
    # Project source code
    .add_local_dir("src", "/app/src", copy=True)
    .add_local_dir("demo/backend", "/app/demo/backend", copy=True)
    .add_local_file("demo/__init__.py", "/app/demo/__init__.py", copy=True)
    # Model files (small joblib files, ~100KB total)
    .add_local_dir("models", "/app/models", copy=True)
)

# Add D3QE weights if available locally (~363MB, enables full mode)
_d3qe_pretrained = os.path.join(os.path.dirname(__file__), "..", "external", "D3QE", "pretrained")
_d3qe_networks = os.path.join(os.path.dirname(__file__), "..", "external", "D3QE", "networks")
_d3qe_available = os.path.isdir(_d3qe_pretrained) and os.path.isdir(_d3qe_networks)

print(f"[modal_backend] D3QE available locally: {_d3qe_available}")
print(f"[modal_backend]   pretrained dir: {_d3qe_pretrained} exists={os.path.isdir(_d3qe_pretrained)}")
print(f"[modal_backend]   networks dir:   {_d3qe_networks} exists={os.path.isdir(_d3qe_networks)}")

if _d3qe_available:
    image = (
        image
        .add_local_dir("external/D3QE/pretrained", "/app/external/D3QE/pretrained", copy=True)
        .add_local_dir("external/D3QE/networks", "/app/external/D3QE/networks", copy=True)
        .add_local_dir("external/D3QE/networks/clip", "/app/external/D3QE/networks/clip", copy=True)
    )


@app.function(
    image=image,
    gpu="T4",
    scaledown_window=300,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from demo.backend.main import app as _app
    return _app
