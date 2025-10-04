"""Prediction CLI placeholder (wnv-predict).

For now this simply launches the FastAPI service (blocking) so users can send
classification requests. In future this can be expanded with batch prediction.
"""

import uvicorn

def main():  # pragma: no cover - runtime convenience
    uvicorn.run("scripts.serve_api:app", host="0.0.0.0", port=8000, reload=False)
