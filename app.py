from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pathlib import Path

from agent_server import app as agent_app
from mock_retailer.main import app as mock_app

app = FastAPI()

app.mount("/agent", agent_app)
app.mount("/mock", mock_app)


# redirects for trailing slash
@app.get("/mock")
def mock_redirect():
    return RedirectResponse(url="/mock/")


@app.get("/agent")
def agent_redirect():
    return RedirectResponse(url="/agent/")


# --- Frontend build ---
dist_dir = Path(__file__).parent / "frontend" / "dist"
assets_dir = dist_dir / "assets"

app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/{path:path}")
def serve_spa(path: str):
    file_path = dist_dir / path
    if path and file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(dist_dir / "index.html")
