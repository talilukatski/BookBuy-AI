from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# import של שני ה-APIs שלך
from agent_server import app as agent_app
from mock_retailer.main import app as mock_app

app = FastAPI()

app.mount("/agent", agent_app)
app.mount("/mock", mock_app)

# --- Frontend build ---
dist_dir = Path(__file__).parent / "frontend" / "dist"
assets_dir = dist_dir / "assets"

# assets של Vite
app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


# SPA fallback (React Router)
@app.get("/{path:path}")
def serve_spa(path: str):
    file_path = dist_dir / path
    if path and file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(dist_dir / "index.html")
