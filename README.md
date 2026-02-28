# BookBuyingAgent

## Project Structure

- `agent_server.py`: Main AI Agent server.
- `recommendation_tool.py`: Core logic for book recommendations.
- `mock_retailer/`: Mock e-commerce service with book catalogs.
- `frontend/`: React-based user interface.
- `scripts/`: Maintenance and utility scripts (e.g., catalog enrichment, architecture diagram).
- `tests/`: Integration tests and simulations.

## Getting Started

1. **Prerequisites**: Ensure you have **Python** and **Node.js** installed.
2. **Run the setup script**:

   **Option A: Recommended (Persist Venv)**
   To run the script and keep the virtual environment active in your terminal (recommended for development), run these two commands:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   . .\start_all.ps1
   ```
   *Note: The leading `.` and space (dot-sourcing) are crucial for keeping the venv active.*

   **Option B: Quick Start (No Persistence)**
   If you just want to run the app without keeping the environment active:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\start_all.ps1
   ```

   The script will automatically:
   - Create a Python virtual environment and install dependencies.
   - Install Node.js packages for the frontend.
   - Launch both the Backend API and Frontend UI.