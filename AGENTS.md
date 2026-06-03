# AGENTS

`aiswakepy` is a Python/Dash web app and pipeline for AIS-based ship-wake wave-height estimation at shorelines. It displays ship tracks on a frontend map canvas, runs wake and shore-impact calculations, and visualizes results.

Development is local under WSL now; the app is being prepared for public VPS deployment. Treat hardcoded/local input-file assumptions as technical debt. AIS should support public API ingestion and/or upload, other inputs should be uploadable, and shapefile areas may be replaced by drawn map polygons.

Use `uv` only:
- Run tests: `uv run pytest tests/ -q`
- Run pipeline: `uv run python main.py --config config.json`
- Run Dash app: `uv run python dash_app.py`
- Validate MATLAB parity: `uv run python validate_pipeline.py`
- Add dependencies with `uv add` or `uv add --dev`; do not use `pip`.

For public-facing features, consider upload validation, storage isolation, and clear error handling. Do not kill or restart a developer-managed Dash server without explicit instruction.
