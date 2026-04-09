"""Empirical ship-wake models.

Each module exposes a ``compute_<model>(df, ...)`` function that accepts a
DataFrame with vessel and depth columns and returns a copy with all
model-specific columns added (no filtering applied here).

Available models
----------------
kriebel  : Kriebel & Seelig (2005) — default model.
"""
