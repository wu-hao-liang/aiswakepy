"""Empirical ship-wake models.

Each module exposes a ``compute_<model>(df, ...)`` function that accepts a
DataFrame with vessel and depth columns and returns a Series of Hmax values
(no row filtering applied; out-of-range rows are set to NaN).

Available models
----------------
kriebel  : Kriebel & Seelig (2005) — default pipeline model.
pianc    : PIANC (1987) — inland waterways, Froude_D scaling.
sorensen : Sorensen (1984) — displacement vessels, Froude_D 0.2–0.8.
bhowmik  : Bhowmik et al. (1982) — Froude_Draft scaling, no distance dep.
gates    : Gates & Herbich (1977) — cusp-line distance decay.
blaauw   : Blaauw et al. (1985) — deep water, hull-type coefficient.
maynord  : Maynord (2005) — semi-planing/planing small craft.
"""
