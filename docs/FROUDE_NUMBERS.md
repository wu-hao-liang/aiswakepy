# Froude Numbers in aiswakepy

Summary of all Froude number types, formulas, default limits, and usage across empirical models.

## Froude Number Definitions

| Variable | Type | Formula | Description |
|----------|------|---------|-------------|
| `Froude_D` | Depth Froude | V / sqrt(g * h) | V = vessel speed (m/s), h = water depth (m) |
| `Froude_L` | Length Froude | V / sqrt(g * L) | V = vessel speed (m/s), L = vessel length (m) |
| `Froude_M` | Modified Froude (Kriebel) | (V / sqrt(g * L)) * exp(Alpha * d / h) | Alpha = 2.35*(1-Cb), d = draught, h = depth |
| `Froude_Displace` | Displacement Froude | V / sqrt(g * W^(1/3)) | W = volumetric displacement (m^3) |
| `Froude_Draft` | Draught Froude | V / sqrt(g * d) | d = vessel draught (m) |

## Default Limits by Model

| Model | Froude Type | Parameter | Default | Condition |
|-------|-------------|-----------|---------|-----------|
| **Kriebel (2005)** | Froude_M | `min_Froude_M` | 0.1 | Invalid if Froude_M < 0.1 |
| | Froude_M | `max_Froude_M` | 0.5 | Invalid if Froude_M > 0.5 |
| | Froude_D | `max_Froude_D` | 1.0 | Invalid if Froude_D >= 1.0 |
| **PIANC (1987)** | Froude_D | `max_Froude_D` | 0.7 | Invalid if Froude_D >= 0.7 |
| **Blaauw (1985)** | Froude_D | `max_Froude_D` | 0.7 | Invalid if Froude_D >= 0.7 |
| **Sorensen (1984)** | Froude_D | `min_Froude_D` | 0.2 | Invalid if Froude_D <= 0.2 |
| | Froude_D | `max_Froude_D` | 0.8 | Invalid if Froude_D >= 0.8 |
| **Gates (1977)** | Froude_L | `max_Froude_L` | 0.7 | Invalid if Froude_L >= 0.7 |
| **Maynord (2005)** | Froude_Displace | `min_Froude_Displace` | 1.3 | Valid if ANY condition met |
| | Froude_L | `min_Froude_L` | 0.4 | Valid if ANY condition met |
| **Bhowmik (1982)** | Froude_Draft | _(none)_ | _(none)_ | No applicability filter |

## Pipeline Column

`Froude_D` is computed once in `aiswakepy/stages/vessel.py` as a DataFrame column and propagated through all downstream stages. Individual models compute their own Froude numbers internally as needed.
