"""Verify all required packages import cleanly."""


def test_numpy():
    import numpy as np
    assert np.__version__


def test_pandas():
    import pandas as pd
    assert pd.__version__


def test_pyproj():
    from pyproj import Geod
    assert Geod(ellps="WGS84")


def test_shapely():
    from shapely.geometry import LineString, Point, Polygon
    assert Point(0, 0)


def test_geopandas():
    import geopandas as gpd
    assert gpd.__version__


def test_mikeio():
    import mikeio
    assert mikeio.__version__


def test_scipy():
    from scipy.spatial import KDTree
    assert KDTree([[0, 0]])


def test_pydantic():
    from pydantic import BaseModel
    class M(BaseModel):
        x: int
    assert M(x=1).x == 1


def test_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assert plt


def test_shipwake_package():
    import aiswakepy
    assert aiswakepy.__version__ == "0.1.0"
