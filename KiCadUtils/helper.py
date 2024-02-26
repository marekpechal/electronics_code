import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import shapely.geometry


def plotShapelyPolygon(ax, poly, **kwargs):
    """
    Helper function to plot a shapely.geometry.Polygon object
    """
    # from https://stackoverflow.com/a/70533052
    A = np.array([1, -1])
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2] * A),
        *[Path(np.asarray(ring.coords)[:, :2] * A) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

def plotShapelyPolyLike(ax, poly, **kwargs):
    """
    Helper function to plot a shapely.geometry.Polygon or
    shapely.geometry.MultiPolygon object
    """
    if isinstance(poly, shapely.geometry.Polygon):
        polys = [poly]
    elif isinstance(poly, shapely.geometry.MultiPolygon):
        polys = poly.geoms
    else:
        raise NotImplementedError(f'{poly}')
    for poly in polys:
        plotShapelyPolygon(ax, poly, **kwargs)
