import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import shapely.geometry
import scipy.interpolate


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

def offsetPath(path, distance):
    """
    Generate a path which is perpendicularly offset from the given path
    by a given distance.

    Parameters
    ----------
    path : numpy.array
        2d array [[x1, y1], [x2, y2], ...], defining the path
    distance : float
        offset of the path; can be negative
    """
    lst = []
    K = np.array([[0, 1], [-1, 0]])
    for i in range(len(path)):
        if i == 0:
            normal_vector = K.dot(path[1] - path[0])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
        elif i == len(path) - 1:
            normal_vector = K.dot(path[-1] - path[-2])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
        else:
            normal_vector1 = K.dot(path[i] - path[i-1])
            normal_vector2 = K.dot(path[i+1] - path[i])
            normal_vector1 = normal_vector1 / np.linalg.norm(normal_vector1)
            normal_vector2 = normal_vector2 / np.linalg.norm(normal_vector2)
            normal_vector = (normal_vector1 + normal_vector2) / (
                1 + np.dot(normal_vector1, normal_vector2))

        lst.append(path[i] + normal_vector * distance)
    return np.array(lst)


def distributePointsOnPath(path, separation, init_gap = 0.0, fin_gap = 0.0):
    """
    Distribute equidistant points along a path.

    Parameters
    ----------
    path : numpy.array
        2d array [[x1, y1], [x2, y2], ...], defining the path
    separation : float
        minimal separation of the points (the number of generated points will
        be the largest which results in at least the given separation)
    init_gap : float, optional
        distance of the first point from the start of the path; defaults to 0
    fin_gap : float, optional
        distance of the last point from the end of the path; defaults to 0
    """
    length_array = np.concatenate(([0.0],
        np.cumsum(np.linalg.norm(path[1:] - path[:-1], axis = 1))))
    total_length = length_array[-1]
    interp_func = scipy.interpolate.interp1d(length_array, path, axis = 0)
    x0 = init_gap
    x1 = total_length - fin_gap
    N = int((x1 - x0) / separation) + 1
    return np.array([interp_func(x) for x in np.linspace(x0, x1, N)])

def distributePointsInPolygon(poly, grid_size,
        pattern_type = 'triangular', margin = 0.0):
    poly = shapely.geometry.Polygon(poly).buffer(-margin)
    x1, y1, x2, y2 = poly.bounds
    w = x2 - x1
    h = y2 - y1
    x0 = (x1 + x2) / 2
    y0 = (y1 + y2) / 2
    if pattern_type == 'triangular':
        dy = grid_size * np.sqrt(3) / 2
        Ny = int(h / dy) - 1
        Nx = int(w / grid_size) - 1
        for i in range(Ny):
            y = dy * (i - (Ny-1)/2)
            for j in range(Nx):
                x = grid_size * ((j - (Nx-1)/2) + 0.25 * (-1)**(i % 2))
                pt = [x0+x, y0+y]
                if poly.contains(shapely.geometry.Point(*pt)):
                    yield pt

    else:
        raise NotImplementedError(f'pattern type {pattern_type}')
