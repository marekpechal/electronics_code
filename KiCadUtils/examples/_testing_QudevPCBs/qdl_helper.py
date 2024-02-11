# Copyright (c) 2020-2023, Graham J. Norris, Nicolas Thill, and ETH Zürich

"""Helper functions."""

import os
import pathlib
from typing import Optional, Union
import numpy as np

import logging
log = logging.getLogger(__name__)


def centered_range(count: int) -> np.ndarray:
    """Create a centered range.

    Centered around zero with unit spacing by default. Correctly
    handles even or odd counts.

    Parameters
    ----------
    count : int
        Number of points in the range.
    """
    return np.arange(-(count - 1)/2, (count - 1)/2 + 1)




def validate_string(input_string: str, sanitize: bool = False) -> str:
    """Check if the string is too long or contains illegal characters.

    Checks to make sure that the string is shorter than 47 characters
    and only contains ASCII characters 'a' to 'z', 'A' to 'Z',
    '0' to '9', and '_'. This is needed to make sure that the GDS cell
    names are compatible with all GDS readers, in particular, the
    Heidelberg instruments laser writers which have trouble with cell
    names containing '-' or which are too long.

    Parameters
    ----------
    input_string : str
        The string to validate or sanitize.
    sanitize : bool, optional
        Whether to make the input string valid by replacing invalid
        characters with underscores. Will raise a ValueError if
        input sring is not valid and sanitize is False. Default: False.

    Returns
    -------
    output_string : str
        The validated and possibly sanitized string.
    """
    # Value that worked; not guaranteed to be maximum possible length
    string_length_limit = 47
    if len(input_string) > string_length_limit:
        raise ValueError(f'String {input_string} is longer than the '
                         f'limit of {string_length_limit} characters.')
    allowed_characters = 'abcdefghijklmnopqrstuvwxyz'
    allowed_characters += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    allowed_characters += '0123456789_'
    # Strip the allowed characters and check if the length of the string
    # is zero
    # TODO: not sure if we should allow empty input strings
    remaining_characters = input_string.strip(allowed_characters)
    if len(remaining_characters) > 0:
        if not sanitize:
            raise ValueError(f'String {input_string} contains the illegal '
                             f'characters "{remaining_characters}"')
        input_string = ''.join([c if c in allowed_characters else '_'
                                for c in input_string])
    return input_string


def rotation_matrix(theta: float) -> np.ndarray:
    """Compute the standard rotation matrix.

    Computes the matrix which will rotate the coordinate system from
    the current values counter-clockwise.

    Parameters
    ----------
    theta : float
        Angle to rotate (in radians).

    Returns
    -------
    rot_mat : array-like[2][2] of float
        Two by two matrix containing the rotation matrix.
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def generate_tile_positions_square(
    start_x: float,
    start_y: float,
    stop_x: float,
    stop_y: float,
    step: float,
):
    """Generate tiling positions in a square lattice.

    Parameters
    ----------
    start_x : float
        Initial position to start tiling along the x-axis.
    start_y : float
        Initial position to start tiling along the y-axis.
    stop_x : float
        Final position to tile along the x-axis.
    stop_y : float
        Final position to tile along the y-axis.
    step : float
        Lattice constant along the x and y axes.

    Returns
    -------
    position : array-like[N][2]
        Array of tiling positions.
    """
    x_vec = np.arange(start_x, stop_x, step)
    y_vec = np.arange(start_y, stop_y, step)
    x_mat, y_mat = np.meshgrid(x_vec, y_vec)
    position = np.array([x_mat.ravel(), y_mat.ravel()]).T
    return position




def ensure_path_exists(file_path: pathlib.Path) -> None:
    """Create the last folder of a path if it does not exist.

    Only works if just the last folder is missing (all higher
    folders must exist).

    Parameters
    ----------
    file_path : pathlib.Path
        File path to check for the last directory of.
    """
    if not file_path.exists():
        os.mkdir(file_path)
