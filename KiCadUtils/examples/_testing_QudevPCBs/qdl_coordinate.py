# Copyright (c) 2022-2023, Graham J. Norris, Nicolas Thill, and ETH Zürich

"""Coordinate representation classes."""

# Stores annotations as strings, so can use self-referential annotations
from __future__ import annotations

# Note: with a minimum python version >= 3.10, we could replace
# Union[A, B] with A | B; see PEP 604

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import logging

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Vec2:
    """2-vector dataclass.

    Attributes
    ----------
    x : float, optional
        X-coordinate. Defaults to 0.0.
    y : float, optional
        Y-coordinate. Defaults to 0.0.
    """

    x: float = 0.0
    y: float = 0.0

    # NOTE for many of these operator methods, it might be faster to
    # re-implement the operation rather than reusing other operators;
    # would need to benchmark it if this becomes a performance issue
    # NOTE __eq__ and __neq__ are defined by dataclass

    def __post_init__(self):

        # check whether x and y are floats
        for name, value in (('x', self.x), ('y', self.y)):
            if not isinstance(value, float):
                try:
                    object.__setattr__(self, name, float(value))
                except ValueError as error:
                    error_message = (
                        f"Argument {name} must be mappable to float. \n"
                        f"{name}: {value} of type {type(value)} fails."
                    )
                    raise ValueError(error_message) from error

    def __getitem__(self, item: int) -> float:
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            raise TypeError('Vec2 only contains coordinate x at index 0 and y'
                            ' at index 1.')

    def __len__(self) -> int:
        return 2

    def __iter__(self):
        return (i for i in [self.x, self.y])

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array([self.x, self.y], dtype=dtype)

    def astype(self, type: type) -> Vec2:
        """Return a Vec2 object with x and y transformed to type."""
        # NOTE This method was implemented because routing ensures that
        # coordinates are given as floats by calling .astype(float) in line 957
        # in routing.py. This check is redundant because Placement and Vec2
        # themselves ensure correct typing of their attributes. Therefore,
        # in the libqudev updade following the merge of dev/placement, the
        # check in routing can be removed after which this method could also be
        # removed.
        log.warning('DeprecationWarning: The `astype` method is deprecated.')
        return Vec2(type(self.x), type(self.y))

    def __add__(
        self,
        other: Union[Vec2, np.ndarray],
    ) -> Union[Vec2, Tuple[Vec2]]:
        if isinstance(other, Vec2):
            return Vec2(self.x + other.x, self.y + other.y)
        elif isinstance(other, np.ndarray):
            # Before Vec2, a coordinate was a (2,) array, we implement
            # this to maintain compatibilibty with existing code.
            log.warning(
                'DeprecationWarning: Vec2 operations with arrays are'
                ' currently allowed but slated for removal in a future'
                ' version of qdl. Please use `Vec2`, `Placement`, and '
                '`MultiPlacement` directly.'
            )
            if other.shape == (2,):
                # assumed to represent (x, y)
                return Vec2(self.x + other[0], self.y + other[1])
            elif other.shape[1] == 2 and len(other.shape) == 2:
                # has shape: (N, 2)
                # assumed to represent a sequence of (x, y) coords
                # returns a tuple of Vec2 objects, similar to
                # MultiPlacement().positions
                return tuple(self + position for position in other)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def __sub__(
        self,
        other: Union[Vec2, np.ndarray],
    ) -> Union[Vec2, Tuple[Vec2]]:
        if isinstance(other, Vec2):
            return Vec2(self.x - other.x, self.y - other.y)
        elif isinstance(other, np.ndarray):
            # Before Vec2, a coordinate was a (2,) array, we implement
            # this to maintain compatibilibty with existing code.
            log.warn('DeprecationWarning: Vec2 operations with arrays are'
                     ' currently allowed but slated for removal in a future'
                     ' version of qdl. Please use `Vec2`, `Placement`, and '
                     '`MultiPlacement` directly.')
            if other.shape == (2,):
                # assumed to represent (x, y)
                return Vec2(self.x - other[0], self.y - other[1])
            elif other.shape[1] == 2:
                # has shape: (N, 2)
                # assumed to represent a sequence of (x, y) coords
                # returns a tuple of Vec2 objects, similar to
                # MultiPlacement().positions
                return tuple(self - position for position in other)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    # Vec2 * other
    def __mul__(
        self,
        other: Union[float, int, Vec2],
    ) -> Vec2:
        if isinstance(other, Vec2):
            return Vec2(self.x * other.x, self.y * other.y)
        if isinstance(other, (float, int)):
            return Vec2(self.x * other, self.y * other)
        return NotImplemented

    # other * Vec2
    def __rmul__(
        self,
        other: Union[float, int, Vec2],
    ) -> Vec2:
        return self * other

    # Vec2 @ Vec2
    def __matmul__(
        self,
        other: Vec2,
    ) -> float:
        return self.x * other.x + self.y * other.y

    # Vec2 / other
    def __truediv__(
        self,
        other: Union[float, int, Vec2],
    ) -> Vec2:
        if isinstance(other, Vec2):
            return Vec2(self.x / other.x, self.y / other.y)
        if isinstance(other, (float, int)):
            return Vec2(self.x / other, self.y / other)
        return NotImplemented

    # other / Vec2
    def __rtruediv__(
        self,
        other: Union[float, int],
    ) -> Vec2:
        return Vec2(other / self.x, other / self.y)

    # Vec2 ** exponent
    def __pow__(
        self,
        exponent: Union[float, int],
    ) -> Vec2:
        x = self.x ** exponent
        y = self.y ** exponent
        if isinstance(x, complex) or isinstance(y, complex):
            raise ValueError(f"Either: self.x ** exponent {x} or "
                             f"self.y ** exponent {y} causes a complex "
                             "number, which is not supported.")
        return Vec2(x, y)

    # -Vec2
    def __neg__(self) -> Vec2:
        return -1 * self

    def norm(
        self,
        power: int = 2,
    ) -> float:
        """Normalize the vector.

        Parameters
        ----------
        power : int, optional
            Power to use when calculating the norm.
            Defaults to 2.

        Returns
        -------
        result : float
            Norm.
        """
        return abs(self.x) ** power + abs(self.y) ** power
