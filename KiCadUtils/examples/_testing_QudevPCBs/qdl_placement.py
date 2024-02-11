# Copyright (c) 2023, Nicolas Thill

"""Classes and functions to define and process placement information."""

from dataclasses import dataclass, field
from typing import Union, Tuple, Optional

import logging
import numpy as np

from qdl_coordinate import Vec2
from qdl_helper import rotation_matrix

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Placement:
    """Define instructions needed to place an object in 2D.

    Parameters
    ----------
    position : qdl.coordinate.Vec2, optional
        x, y coordinates from origin.
        Defaults to qdl.coordinate.Vec2().
    rotation : float, optional
        Rotation angle in degrees, is mapped to [0, 360).
        Defaults to 0.0.
    scale_factor : float, > 0,
        Strictly positive factor by which object placement is scaled.
        Defaults to 1.0.
    reflect_x : bool, optional
        If True, reflection along x axis.
        Defaults to False.

    Attributes
    ----------
    position : qdl.coordinate.Vec2
        x, y coordinates from origin.
    rotation : float
        Rotation angle in degrees, mapped to [0, 360).
    scale_factor : float, > 0,
        Strictly positive factor by which object placement is scaled.
    reflect_x : bool
        If True, reflection along x axis.
    kwargs : dict
        JSON-serializable dictionary of the above attributes.

    Methods
    -------
    place(super_placement: Placement) -> Placement
        Returns a new placement which is 'self' placed within 'super_placement'
        with order of operations: reflections, rotation, scaling and finally
        translation.
    __getitem__
        Ensures Placement acts as a dictionary.
    """

    position: Vec2 = field(default=Vec2())
    rotation: float = field(default=0.0)
    scale_factor: float = field(default=1.0)
    reflect_x: bool = field(default=False)

    def __post_init__(self):
        """Validate and normalize attributes.

        Raise
        --------
        TypeError
            If any argument cannot be mapped the correct format.
        """
        # validate position
        object.__setattr__(self, "position", Vec2(*self.position))

        # validate rotation
        try:
            rotation = float(self.rotation % 360)
            object.__setattr__(self, "rotation", rotation)

        except (ValueError, TypeError) as error:
            raise ValueError(
                "Placement: 'rotation' must be such that it can be mapped to"
                + f" a float modulo 360. {self.rotation} fails."
            ) from error

        # validate scale_factor
        try:
            if self.scale_factor > 0:
                scale_factor = float(self.scale_factor)
            else:
                raise ValueError("Negative 'scale_factor' not allowed.")

            object.__setattr__(self, "scale_factor", scale_factor)

        except (TypeError, ValueError) as error:
            raise ValueError(
                "Placement: 'scale_factor' must be such that it can be mapped"
                + f" to a strictly positive float. {self.scale_factor} fails."
            ) from error

        # validate reflect_x
        if not isinstance(self.reflect_x, bool):
            raise ValueError(
                "Placement: 'reflect_x' must be of type boolean. "
                f"{self.reflect_x} of type {type(self.reflect_x)} fails."
            )

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def kwargs(self) -> dict:
        """JSON-serializable of kwargs that can generate self."""
        return {
            "position": tuple(self.position),
            "rotation": self.rotation,
            "scale_factor": self.scale_factor,
            "reflect_x": self.reflect_x,
        }

    @ property
    def positions(self) -> Tuple[Vec2, ...]:
        """Returns a tuple with a single element, self.position.

        Permits using 'for position in placement.positions:' with Placement
        just like with MultiPlacement.
        """
        return (self.position,)

    @property
    def placements(self) -> Tuple['Placement']:
        """A tuple containing only this Placement, self.

        Permits using 'for placement in placement.placements:' with Placement
        just like with MultiPlacement.
        """
        return (self,)

    @property
    def x(self) -> float:
        """Horizontal x component of position."""
        return self.position.x

    @property
    def y(self) -> float:
        """Vertiacal y component of position."""
        return self.position.y

    @property
    def direction(self) -> float:
        """Direction of the placement in radians."""
        return self.rotation * np.pi / 180

    def place(self, super_placement: 'Placement') -> 'Placement':
        """
        Places 'self' into 'super_placement'.

        Returns Placement which is 'self' placed within 'super_placement'
        with order of operations: reflections, rotation, scaling and finally
        translation.

        This can be thought of as:
        absolute_placement = relative_placement.place(super_placement)

        Parameters
        ----------
        super_placement: Placement
            Placement which 'self' is to be placed within

        Raises
        ------
        TypeError
            If super_placement not of type Placement.
        """
        if not isinstance(super_placement, Placement):
            raise TypeError(
                "To transform a placement, 'super_placement' must be of type"
                + f" Placement, not {type(super_placement)}."
            )

        position = np.array(self.position)
        rotation = self.rotation
        # Reflect x
        if super_placement.reflect_x:
            position = position * np.array([1, -1])
            rotation = - rotation % 360

        # Rotate
        rot_mat = rotation_matrix(
            super_placement.rotation * np.pi/180
        )
        position = rot_mat @ position
        rotation = (rotation + super_placement.rotation) % 360

        # Scale
        position = super_placement.scale_factor * position

        # Translate
        position = Vec2(*position) + super_placement.position

        # update scale_factor and reflect_x
        scale_factor = self.scale_factor * super_placement.scale_factor
        reflect_x = self.reflect_x ^ super_placement.reflect_x

        return Placement(position, rotation, scale_factor, reflect_x)


@dataclass(frozen=True)
class MultiPlacement:
    """Define instructions needed to place a set of objects in 2D.

    NOTE The purpose of this class is to provide a starting point for
    implementing speedups for methods involving objects which are placed
    many times, *i.e.* for tilings.

    Parameters
    ----------
    positions : Tuple[qdl.coordinate.Vec2, ...], optional
        tuple of x, y coordinates from origin.
        Defaults to (qdl.coordinate.Vec2(),).
    rotation : float, optional
        Rotation angle in degrees, is mapped to [0, 360).
        Defaults to 0.0.
    scale_factor : float, > 0,
        Strictly positive factor by which object placement is scaled.
        Defaults to 1.0.
    reflect_x : bool, optional
        If True, reflection along x axis.
        Defaults to False.

    Attributes
    ----------
    positions : Tuple[qdl.coordinate.Vec2, ...]
        Tuple of x, y coordinates from origin.
    rotation : float
        Rotation angle in degrees, mapped to [0, 360).
    scale_factor : float, > 0,
        Strictly positive factor by which object placement is scaled.
    reflect_x : bool, optional
        If True, reflection along x axis.
    kwargs : dict
        JSON-serializable dictionary of the above attributes.

    Methods
    -------
    place(super_placement: Placement) -> MultiPlacement
        Returns MultiPlacement which is 'self' placed within 'super_placement'
        with order of operations: reflections, rotation, scaling and finally
        translation.
    __getitem__
        Ensures Placement acts as a dictionary.
    """

    positions: Tuple[Vec2, ...] = field(default=(Vec2(),))
    rotation: float = field(default=0.0)
    scale_factor: float = field(default=1.0)
    reflect_x: bool = field(default=False)

    def __post_init__(self):
        """Validate and normalize attributes.

        Raise
        --------
        TypeError
            If any argument cannot be mapped the correct format.
        """
        # validate positions
        try:
            all_positions = []
            for position in self.positions:
                if not len(position) == 2:
                    raise ValueError(f"Position {position} has len != 2")
                all_positions.append(Vec2(*position))
        except Exception as error:
            raise ValueError("Placement: 'position' must be such that it can "
                             "be mapped to a list of Vec2 objects. "
                             f"{self.positions} fails."
                             ) from error

        object.__setattr__(self, "positions", tuple(all_positions))

        # validate rotation
        try:
            rotation = float(self.rotation % 360)
            object.__setattr__(self, "rotation", rotation)

        except (ValueError, TypeError) as error:
            raise ValueError(
                "Placement: 'rotation' must be such that it can be mapped to"
                + f" a float modulo 360. {self.rotation} fails."
            ) from error

        # validate scale_factor
        try:
            if self.scale_factor > 0:
                scale_factor = float(self.scale_factor)
            else:
                raise ValueError("Negative 'scale_factor' not allowed.")

            object.__setattr__(self, "scale_factor", scale_factor)

        except (TypeError, ValueError) as error:
            raise ValueError(
                "Placement: 'scale_factor' must be such that it can be mapped"
                + f" to a strictly positive float. {self.scale_factor} fails."
            ) from error

        # validate reflect_x
        if not isinstance(self.reflect_x, bool):
            raise ValueError(
                "Placement: 'reflect_x' must be of type boolean. "
                f"{self.reflect_x} of type {type(self.reflect_x)} fails."
            )

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def kwargs(self) -> dict:
        """JSON-serializable dictionary of kwargs that can generate self."""
        return {
            "positions": tuple(tuple(position) for position in self.positions),
            "rotation": self.rotation,
            "scale_factor": self.scale_factor,
            "reflect_x": self.reflect_x,
        }

    @property
    def placements(self) -> Tuple[Placement]:
        """A tuple of Placements, one for each position in self.positions."""
        return tuple(
            Placement(position=position, rotation=self.rotation,
                      scale_factor=self.scale_factor,
                      reflect_x=self.reflect_x)
            for position in self.positions
        )

    def place(self, super_placement: Placement) -> 'MultiPlacement':
        """
        Places 'self' into 'super_placement'.

        Returns Placement which is 'self' placed within 'super_placement'
        with order of operations: reflections, rotation, scaling and finally
        translation.

        Parameters
        ----------
        super_placement : Placement
            Placement which 'self' is to be placed within

        Raises
        ------
        TypeError
            If super_placement not of type Placement.
        """
        if not isinstance(super_placement, Placement):
            raise TypeError(
                "To transform a MultiPlacement, 'super_placement' must be of"
                f" type  Placement, not {type(super_placement)}."
            )

        positions = np.array(list(position for position in self.positions))
        rotation = self.rotation

        # Reflect x
        if super_placement.reflect_x:
            positions = positions * np.array([1, -1])
            rotation = - rotation % 360

        # Rotate
        rot_mat = rotation_matrix(
            super_placement.rotation * np.pi/180
        )
        positions = np.array([rot_mat @ position for position in positions])
        rotation = (rotation + super_placement.rotation) % 360

        # Scale
        positions = super_placement.scale_factor * positions

        # Translate
        positions = tuple(Vec2(*position) + super_placement.position
                          for position in positions)

        # update scale_factor and reflect_x
        scale_factor = self.scale_factor * super_placement.scale_factor
        reflect_x = self.reflect_x ^ super_placement.reflect_x

        return MultiPlacement(positions, rotation, scale_factor, reflect_x)


def _process_placement_arguments(
            placement: Optional[Union[Placement, MultiPlacement]] = None,
            position: Optional[np.ndarray] = None,
            rotation: Optional[float] = None,
            scale_factor: Optional[float] = None,
            reflect_x: Optional[bool] = None,
        ) -> Union[Placement, MultiPlacement]:
    """Verify input and output consistent (Multi)Placement.

    Processes arguments of functions which were expanded to support placement.
    Checks whether arguments provided are of type Placement directly, or
    are previous arguments that can generate a Placement or MultiPlacement.
    If both placement and other arguments are passed, inconsistency raises
    ValueError. Further, warnings are given that the use of previous arguments
    is deprecated.

    Parameters
    ----------
    placement : Placement | MultiPlacement, optional
        Instructions needed to place an object in 2D., by default None
    position : np.ndarray, optional
        Has shape (2,) or (N, 2), by default None
    rotation : float, optional
        Rotation angle in degree, by default 0.0
    scale_factor : float, optional
        Scaling factor, by default 1.0
    reflect_x : bool, optional
        Whether to reflect along x-axis, by default False

    Raises
    ------
    ValueError
        If 'placement' is passed but doesn't match another argument.
    TypeError
        If position argument does not have shape (N, 2) or (2,)

    Returns
    -------
    placement : Placement or MultiPlacement
        Correct placement object corresponding to provided arguments.
    """
    # NOTE This function might become redundant once support of position,
    # rotation, scale_factor, reflect_x is abandoned for use of placement.

    if position is not None:  # cast position to array
        try:
            positions = np.array(position)
        except ValueError as error:
            raise ValueError(f"Position {position} cannot be cast to array.")\
                from error
    else:
        positions = None

    # If we have a placement, then check that its arguments match the
    # other parameters
    if placement is not None:
        if not isinstance(placement, (Placement, MultiPlacement)):
            ValueError("placement must have type Placement or MultiPlacement,"
                       f" not {type(placement)}.")
        # If we have a position, then check it
        if positions is not None:
            # ensure shape is (N, 2)
            if positions.shape == (2,):
                positions = np.array([positions])
            # check that number of positions matches
            if len(positions) != len(placement.positions):
                raise ValueError(f"Provided position {position}, which "
                                 f"transforms to {positions}, doesn't "
                                 "have the same number of positions as "
                                 f"placement.positions {placement.positions}.")
            # ensure each position matches each in placement.positions
            for position_pos, placement_pos \
                    in zip(positions, placement.positions):
                if Vec2(*position_pos) != placement_pos:
                    raise ValueError(f"Provided position {position}, which "
                                     f"transforms to {positions}, doesn't "
                                     "match placement.position "
                                     f"{placement.positions}.")
        # If we have a rotation, then check that it matches
        if (rotation is not None and
                rotation != placement.rotation):
            raise ValueError(f"Provided rotation {rotation} doesn't match "
                             f"placement.rotation {placement.rotation}.")
        # If we have a scale_factor, then check that it matches
        if (scale_factor is not None and
                scale_factor != placement.scale_factor):
            raise ValueError(f"Provided scale_factor {scale_factor} doesn't "
                             "match placement.scale_factor "
                             f"{placement.scale_factor}.")
        # If we have a reflect_x, then check that it matches
        if (reflect_x is not None and
                reflect_x != placement.reflect_x):
            raise ValueError(f"Provided reflect_x {reflect_x} doesn't match "
                             f"placement.reflect_x {placement.reflect_x}.")
        # At this point, placement matches all other provided arguments
        pass
    # Otherwise, we do not have a placement and must build it from the
    # other parameters
    else:
        non_pos_kwargs = {}  # use as kwargs if not None
        if rotation is not None:
            non_pos_kwargs['rotation'] = rotation
        if scale_factor is not None:
            non_pos_kwargs['scale_factor'] = scale_factor
        if reflect_x is not None:
            non_pos_kwargs['reflect_x'] = reflect_x

        # If no positions, use the default
        if positions is None:
            placement = Placement(position=Vec2(),
                                  **non_pos_kwargs)
        # Otherwise, parse the positions
        else:
            # If positions is a single 2D array
            if positions.shape == (2,):
                placement = Placement(position=Vec2(*positions),
                                      **non_pos_kwargs)
            # Or if positions is multiple 2D arrays
            elif (len(positions.shape) == 2
                    and positions.shape[1] == 2):
                positions = tuple(Vec2(*p) for p in positions)
                placement = MultiPlacement(positions=positions,
                                           **non_pos_kwargs)
            # Otherwise, positions is incorrectly formatted
            else:
                raise TypeError(f"Position {positions}")

    return placement
