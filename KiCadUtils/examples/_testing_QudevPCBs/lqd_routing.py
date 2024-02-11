# Copyright (c) 2021, Marek Pechal and ETH Zurich
# Copyright (c) 2022-2023, Graham J. Norris and ETH Zurich
# helper functions for routing of CPWs

# Use generic type annotations
from __future__ import annotations
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from qdl_placement import _process_placement_arguments

logger = logging.getLogger(__name__)

K = np.array([[0.0, 1.0], [-1.0, 0.0]])


@dataclass
class RouteElementMetadata:
    """Class for organizing RouteElement metadata."""
    airbridges: Optional[List[AirbridgeInfo]] = None

    def get_airbridge_positions(
        self,
        length: float,
    ) -> List[Tuple[float, Dict]]:
        """Get a list of merged positions.

        Attributes
        ----------
        length : float
            Length to use when converting relative positions into
            absolute ones.

        Returns
        -------
        position_list : list[tuple(float, dict)]
            List of tuples containing airbridge positions and keyword
            arguments (None for defaults).
        """
        output_tuples = []
        if self.airbridges is not None:
            for ab in self.airbridges:
                ab_positions = ab.get_merged_positions(length)
                for pos in ab_positions:
                    output_tuples.append((pos, ab.airbridge_kwargs))
        return sorted(output_tuples, key=lambda tup: tup[0])


@dataclass
class AirbridgeInfo:
    """Class for storing airbridge information."""
    relative_positions: List[float] = field(default_factory=list)
    absolute_positions: List[float] = field(default_factory=list)
    airbridge_kwargs: Optional[Dict] = None

    def get_merged_positions(
        self,
        length: float,
    ) -> List[float]:
        """Get a list of merged positions.

        Attributes
        ----------
        length : float
            Length to use when converting relative positions into
            absolute ones.

        Returns
        -------
        position_list : list[float]
            List of absolute positions along this segment.
        """
        self._ensure_lists()
        abs_rel_pos_list = list(
            np.asarray(self.relative_positions) * length
        )
        return sorted(self.absolute_positions + abs_rel_pos_list)

    def _ensure_lists(self):
        """Ensure that positions are lists not numpy arrays.
        """
        # TODO log/warn when correction made?
        if isinstance(self.absolute_positions, np.ndarray):
            self.absolute_positions=self.absolute_positions.tolist()
        if isinstance(self.relative_positions, np.ndarray):
            self.relative_positions=self.relative_positions.tolist()



##########################
# Various helper functions
##########################

def unit_vec(angle):
    """
    Returns unit vector at a given angle (in radians) with respect to the x axis.

    Parameters
    ----------
    angle : float
    """
    return np.array([np.cos(angle), np.sin(angle)])

def vec_direction(v):
    """
    Returns angle (in radians) of a given vector with respect to the x axis.

    Parameters
    ----------
    v : 2d numpy array, list or tuple
    """
    return np.angle(np.dot([1, 1j], v))

def rotate_vec(v, angle):
    """
    Rotates a vector by a given angle.

    Parameters
    ----------
    v : 2d numpy array, list or tuple
    angle : float
    """
    return v * np.cos(angle) - K.dot(v) * np.sin(angle)

def angle_between(u, v):
    """
    Returns angle (in radians) between two given vectors.

    Parameters
    ----------
    u, v : 2d numpy array, list or tuple
    """
    return np.angle(np.dot([1, 1j], v) / np.dot([1, 1j], u))

def turn_angle(initial_point, final_point, direction):
    """
    Returns angle (in radians) of a turn connecting two given points,
    with a given initial direction.

    Parameters
    ----------
    initial_point, final_point : 2d numpy array
    direction : 2d numpy array
        vector representing the initial direction (size does not matter)
    """
    return 2 * angle_between(direction, final_point - initial_point)

def make_arc(initial_point, radius, e, n, angle):
    """
    Returns the list of points of an arc.

    Parameters
    ----------
    initial_point: 2d numpy array
    radius: float
    e, n: 2d numpy array
        unit vectors tangent and normal (pointing towards the arc center) to
        the arc at its initial point
    angle: float
    """
    return np.array([
        initial_point + radius * (e * np.sin(u) + n * (1 - np.cos(u)))
        for u in np.linspace(0, angle, 41)[1:]])


class RouteElement:
    """
    Class representing an element of a Route.

    Attributes
    ----------
    metadata : RouteElementMetadata
        Metadata related to this route element.
    """

    def __init__(
        self,
        metadata: Optional[RouteElementMetadata] = None,
        **kwargs
    ):
        """Create a RouteElement.

        Parameters
        ----------
        metadata : RouteElementMetadata
            Metadata related to this route element.
        **kwargs : Optional
           Other attributes to set for this object.
        """
        if metadata is None:
            metadata = RouteElementMetadata()
        self.metadata = metadata
        for key, value in kwargs.items():
            setattr(self, key, value)



class Turn(RouteElement):
    """
    Class representing a circular arc.

    Attributes
    ----------
    radius : float
        Turn radius.
    angle : float
        Angle of rotation of the path in radians relative
        to the current direction. For example, an angle of
        π/2 corresponds to a 90-degree turn to the left.
    length : float
        Length of the turn.
    """
    def __init__(
        self,
        radius,
        angle,
        metadata: Optional[RouteElementMetadata] = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        radius : float
            Turn radius.
        angle : float
            Angle of rotation of the path in radians relative
            to the current direction. For example, an angle of
            π/2 corresponds to a 90-degree turn to the left.
        """
        self.radius = radius
        self.angle = angle
        self.length = abs(angle) * radius
        super().__init__(
            metadata=metadata,
            **kwargs,
        )

    def __repr__(self) -> str:
        """Return a representation of this turn."""
        return (f"Turn({self.radius}, {self.angle}, {self.metadata})")

    def __str__(self) -> str:
        """Return a string representation of this turn."""
        pi_angle = self.angle / np.pi
        return(
            f"Turn, radius {self.radius}, angle {pi_angle} π, "
            f"{self.metadata}"
        )


class Segment(RouteElement):
    """
    Class representing a straight segment.

    Attributes
    ----------
    length : float
        Length of the segment.
    """
    def __init__(
        self,
        length,
        metadata: Optional[RouteElementMetadata] = None,
        **kwargs
    ):
        """Create a straight segment.

        Parameters
        ----------
        length : float
            Length of the segment.
        **kwargs : optional other attributes to set
        """
        self.length = length
        super().__init__(
            metadata=metadata,
            **kwargs,
        )

    def __repr__(self) -> str:
        """Return a representation of this segment."""
        # Ignores kwargs
        return (f"Segment({self.length}, {self.metadata})")

    def __str__(self) -> str:
        """Return a human-readable string representation of this segment."""
        return(f"Segment, length {self.length:.2f}, {self.metadata}")


class RouteDescription:
    """
    Class describing a path. Contains a list of components which can be
    instances of Turn or Segment.

    Attributes
    ----------
    elements : list[RouteElement]
        List of RouteElements (Segment or Turn).
    initial_position : array-like[2], optional
        Initial position of Route. Defaults to None.
    initial_direction : float, optional
        Initial direction of the Route in radians. Defaults to None.
    metadata : dict, optional
        Dictionary of additional parameters. Defaults to {}.
    """
    def __init__(
        self,
        *args,
        initial_position: Optional[List[float]] = None,
        initial_direction: Optional[float] = None,
        initial_placement: Optional[Placement] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        *args : variable-length arguments list
            Either, no arguments: then the list of elements is
            initialized as empty; or a single argument: the list of
            elements.
        initial_position : array-like[2], optional
            Initial position of Route. Defaults to None.
        initial_direction : float, optional
            Initial direction of the Route in radians. Defaults to None.
        initial placement : Placement, optional
            Starting position and direction of the CPW line.
            Defaults to qdl.placement.Placement()
        metadata : Dict, optional
            Dictionary of additional parameters. Defaults to {}.
        """
        initial_rotation = None if initial_direction is None \
            else np.rad2deg(initial_direction)
        self.initial_placement = _process_placement_arguments(
            placement=initial_placement,
            position=initial_position,
            rotation=initial_rotation,
        )
        if args:
            self.elements = args[0]
        else:
            self.elements = []
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    # @property
    # def initial_position(self):
    #     return self.initial_placement.position
    #
    # @property
    # def initial_direction(self):
    #     return self.initial_placement.direction

    def add(self, element):
        """
        Appends element to path.

        Parameters
        ----------
        element: Turn or Segment
        """
        self.elements.append(element)

    def __add__(self, other):
        return RouteDescription(
            self.elements + other.elements,
            initial_placement=self.initial_placement,
            # TODO: imperfect, ignores metadata in other
            # Should merge together instead?
            metadata=self.metadata,
        )

    def length(self):
        """
        Returns the length of the path.
        """
        res = 0.0
        for element in self.elements:
            if isinstance(element, Segment):
                res += element.length
            elif isinstance(element, Turn):
                res += element.length
            else:
                raise ValueError('invalid element of RouteDescription: '
                    + str(element))
        return res

    def point(self, d):
        """
        Returns point a given distance d along the path.

        Parameters
        ----------
        d : float
        """

        pt = self.initial_position
        e = self.initial_direction_vector

        for element in self.elements:
            if isinstance(element, Segment):
                if d < element.length:
                    # specified point lies in the current segment
                    return pt + d * e, vec_direction(e)
                # get starting point for the next element (direction stays)
                pt = pt + e * element.length
                d -= element.length
            elif isinstance(element, Turn):
                n = -K.dot(e) * element.angle / abs(element.angle)
                angle = abs(element.angle)
                radius = element.radius
                if d < radius*angle:
                    # specified point lies in the current turn
                    c = np.dot(e, K.dot(n))
                    return (make_arc(pt, radius, e, n, d / radius)[-1],
                        vec_direction(e) + c * d / radius)
                # get starting point and direction for the next element
                pt = make_arc(pt, radius, e, n, angle)[-1]
                e = rotate_vec(e, element.angle)
                d -= radius * angle

    def __repr__(self) -> str:
        """Return a representation of this RouteDescription."""
        # TODO ignores metadata
        elements_repr = ""
        for element in self.elements:
            elements_repr += element.__repr__() + ","
        return (
            f"RouteDescription([{elements_repr}],"
            f"{self.initial_placement})"
        )

    def __str__(self) -> str:
        """Return a string representation of this RouteDescription."""
        elements_str = ""
        for element in self.elements:
            elements_str += element.__str__() + "; "
        return(
            f"RouteDescription, Elements [{elements_str}]; "
            f"initial placement {self.initial_placement}; "
        )

    def simplify(self):
        """
        Simplifies the route by combining consecutive straight segments
        and arcs with identical radii.

        Returns
        -------
        route : RouteDescription
            The simplified path.
        """
        route = RouteDescription(
            initial_placement=self.initial_placement,
            metadata=copy.deepcopy(self.metadata),
        )
        for element in self.elements:
            if (route.elements and isinstance(route.elements[-1], Segment)
                    and isinstance(element, Segment)):
                # both current and previous element are segments
                route.elements[-1].length += element.length
                continue
            if (route.elements and isinstance(route.elements[-1], Turn) and
                    isinstance(element, Turn)
                    and route.elements[-1].radius == element.radius
                    and route.elements[-1].angle * element.angle > 0):
                # both current and previous element are turns with
                # the same radius and in the same direction
                route.elements[-1].angle += element.angle
                continue
            route.add(element)
        return route

    def add_airbridges(
        self,
        end_separation: float = 200,
        airbridge_separation: float = 1000,
        airbridge_segment_kwargs: Optional[Dict] = None,
        airbridge_turn_kwargs: Optional[Dict] = None,
    ) -> RouteDescription:
        """Adds airbridges to the route.

        Replaces n*π/2 turns with airbridge-covered turns and segments
        with airbridge-covered segments.

        Parameters
        ----------
        end_separation : float, optional
            Minimum distance of an airbridge from either end of a
            straight segment. Defaults to 200.
        airbridge_separation : float, optional
            Distance between airbridges along a straight segment.
            Defaults to 1000.
        airbridge_segment_kwargs : dict, optional
            Additional keyword arguments for the airbridge segment
            constructor. Defaults to None, meaning {}.
        airbridge_turn_kwargs : dict, optional
            Additional keyword arguments for the airbridge turn
            constructor. Defaults to None, meaning {}.

        Returns
        -------
        RouteDescription
            The path with airbridge segments.
        """
        route = RouteDescription(
            initial_placement=self.initial_placement,
            metadata=copy.deepcopy(self.metadata),
        )
        for element in self.elements:
            # Replace n*π/2 turns by airbridge turns
            # TODO generalize to other angles and take the turn radius
            # into account
            if (
                isinstance(element, Turn)
                and np.isclose(np.mod(element.angle, np.pi/2), 0)
            ):
                pi_half_count = int(2*np.abs(element.angle)/np.pi)
                if pi_half_count == 2:
                    rel_pos = [0.25, 0.75]
                # π/2 turn and fallback
                else:
                    rel_pos = [0.5]
                route.add(Turn(
                    radius=element.radius,
                    angle=element.angle,
                    metadata=RouteElementMetadata(
                        airbridges=[
                            AirbridgeInfo(
                                relative_positions=rel_pos,
                                airbridge_kwargs=airbridge_turn_kwargs,
                            )
                        ],
                    ),
                ))
            # Add airbridges to segments with enough space for
            # airbridges
            elif (
                isinstance(element, Segment)
                and element.length > 2*end_separation
            ):
                # If length > 2x end sep, can add one AB
                airbridge_count = 1
                # If remaining length > ab_sep, can add more ABs
                remaining_length = element.length - 2*end_separation
                airbridge_count += np.floor(
                    remaining_length / airbridge_separation)
                # List from 0 to length of positions
                ab_positions = (
                    airbridge_separation
                    * centered_range(airbridge_count)
                    + element.length/2)
                route.add(Segment(
                    length=element.length,
                    metadata=RouteElementMetadata(
                        airbridges=[
                            AirbridgeInfo(
                                absolute_positions=ab_positions,
                                airbridge_kwargs=airbridge_segment_kwargs,
                            )
                        ],
                    ),
                ))
            else:
                route.add(element)
        return route

    def get_airbridge_positions(self) -> List[Tuple[float, Dict]]:
        """Get positions and types of airbridges along this Route.

        Returns
        -------
        position_list : list[tuple(float, dict)]
            List of tuples containing airbridge distances from the start
            of the Route and keyword arguments for the airbridge
            constructor (None for defaults).
        """
        output_tuples = []
        current_length = 0.0
        for element in self.elements:
            if element.metadata.airbridges is not None:
                ab_list = element.metadata.get_airbridge_positions(
                        length=element.length,
                    )
            else:
                ab_list = []
            for ab_pos, ab_kwargs in ab_list:
                output_tuples.append(
                    (ab_pos + current_length, ab_kwargs)
                )
            current_length += element.length
        return output_tuples

    @classmethod
    def meander(cls, w, dL, bend_radius, mode):
        """
        Constructs RouteDescription instance representing a set of meanders.

        Parameters
        ----------
        w : float
            Distance between the initial and final point.
        dL : float
            Excess length (difference between the length of the
            meandered line and the straight distance w)
        bend_radius : float
            Bend radius.
        mode : str
            Specification of the meanders. Can take values:
                '<' ... meanders pointing to the left
                '>' ... meanders pointing to the right
                'S' ... symmetric meanders start to the left
                'Z' ... symmetric meanders start to the right

        Returns
        -------
        RouteDescription
        """

        if dL < 0:
            raise ValueError('specified length too short')

        if mode == 'S':
            return cls._meander_symmetric(w, dL, bend_radius, turn_right=False)
        elif mode == 'Z':
            return cls._meander_symmetric(w, dL, bend_radius, turn_right=True)
        else:
            return cls._meander_singlesided(w, dL, bend_radius, mode)

    @classmethod
    def _meander_symmetric(cls, w, dL, bend_radius, turn_right):
        meander_count = int((w - 2 * bend_radius) / (2 * bend_radius))       # number of meanders to be made
        Lm = dL + 2 * bend_radius * (meander_count + 1)            # length of meanders

        def get_meander_length(h):
            # find meander length from meander height
            res = 0.0
            if h >= 2 * bend_radius:
                res += 2 * (np.pi * bend_radius + h - 2 * bend_radius)
            else:
                res += 2 * (
                    (np.pi + 2 - 4 * np.arctan(1 - h / (2 * bend_radius)))
                      * bend_radius - h)
            if h >= bend_radius:
                res += (
                    2 * (meander_count - 1)
                    * (0.5 * np.pi * bend_radius + h - bend_radius))
            else:
                res += 2 * (meander_count - 1) * (
                    (0.5 * np.pi + 1 - 2 * np.arctan(1 - h / bend_radius))
                    * bend_radius - h)
            return res
        explicit_meander_length = (
            (np.pi * (meander_count + 1) + 2 * (meander_count - 1))
            * bend_radius
        )
        if Lm > explicit_meander_length:
            # if the meanders are longer than pi*bend_radius*(meander_count+1)
            # + 2*bend_radius*(meander_count-1),
            # the height can be calculated explicitly
            h = (
                (Lm - explicit_meander_length) / (2 * meander_count)
                + 2 * bend_radius
            )
        else:
            # if the meander is shorter than that, we need to find the
            # height by inverting the function `get_meander_length`
            X = np.linspace(0, 2 * bend_radius, 401)
            Y = [get_meander_length(h) for h in X]
            h = interp1d(Y, X)(Lm)
        if h < 2 * bend_radius:
            theta1 = np.pi / 2 - 2 * np.arctan(1 - h / (2 * bend_radius))
        else:
            theta1 = np.pi / 2
        if h < bend_radius:
            theta2 = np.pi / 2 - 2 * np.arctan(1 - h / bend_radius)
        else:
            theta2 = np.pi / 2
        l1 = np.sqrt((h - 2 * bend_radius * (1 - np.cos(theta1))) ** 2
            + (2 * bend_radius * (1 - np.sin(theta1))) ** 2)
        l2 = np.sqrt((2 * h - 2 * bend_radius * (1 - np.cos(theta2))) ** 2
            + (2 * bend_radius * (1 - np.sin(theta2))) ** 2)

        if not turn_right:
            res = cls()         # initialize a PathDescription instance
            res.add(            # first straight segment leading to meanders
                Segment((w - 2 * bend_radius * (meander_count + 1)) / 2))
            res.add(Turn(bend_radius, theta1))
            res.add(Segment(l1))
            res.add(Turn(bend_radius, -theta1))
            for i in range(meander_count - 1):
                res.add(Turn(bend_radius, -theta2))
                res.add(Segment(l2))
                res.add(Turn(bend_radius, theta2))
                theta2 = -theta2
                theta1 = -theta1
            res.add(Turn(bend_radius, -theta1))
            res.add(Segment(l1))
            res.add(Turn(bend_radius, theta1))
            res.add(            # last straight segment leading to end point
                Segment((w - 2 * bend_radius * (meander_count + 1)) / 2))
        else:
            res = cls()         # initialize a PathDescription instance
            res.add(            # first straight segment leading to meanders
                Segment((w - 2 * bend_radius * (meander_count + 1)) / 2))
            res.add(Turn(bend_radius, -theta1))
            res.add(Segment(l1))
            res.add(Turn(bend_radius, theta1))
            for i in range(meander_count - 1):
                res.add(Turn(bend_radius, +theta2))
                res.add(Segment(l2))
                res.add(Turn(bend_radius, -theta2))
                theta2 = -theta2
                theta1 = -theta1
            res.add(Turn(bend_radius, theta1))
            res.add(Segment(l1))
            res.add(Turn(bend_radius, -theta1))
            # last straight segment leading to end point
            res.add(Segment((w - 2 * bend_radius * (meander_count + 1)) / 2))
        return res


    @classmethod
    def _meander_singlesided(cls, w, dL, bend_radius, mode):
        meander_count = int(w / (4 * bend_radius))
        Lm = dL / (2 * meander_count) + 2 * bend_radius  # length of each meander

        def get_theta(h):
            # find turning angle from meander height (distance to which the
            # meanders are "sticking out")
            if h >= 2 * bend_radius:
                return np.pi / 2
            else:
                return (
                    np.arctan(2 * bend_radius / (2 * bend_radius - h))
                    - np.arcsin(
                        (2 * bend_radius - h)
                        / np.sqrt(
                            8 * bend_radius ** 2 + h ** 2
                            - 4 * h * bend_radius)
                        )
                    )

        def get_meander_length(h):
            # find meander length from meander height
            return 2*get_theta(h) * bend_radius + abs(2 * bend_radius - h)

        if Lm > np.pi * bend_radius:
            # if the meander is longer than pi*bend_radius, the turning angle is
            # simply pi/2 and the height is easy to calculate
            h = 2 * bend_radius + Lm - np.pi * bend_radius
        else:
            # if the meander is shorter than pi*bend_radius, we need to find the
            # height by inverting the function `get_meander_length`
            X = np.linspace(0, 2 * bend_radius, 81)
            Y = [get_meander_length(h) for h in X]
            h = interp1d(Y, X)(Lm)

        theta = get_theta(h)
        if mode == '>':     # choose the right sign for the turn angle
            theta = -theta  # based on the direction of the meander

        res = cls()         # initialize a RouteDescription instance
        res.add(            # first straight segment leading to meanders
            Segment((w - 4 * bend_radius * meander_count) / 2))

        for i in range(meander_count):
            res.add(Turn(bend_radius, theta))
            if h > 2 * bend_radius:
                ls = h - 2 * bend_radius
            else:
                ls = np.sqrt((h - 2 * bend_radius * (1 - np.cos(theta))) ** 2
                    + (2 * bend_radius * (1 - abs(np.sin(theta)))) ** 2)
            res.add(Segment(ls))
            res.add(Turn(bend_radius, -2 * theta))
            res.add(Segment(ls))
            res.add(Turn(bend_radius, theta))

        res.add(            # last straight segment leading to end point
            Segment((w - 4 * bend_radius * meander_count) / 2))

        return res

    @classmethod
    def straight_between_points(cls, pt1, dir1, pt2, dir2):
        """
        Constructs RouteDescription instance representing a straight
        connection between two points.

        Parameters
        ----------
        pt1 : 2d numpy array
            Initial point.
        dir1 : float
            Initial direction.
        pt2 : 2d numpy array
            Final point.
        dir2 : float
            Final direction.

        Returns
        -------
        RouteDescription
        """
        tol = 1e-6
        if abs(np.exp(1j * dir1) + np.exp(1j * dir2)) > tol:
            raise ValueError(
                "Ports whose directions are not opposite "
                "cannot be connected by a straight route"
            )
        if abs(np.dot(unit_vec(dir1), K.dot(pt2 - pt1))) > tol:
            raise ValueError(
                "Ports whose positions are not collinear with their "
                "direction cannot be connected by a straight route. "
                f"Ports: {pt1}, {pt2}"
            )
        return cls([Segment(np.linalg.norm(pt2 - pt1))])

    @classmethod
    def bend_between_points(cls, pt1, dir1, pt2, dir2, radius):
        """
        Constructs RouteDescription instance representing a connection
        with a single bend between two points.

        Parameters
        ----------
        pt1 : 2d numpy array
            Initial point.
        dir1 : float
            Initial direction.
        pt2 : 2d numpy array
            Final point.
        dir2 : float
            Final direction.
        radius : float

        Returns
        -------
        RouteDescription
        """
        e1 = unit_vec(dir1)     # initial direction vector
        e2 = unit_vec(dir2)     # final direction vector
        n1 = K.dot(e1)          # initial normal vector
        n2 = K.dot(e2)          # final normal vector
        if np.dot(e2, n1) > 0:  # sign of normal vectors is chosen so they
            n1 = -n1            # point to the inside of the bend
        if np.dot(e1, n2) > 0:
            n2 = -n2
        # find intersection `intp` of the two segments to be connected, i.e.
        # find numbers x, y such that pt1+x*e1 = pt2+y*e2 =: intp
        x, y = np.linalg.solve(np.array([-e1, e2]).transpose(), pt1 - pt2)
        intp = pt1 + x * e1
        # find center point of the bend
        ctr = intp + radius * (n1 + n2) / (1 + np.dot(n1, n2))

        tp1 = pt1 + e1 * np.dot(e1, ctr - pt1) # find tangent points to the bend
        tp2 = pt2 + e2 * np.dot(e2, ctr - pt2)
        a = abs(turn_angle(tp1, tp2, e1))      # find the turn angle

        return cls([
            Segment(np.linalg.norm(tp1 - pt1)),
            Turn(radius, a * np.dot(e1, K.dot(n1))),
            Segment(np.linalg.norm(tp2 - pt2))
            ])

    @classmethod
    def between_points(cls, pt1, dir1, pt2, dir2, radius,
            length = None, cmd = '', meander_radius = None):
        """
        Constructs RouteDescription instance representing a general
        connection between two points.

        Parameters
        ----------
        pt1 : 2d numpy array
            Initial point.
        dir1 : float
            Initial direction.
        pt2 : 2d numpy array
            Final point.
        dir2 : float
            Final direction.
        radius : float
            Default radius of all bends.
        length : float or None, optional
            Length of the path if it should be enforced or None if it
            should not. Defaults to None. Only has an effect if a
            meander is specified in the routing command `cmd`.
        cmd : str, optional
            Command describing the routing of the path. Defaults to an
            empty string. For details on the syntax, see the examples in
            `docs/examples/routing.py`
        meander_radius : float or None, optional
            Meander radius if it should override the default bend radius,
            otherwise None. Defaults to None.

        Returns
        -------
        RouteDescription
        """
        tol = 1e-6
        # straight connection (possibly with meanders)
        if cmd in ['', '<', '>', 'S', 'Z']:
            # take care of cases when this is not possible
            if abs(np.exp(1j * dir1) + np.exp(1j * dir2)) > tol:
                raise ValueError(
                    "Ports whose directions are not opposite "
                    "cannot be connected by a straight route"
                )
            if abs(np.dot(unit_vec(dir1), K.dot(pt2 - pt1))) > tol:
                print(pt1, pt2)
                raise ValueError(
                    "Ports whose positions are not collinear "
                    "with their direction cannot be connected "
                    "by a straight route"
                )

            if cmd == '': # without meanders
                return cls([Segment(np.linalg.norm(pt2 - pt1))])
            else: # with meanders
                L0 = np.linalg.norm(pt2 - pt1)
                if meander_radius is not None:  # use explicitly specified
                    Rm = meander_radius         # meander radius,
                else:                           # otherwise use the generic
                    Rm = radius                 # bend radius
                mroute = cls.meander(L0, length - L0, Rm, cmd)
                return mroute

        def parse_command(cmd, d, pt, e, ptt, et, bend_radius):
            # Parses half of the routing command into a RouteDescription object
            # d ... direction (+1 or -1)
            # pt ... start point
            # e ... start direction
            # ptt ... target point
            # et ... target direction
            # bend_radius ... default bend radius

            def choose_direction(en, pt, e, ptt, et):
                # Chooses the turn direction for commands `x` and `y` which
                # do not explicitly say if the path should turn left or
                # right. The direction is chosen to be "towards the end point".
                # en ... new direction vector (initial guess)
                # pt ... start point
                # e ... start direction
                # ptt ... target point
                # et ... target direction
                c = np.dot(ptt - pt, en)
                if c != 0:               # first choice: choose direction to
                    en = en * c / abs(c) # point "towards the target"
                else:                    # if not possible (direction perp.
                    c = np.dot(e, en)    # to ptt-pt, second choice:
                    if c != 0:           # choose direction which is better
                        en = en * c / abs(c) # aligned with current direction
                    else:                  # if not possible (direction
                        c = np.dot(et, en) # perpendicular), third choice:
                        if c != 0:   # choose direction which is better aligned
                            en = en * c / abs(c) # with the final direction
                        else:
                            raise ValueError('direction choice failed')
                return en

            ex = np.array([1.0, 0.0])
            ey = np.array([0.0, 1.0])

            currentPt = pt.astype(float) # start at the initial point
            path = cls()                 # and with an empty RouteDescription

            # set initial index according to if we're going forward or backward
            i = 0 if d == 1 else len(cmd)-1
            bracket = False # flag for whether we're inside a bracket
            auxcmd = None
            s = ''

            while (d == 1 and i < len(cmd)) or (d == -1 and i >= 0):
                # while we haven't finished parsing the whole string

                if (d == 1 and cmd[i] == '[') or (d == -1 and cmd[i] == ']'):
                    # opening bracket
                    s = ''
                    bracket = True

                elif (d == 1 and cmd[i] == ']') or (d == -1 and cmd[i] == '['):
                    # closing bracket

                    # If bracket of the form 'R...', set the bend radius
                    if s[0] == 'R':
                        bend_radius = float(s[1:])

                    # If bracket of the form 'r...' or 'l...', parse as
                    # a turn
                    elif s[0] in ['l', 'r']:
                        # Extract individual bend radius
                        if s.find('R') >= 0:
                            s, Rs = s.split('R')
                            bend_radius2 = float(Rs)
                        # Otherwise default radius
                        else:
                            bend_radius2 = bend_radius

                        # substring after 'l' or 'r' (and before 'R') is the
                        # turn angle in degrees
                        a = float(s[1:]) * np.pi / 180

                        # choose sign of normal vector based on turn direction
                        if s[0] == 'r':
                            n = K.dot(e)
                        else:
                            n = -K.dot(e)

                        # move current point
                        currentPt += bend_radius2 * (
                            e * np.sin(a) + n * (1 - np.cos(a)))
                        # add turn to RouteDescription
                        path.add(Turn(bend_radius2, a * np.dot(e, K.dot(n))))
                        # update direction vector
                        e = e * np.cos(a) + n * np.sin(a)

                        # if aux command is available, attach to the object
                        if auxcmd is not None: path.elements[-1].cmd = auxcmd
                        auxcmd = None

                    else:
                        # remaining case: bracket contect is a number specifying
                        # the length of a straight segment

                        # move current point
                        currentPt = currentPt + e * float(s)
                        # add segment to RouteDescription
                        path.add(Segment(float(s)))

                        # if aux command is available, attach to the object
                        if auxcmd is not None: path.elements[-1].cmd = auxcmd
                        auxcmd = None
                    bracket = False

                elif bracket:
                    # if we're inside a bracket, collect content into `s`
                    if d == 1:
                        s = s + cmd[i]
                    else:
                        s = cmd[i] + s

                elif cmd[i] in ['x', 'y']:
                    # current character is 'x' or 'y' -> turn in that direction

                    # pick normal vector
                    en = {'x' : ex, 'y' : ey}[cmd[i]]
                    # choose sign (whether to turn left of right)
                    en = choose_direction(en, currentPt, e, ptt, et)

                    # find turn angle
                    a = abs(angle_between(e, en))
                    n = K.dot(e)
                    if np.dot(n, en) < 0: n = -n

                    # move current point
                    currentPt += bend_radius * (
                        e * np.sin(a) + n * (1 - np.cos(a)))
                    # add turn to RouteDescription
                    path.add(Turn(bend_radius, a*np.dot(e, K.dot(n))))
                    # update direction vector
                    e = e * np.cos(a) + n * np.sin(a)

                    # if aux command is available, attach to the object
                    if auxcmd is not None: path.elements[-1].cmd = auxcmd
                    auxcmd = None

                else:
                    # any character which does not conform to the rules handled
                    # above will be stored as aux command and attached to the
                    # next path element

                    # the meander characters '>', '<' are handled this way
                    auxcmd = cmd[i]

                i += d

            # return the final reached point, the constructed RouteDescription,
            # the final direction vector and any unattached aux command
            return currentPt, path, e, auxcmd


        Rs = None

        if cmd.find('|') >= 0:   # if command contains the separator '|'
            tup = cmd.split('|')
            if len(tup) == 2:    # and it occurs only once,
                cmdI, cmdF = tup # split string into the two halves
            else:
                cmdI, Rs, cmdF = tup # if it occurs twice, get the two halves
                Rs = float(Rs) # and the bend radius sandwiched between them

            # when constructed, the path halves will be finished with a bend
            finalConnection = 'singleBend'

        elif cmd.find('-') >= 0: # if command contains the separator '-'
            cmdI, cmdF = cmd.split('-') # split string into the two halves

            # when constructed, the path halves will be finished with a bend
            finalConnection = 'straight'

        else:
            raise ValueError('routing command ' + cmd
                + ' does not contain splitting character')

        e1 = unit_vec(dir1)
        e2 = unit_vec(dir2)

        # parse the two halves of the string
        pt1, pathI, e1, auxcmdI = parse_command(
            cmdI, 1, pt1, e1, pt2, e2, radius)
        pt2,pathF,e2,auxcmdF = parse_command(
            cmdF, -1, pt2, e2, pt1, e1, radius)
        dir1 = vec_direction(e1)
        dir2 = vec_direction(e2)


        # HANDLING OF MEANDERS

        def smap(s, dict):
            # helper function making replacements in a string according to
            # a dictionary
            return ''.join([dict[ch] if ch in dict else ch for ch in s])
        def repl(element):
            # helper function for flipping elements constructed from the second
            # half of the routing command; the code above only took care that
            # the elements are constructed in the correct direction (from the
            # end of the string towards the separator) but we still need to:
            #   1) flip direction of turns
            #   2) flip direction of meanders
            if hasattr(element,'angle'):
                element.angle = -element.angle
            if hasattr(element,'cmd'):
                element.cmd = smap(element.cmd, {'<' : '>', '>' : '<'})
            return element

        # if there was an unattached meander command in the second half of the
        # routing command, flip it
        if auxcmdF: auxcmdF = smap(auxcmdF, {'<' : '>', '>' : '<'})

        if finalConnection == 'singleBend':
            # if final connection should be a bend
            if Rs is not None: radius = Rs
            pathM = cls.bend_between_points(pt1, dir1, pt2, dir2, radius)

            # unattached meander commands from the two halves of the routing
            # command will be attached to the straight segments of the
            # created connection
            if auxcmdI is not None: pathM.elements[0].cmd = auxcmdI
            if auxcmdF is not None: pathM.elements[2].cmd = auxcmdF

        elif finalConnection == 'straight':
            # if final connection shoule be straight
            pathM = cls.straight_between_points(pt1, dir1, pt2, dir2)
            # unattached meander commands from the two halves of the routing
            # command will be attached to this connection
            #
            # NOTE: This will FAIL if commands were specified on both sides of
            # the straight connection, e.g. if the routing command = '...<-<...'
            if auxcmdI is not None: pathM.elements[0].cmd = auxcmdI
            if auxcmdF is not None: pathM.elements[0].cmd = auxcmdF

        # concatenate RouteDescription objects constructed from the two
        # halves of the routing command and from the connection
        path = cls(pathI.elements + pathM.elements
            + [repl(element) for element in pathF.elements[::-1]])

        # get length of the constructed path (without meanders so far)
        L0 = path.length()

        # count how many elements have a meander command attached to them
        Nfree = sum(
            [
                1 if hasattr(e, 'cmd') and e.cmd in ['<','>','S', 'Z']
                else 0
                for e in path.elements
            ]
        )

        if length is not None: # If a length was specified for the path...
            if Nfree == 0:
                raise ValueError('routing method ' + cmd
                    + ' incompatible with length constraint')
            # ...and there is at least one element which should have meanders
            # calculate extra length for one set of meanders
            dL = (length - L0) / Nfree

            if dL < 0:
                raise ValueError('specified length too short')
            # Continue only if the requested length is larger than the
            # length without meanders

            mres = cls() # make new empty RouteDescription `mres`
            for element in path.elements:
                meander_cmds = ['<', '>', 'S', 'Z']
                if hasattr(element,'cmd') and element.cmd in meander_cmds:
                    if not isinstance(element, Segment):
                        raise ValueError(
                            "free length modifier can only be applied "
                            "to segments."
                        )
                    if meander_radius is not None:
                        Rm = meander_radius
                    else:
                        Rm = radius
                    # make meanders and add the meandered segment to `mres`
                    mroute = cls.meander(
                        element.length, dL, Rm, element.cmd)
                    mres.elements += mroute.elements
                else:
                    # elements which should not have meanders are just
                    # added to `mres` unchanged
                    mres.add(element)

            path = mres

        return path

    @classmethod
    def between_ports(
        cls,
        initial_port,
        final_port,
        intermediate_points=None,
        cmd='',
        length=None,
        cpw_routing_args={},
        abcmd=None,
    ) -> None:
        """
        Creates RouteDescription connecting two given ports.

        Parameters
        ----------
        initial_port : qdl.port.Port
            Starting port.
        final_port : qdl.port.Port
            Ending port.
        intermediate_points : list[libqudev.routing.RoutingPoint], optional
            specifies intermediate points through which the route should
            pass.
            Defaults to None which is equivalent to
            `intermediate_points = []`
            If N intermediate points are given, the argument cmd should
            be a list of N+1 routing commands (strings)
        cmd : str or list[str], optional
            command specifying routing of the CPW; for description of
            its syntax, see the examples in
            `libqudev/docs/examples/routing.py`.
            Defaults to '' (connection of collinear ports facing each
            other by a straight CPW).
            If N intermediate points were specified, cmd can be a list
            of N+1 routing commands.
        length : float or list[float], optional
            Desired length of the CPW or None; has an effect only if the
            routing command contains a meander character (e.g. '<' or
            '>').
            If N intermediate points were specified, the argument can be
            a list of N+1 lengths of the individual route segments
        cpw_routing_args : dict or list[dict]
            Other parameters describing the route.
            cpw_routing_args['bend_radius'] : float; defaults to 100
            cpw_routing_args['meander_radius'] : float; defaults to 50

        Returns
        -------
        RouteDescription
        """

        if intermediate_points is None: intermediate_points = []
        # if intermediate points are specified
        # create each section with another call
        # to makeCPWsection
        if intermediate_points:
            N = len(intermediate_points) + 1
            allpts = [initial_port] + intermediate_points + [final_port]
            res = cls()
            for i in range(N):
                p1 = allpts[i]
                p2 = allpts[i + 1]
                p2rev = RoutingPoint(
                    placement=Placement(
                        position=p2.placement.position,
                        rotation=(p2.placement.rotation
                                  + (180 if i < N - 1 else 0)),
                    )
                )
                section = cls.between_ports(p1, p2rev,
                    intermediate_points = None,
                    cmd = cmd[i] if isinstance(cmd, list) else cmd,
                    length = length[i] if isinstance(length, list) else length,
                    cpw_routing_args = cpw_routing_args[i]
                        if isinstance(cpw_routing_args, list)
                        else cpw_routing_args)
                res = res + section
            # Attach metadata to the overall path description
            res.metadata = {
                'cmd':cmd,
                'length':length,
                'cpw_routing_args': cpw_routing_args,
                'ports': (initial_port, final_port),
                'intermediate_points': intermediate_points}
            res.initial_placement = initial_port.placement
            return res

        # continue if no intermediate points were specified
        # (making single section)
        # default routing argument values
        cpwRA = {
            'bend_radius': 100,
            'meander_radius': 50}
        # Override with user-specified values
        cpwRA.update(cpw_routing_args)


        if length is not None:
            for p in [initial_port, final_port]:
                if hasattr(p, 'length_offset'):
                    length -= getattr(p, 'length_offset')

        # make route between the specified points from routing command...
        route = cls.between_points(
            initial_port.position,
            initial_port.direction,
            final_port.position,
            final_port.direction,
            cpwRA['bend_radius'],
            length = length,
            cmd = cmd,
            meander_radius = cpwRA['meander_radius'])
        routeLength = route.length()
        # ...and simplify it if possible
        route = route.simplify()

        route.initial_placement = initial_port.placement

        route.metadata = {                  # attach metadata
            'cmd': cmd,
            'length': length,
            'cpw_routing_args': cpw_routing_args,
            'ports': (initial_port, final_port),
            'intermediate_points': intermediate_points}

        return route
