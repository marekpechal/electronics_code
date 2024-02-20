import sys, os
filepath = os.path.dirname(os.path.realpath(__file__))
libpath = os.path.normpath(os.path.join(filepath, '..\\..'))
sys.path.insert(-1, libpath)

# TODO: fix the discrepancy in routing bend directions due to the KiCad using
# the "positive x = right & positive y = down" convention
# TODO: check why demo from testboard.json gives DRC violations (likely
# incorrectly assigned netclass)

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

import shapely
import shapely.geometry, shapely.ops

from KiCadStructure import KiCadStructure, _rotation_matrix
from lqd_routing import RouteDescription
import Footprints
import numpy as np
import json
import itertools

def plotShapelyPolygon(ax, poly, **kwargs):
    """
    Helper function to plot a shapely.geometry.Polygon object
    """
    # from https://stackoverflow.com/a/70533052
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

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

for fname in os.listdir(filepath):
    if not fname.endswith('.json'):
        continue

    template_filename = os.path.join(filepath, 'Template')
    output_filename = os.path.join(filepath, 'output', fname.split('.')[0])
    config_file = os.path.join(filepath, fname)

    # prepare template
    os.system(f'copy {template_filename}.kicad_pro {output_filename}.kicad_pro')
    board = KiCadStructure.fromPCBfile(template_filename + '.kicad_pcb')
    gnd_net_idx = board.addNet('GND')
    pcb_center = [100.0, 100.0]
    with open(template_filename + '.kicad_pro', 'r') as f:
        design_rules = json.load(f)['board']['design_settings']['rules']

    with open(config_file, 'r') as f:
        config = json.load(f)

    # BOARD OUTLINE
    if config['board_shape'] == 'CIRCULAR':
        pcb_radius = config['board_diameter']/2
        board.addEdgeCutCircle(pcb_center[0], pcb_center[1], pcb_radius)
        for layer, net, net_name, clearance in [
                ('F.Cu', gnd_net_idx, 'GND', 0.05),
                ('B.Cu', gnd_net_idx, 'GND', 0.05),
                ('F.Mask', 0, '""', None),
                ('B.Mask', 0, '""', None)]:
            board.addAsChild(KiCadStructure.filledZoneCircle(
                net, net_name, layer, pcb_center[0], pcb_center[1], pcb_radius,
                clearance = clearance))
    elif config['board_shape'] == 'RECTANGULAR':
        pcb_width = config['board_width']
        pcb_height = config['board_height']
        board.addEdgeCutRectangle(
            pcb_center[0]-pcb_width/2, pcb_center[1]-pcb_height/2,
            pcb_center[0]+pcb_width/2, pcb_center[1]+pcb_height/2)
        for layer, net, net_name, clearance in [
                ('F.Cu', gnd_net_idx, 'GND', 0.05),
                ('B.Cu', gnd_net_idx, 'GND', 0.05),
                ('F.Mask', 0, '""', None),
                ('B.Mask', 0, '""', None)]:
            board.addAsChild(KiCadStructure.filledZoneRectangle(
                net, net_name, layer,
                pcb_center[0]-pcb_width/2, pcb_center[1]-pcb_height/2,
                pcb_center[0]+pcb_width/2, pcb_center[1]+pcb_height/2,
                clearance = clearance))
    else:
        raise NotImplementedError(f'Board shape {config["board_shape"]}')

    # CUTOUT
    if not 'cutout' in config:
        pass
    elif config['cutout']['shape'] == 'RECTANGULAR':
        cutout_w, cutout_h = config['cutout']['dimensions']
        relief_r = config['cutout']['internal_relief_diameter']
        d = relief_r*np.sqrt(2)
        board.addEdgeCutLine(
            pcb_center[0]-cutout_w/2+d, pcb_center[1]-cutout_h/2,
            pcb_center[0]+cutout_w/2-d, pcb_center[1]-cutout_h/2
            )
        board.addEdgeCutArc(
            [pcb_center[0]+cutout_w/2-d, pcb_center[1]-cutout_h/2],
            [pcb_center[0]+cutout_w/2, pcb_center[1]-cutout_h/2],
            [pcb_center[0]+cutout_w/2, pcb_center[1]-cutout_h/2+d]
            )
        board.addEdgeCutLine(
            pcb_center[0]+cutout_w/2, pcb_center[1]-cutout_h/2+d,
            pcb_center[0]+cutout_w/2, pcb_center[1]+cutout_h/2-d
            )
        board.addEdgeCutArc(
            [pcb_center[0]+cutout_w/2, pcb_center[1]+cutout_h/2-d],
            [pcb_center[0]+cutout_w/2, pcb_center[1]+cutout_h/2],
            [pcb_center[0]+cutout_w/2-d, pcb_center[1]+cutout_h/2]
            )
        board.addEdgeCutLine(
            pcb_center[0]+cutout_w/2-d, pcb_center[1]+cutout_h/2,
            pcb_center[0]-cutout_w/2+d, pcb_center[1]+cutout_h/2
            )
        board.addEdgeCutArc(
            [pcb_center[0]-cutout_w/2+d, pcb_center[1]+cutout_h/2],
            [pcb_center[0]-cutout_w/2, pcb_center[1]+cutout_h/2],
            [pcb_center[0]-cutout_w/2, pcb_center[1]+cutout_h/2-d]
            )
        board.addEdgeCutLine(
            pcb_center[0]-cutout_w/2, pcb_center[1]+cutout_h/2-d,
            pcb_center[0]-cutout_w/2, pcb_center[1]-cutout_h/2+d
            )
        board.addEdgeCutArc(
            [pcb_center[0]-cutout_w/2, pcb_center[1]-cutout_h/2+d],
            [pcb_center[0]-cutout_w/2, pcb_center[1]-cutout_h/2],
            [pcb_center[0]-cutout_w/2+d, pcb_center[1]-cutout_h/2]
            )
        cutout_polygon = np.concatenate((
            [[-cutout_w/2+d/2+relief_r*np.cos(u), -cutout_h/2+d/2+relief_r*np.sin(u)]
                for u in np.linspace(3*np.pi/4, 7*np.pi/4, 21)],
            [[+cutout_w/2-d/2+relief_r*np.cos(u), -cutout_h/2+d/2+relief_r*np.sin(u)]
                for u in np.linspace(5*np.pi/4, 9*np.pi/4, 21)],
            [[+cutout_w/2-d/2+relief_r*np.cos(u), +cutout_h/2-d/2+relief_r*np.sin(u)]
                for u in np.linspace(7*np.pi/4, 11*np.pi/4, 21)],
            [[-cutout_w/2+d/2+relief_r*np.cos(u), +cutout_h/2-d/2+relief_r*np.sin(u)]
                for u in np.linspace(9*np.pi/4, 13*np.pi/4, 21)]
            )) + np.array(pcb_center)
    else:
        raise NotImplementedError(f'Cutout shape {config["cutout"]["shape"]}')

    # CONNECTORS
    connector_pads = []
    connectors_count = 0
    for connector in config['connectors']:
        footprint = getattr(Footprints, connector['footprint'])()
        if connector['pattern_type'] == 'CIRCULAR':
            r = connector['pattern_diameter']/2
            for i in range(connector['pattern_count']):
                phi = 2*np.pi*(i+connector['pattern_relative_rotation'])/connector['pattern_count']
                pt = [pcb_center[0] + r*np.cos(phi), pcb_center[1] - r*np.sin(phi)]
                orientation = 90 + 180*phi/np.pi
                footprint_instance = board.placeAt(footprint, pt, orientation)
                for pad in footprint_instance.getChildren('pad'):
                    if pad.content[0] == '1':
                        connector_pads.append(pad)
                        for net_entry in pad.getChildren('net'):
                            net = connectors_count + 2
                            net_name = f'NET_{net-1}'
                            board.addNet(net_name)
                            net_entry.content = [str(net), net_name]
                connectors_count += 1
        else:
            NotImplementedError(f'Pattern type {connector["pattern_type"]}')

    # HOLES
    for hole in config['holes']:
        if hole['pattern_type'] == 'CIRCULAR':
            r = hole['pattern_diameter']/2
            for i in range(hole['pattern_count']):
                phi = 2*np.pi*(i+hole['pattern_relative_rotation'])/hole['pattern_count']
                pt = [pcb_center[0] + r*np.cos(phi), pcb_center[1] + r*np.sin(phi)]
                board.addHole(pt, hole['drill_diameter'])
        else:
            NotImplementedError(f'Pattern type {connector["pattern_type"]}')

    # LAUNCHERS
    launcher_pads = []
    margin = config['cutout']['launcher_length'] / 2 + 1.1 * design_rules['min_copper_edge_clearance']
    x0 = config['cutout']['dimensions'][0] / 2 + margin
    y0 = config['cutout']['dimensions'][1] / 2 + margin
    launchers = KiCadStructure(name = 'module', content = ['Launchers'])
    launcher_idx = 1
    for side, xP, sgn, flip, angle in [
            ('left',  -x0, 1, True, 180),
            ('bottom', y0, 1, False, 270),
            ('right', x0, -1, True, 0),
            ('top', -y0, -1, False, 90)]:

        for i in range(config['cutout']['launcher_numbers'][side]):
            ds = config['cutout']['launcher_spacings'][side]
            xL = sgn*(i - (config['cutout']['launcher_numbers'][side]-1)/2)*ds
            x, y = xL, xP
            if flip: x, y = y, x
            if str(launcher_idx) in config['launcher_to_connector_mapping']:
                net = int(config['launcher_to_connector_mapping'][str(launcher_idx)][0]) + 1
                net_name = f'NET_{net-1}'
            else:
                net = 0
                net_name = '""'

            pad = KiCadStructure.pad_rect(
                [pcb_center[0] + x, pcb_center[1] + y, angle],
                launcher_idx,
                config['cutout']['launcher_length'],
                config['cutout']['launcher_width'],
                ['F.Cu'], net, net_name)
            launcher_pads.append(pad)
            launchers.addAsChild(pad)
            launcher_idx += 1

    board.addAsChild(launchers)

    # =====================

    # placing traces joining connectors with launchers
    for conn_pad_idx, pad in enumerate(connector_pads):
        key = str(conn_pad_idx + 1)
        if key in config['launcher_to_connector_mapping']:
            lncr_pad_idx = int(config['launcher_to_connector_mapping'][key][0]) - 1

            conn_pad = connector_pads[conn_pad_idx]
            lncr_pad = launcher_pads[lncr_pad_idx]

            conn_pad_T = conn_pad.getTransformation()
            lncr_pad_T = lncr_pad.getTransformation()

            pt1 = conn_pad_T.translation_vector
            dir1 = conn_pad_T.rotation_angle
            pt2 = lncr_pad_T.translation_vector
            dir2 = lncr_pad_T.rotation_angle

            bend_radius = 2.0
            route = RouteDescription.between_points(pt1, -dir1, pt2, -dir2,
                bend_radius, cmd = config['launcher_to_connector_mapping'][key][1])

            route.initial_position = pt1
            route.initial_direction_vector = np.array([np.cos(dir1), -np.sin(dir1)])

            poly = np.array(
                [route.point(s)[0] for s in np.linspace(0, route.length(), 101)[1:-1]])
            board.addPolySegment(poly, 0.17, 'F.Cu', int(key) + 1)
            plt.plot(poly[:, 0], poly[:, 1], color = 'black')

    for pad in connector_pads + launcher_pads:
        pad_T = pad.getTransformation()

        pt = pad_T.translation_vector
        dir = pad_T.rotation_angle
        R = _rotation_matrix(dir)
        pts = np.array([[2.0, -1.0, -1.0], [0., 0.5, -0.5]])
        pts = R.dot(pts).transpose() + pt
        plt.fill(pts[:, 0], pts[:, 1], color = 'red')

    plt.show()


    ######################

    # creating polygon representations of objects on the PCB
    # for plotting and for via distribution
    polys = {
        'cutout': [shapely.geometry.Polygon(cutout_polygon)],
        'segment': [],
        'pad': [],
        'footprint': []
        }

    # convert segments and pads
    for obj in board.filter(
            name_filter = (lambda s: s in ['segment', 'pad']),
            layer_filter = (lambda c: 'F.Cu' in c or 'B.Cu' in c or '*.Cu' in c),
            recursive = True,
            check = lambda obj: obj.name != 'footprint'):
        polys[obj.name].append(shapely.geometry.Polygon(obj.toPolygon()))

    # convert footprints (represented by the convex hull of all contained objects)
    for footprint in board.getChildren('footprint'):
        points = []
        for obj in footprint.filter(
            name_filter = (lambda s: s in ['zone', 'segment', 'pad']),
            layer_filter = (lambda c: 'F.Cu' in c or 'B.Cu' in c or '*.Cu' in c),
            recursive = True
            ):
            points += list(obj.toPolygon())

        polys['footprint'].append(shapely.geometry.MultiPoint(points).convex_hull)

    # parameters for via distribution
    via_diameter = 0.5
    via_drill_diameter = 0.3
    clearances = {
        'cutout': 0.1,
        'segment': 0.25,
        'pad': 0.25,
        'footprint': 0.1
        }

    buffer_distance = via_diameter / 2
    poly_combined = shapely.ops.unary_union([poly
        for name, lst in polys.items() for poly in lst])
    poly_buff_combined = shapely.ops.unary_union(
        [poly.buffer(clearances[name] + via_diameter/2)
        for name, lst in polys.items() for poly in lst])


    fig, ax = plt.subplots()

    # plot and distribute vias
    dr = 1.5 * via_diameter
    if config['board_shape'] == 'CIRCULAR':
        R = config['board_diameter']/2
    elif config['board_shape'] == 'RECTANGULAR':
        R = np.sqrt(config['board_width']**2 + config['board_height']**2)/2
    for r in np.arange(0, R - 0.5*dr, dr):
        if r == 0:
            phi_rng = [0]
        else:
            N = int(round(2*np.pi*r / dr))
            phi_rng = np.linspace(0, 2*np.pi, N+1)[:-1]
        for phi in phi_rng:
            pt = [pcb_center[0]+r*np.cos(phi), pcb_center[1]+r*np.sin(phi)]
            if not poly_buff_combined.contains(shapely.geometry.Point(*pt)):
                plt.plot(*pt, '.', color = 'green')
                board.addVia(*pt, via_diameter, via_drill_diameter, 1)

    plotShapelyPolyLike(ax, poly_buff_combined, facecolor=(0., 0., 1., 0.2))
    plotShapelyPolyLike(ax, poly_combined, facecolor=(0., 0., 0., 0.2))
    plt.show()

    # save KiCad PCB file
    board.toPCBfile(output_filename + '.kicad_pcb')
