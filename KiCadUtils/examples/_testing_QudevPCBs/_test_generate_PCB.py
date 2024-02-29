import sys, os
filepath = os.path.dirname(os.path.realpath(__file__))
libpath = os.path.normpath(os.path.join(filepath, '..\\..'))
sys.path.insert(-1, libpath)

# TODO: check why demo from testboard.json gives DRC violations (likely
# incorrectly assigned netclass)
# TODO: instead of hard-coding trace width, read it from template project file
# TODO: instead of hard-coding via dimensions, read them from .json
# TODO: instead of hard-coding limits for via fence distribution, use the footprint polygons for exclusion

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

import shapely
import shapely.geometry, shapely.ops

from helpers import plotShapelyPolygon, plotShapelyPolyLike, offsetPath, distributePointsOnPath
from KiCadStructure import KiCadStructure, _rotation_matrix
from lqd_routing import RouteDescription
import Footprints
import numpy as np
import json
import itertools

def getParam(dct, key, namespace):
    return eval(str(dct[key]), namespace)

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
    pcb_center = np.array([100.0, 100.0])
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
    elif isinstance(config['board_shape'], list): # custom shape
        point_list = []
        for command, *params in config['board_shape']:
            if command == 'start':
                last_point = [params[0]+pcb_center[0], params[1]+pcb_center[1]]
            elif command == 'line_to':
                new_point = [params[0]+pcb_center[0], params[1]+pcb_center[1]]
                board.addEdgeCutLine(*last_point, *new_point)
                last_point = new_point
                point_list.append(new_point)
            elif command == 'arc_to':
                new_point = [params[0]+pcb_center[0], params[1]+pcb_center[1]]
                radius = params[2]
                ex = np.array(new_point) - np.array(last_point)
                d = np.linalg.norm(ex)
                h = np.sqrt(radius**2 - (0.5*d)**2) * radius / abs(radius)
                ex = ex / d
                ey = np.array([[0, -1], [1, 0]]).dot(ex)
                ctr = np.array(last_point) + 0.5 * d * ex - h * ey
                theta = np.arcsin(0.5 * d / radius)
                pts = []
                for phi in np.linspace(-theta, theta, 41)[0:]:
                    pts.append(list(
                        ctr + (ey*np.cos(phi)+ex*np.sin(phi))*radius))
                board.addEdgeCutPolyLine(pts)
                last_point = new_point
                point_list += pts
        for layer, net, net_name, clearance in [
                ('F.Cu', gnd_net_idx, 'GND', 0.05),
                ('B.Cu', gnd_net_idx, 'GND', 0.05),
                ('F.Mask', 0, '""', None),
                ('B.Mask', 0, '""', None)]:
            board.addAsChild(KiCadStructure.filledZone(
                net, net_name, layer, point_list, clearance = clearance))

    else:
        raise NotImplementedError(f'Board shape {config["board_shape"]}')

    # CUTOUT
    if not 'cutout' in config:
        cutout_polygon = None
    elif config['cutout']['shape'] == 'RECTANGULAR':
        cutout_w, cutout_h = config['cutout']['dimensions']
        relief_r = config['cutout']['internal_relief_diameter']
        if 'center' in config['cutout']:
            x0, y0 = config['cutout']['center']
        else:
            x0, y0 = 0, 0
        d = relief_r*np.sqrt(2)
        board.addEdgeCutLine(
            x0+pcb_center[0]-cutout_w/2+d, y0+pcb_center[1]-cutout_h/2,
            x0+pcb_center[0]+cutout_w/2-d, y0+pcb_center[1]-cutout_h/2
            )
        board.addEdgeCutArc(
            [x0+pcb_center[0]+cutout_w/2-d, y0+pcb_center[1]-cutout_h/2],
            [x0+pcb_center[0]+cutout_w/2, y0+pcb_center[1]-cutout_h/2],
            [x0+pcb_center[0]+cutout_w/2, y0+pcb_center[1]-cutout_h/2+d]
            )
        board.addEdgeCutLine(
            x0+pcb_center[0]+cutout_w/2, y0+pcb_center[1]-cutout_h/2+d,
            x0+pcb_center[0]+cutout_w/2, y0+pcb_center[1]+cutout_h/2-d
            )
        board.addEdgeCutArc(
            [x0+pcb_center[0]+cutout_w/2, y0+pcb_center[1]+cutout_h/2-d],
            [x0+pcb_center[0]+cutout_w/2, y0+pcb_center[1]+cutout_h/2],
            [x0+pcb_center[0]+cutout_w/2-d, y0+pcb_center[1]+cutout_h/2]
            )
        board.addEdgeCutLine(
            x0+pcb_center[0]+cutout_w/2-d, y0+pcb_center[1]+cutout_h/2,
            x0+pcb_center[0]-cutout_w/2+d, y0+pcb_center[1]+cutout_h/2
            )
        board.addEdgeCutArc(
            [x0+pcb_center[0]-cutout_w/2+d, y0+pcb_center[1]+cutout_h/2],
            [x0+pcb_center[0]-cutout_w/2, y0+pcb_center[1]+cutout_h/2],
            [x0+pcb_center[0]-cutout_w/2, y0+pcb_center[1]+cutout_h/2-d]
            )
        board.addEdgeCutLine(
            x0+pcb_center[0]-cutout_w/2, y0+pcb_center[1]+cutout_h/2-d,
            x0+pcb_center[0]-cutout_w/2, y0+pcb_center[1]-cutout_h/2+d
            )
        board.addEdgeCutArc(
            [x0+pcb_center[0]-cutout_w/2, y0+pcb_center[1]-cutout_h/2+d],
            [x0+pcb_center[0]-cutout_w/2, y0+pcb_center[1]-cutout_h/2],
            [x0+pcb_center[0]-cutout_w/2+d, y0+pcb_center[1]-cutout_h/2]
            )
        cutout_polygon = np.concatenate((
            [[x0-cutout_w/2+d/2+relief_r*np.cos(u), y0-cutout_h/2+d/2+relief_r*np.sin(u)]
                for u in np.linspace(3*np.pi/4, 7*np.pi/4, 21)],
            [[x0+cutout_w/2-d/2+relief_r*np.cos(u), y0-cutout_h/2+d/2+relief_r*np.sin(u)]
                for u in np.linspace(5*np.pi/4, 9*np.pi/4, 21)],
            [[x0+cutout_w/2-d/2+relief_r*np.cos(u), y0+cutout_h/2-d/2+relief_r*np.sin(u)]
                for u in np.linspace(7*np.pi/4, 11*np.pi/4, 21)],
            [[x0-cutout_w/2+d/2+relief_r*np.cos(u), y0+cutout_h/2-d/2+relief_r*np.sin(u)]
                for u in np.linspace(9*np.pi/4, 13*np.pi/4, 21)]
            )) + np.array(pcb_center)
    else:
        raise NotImplementedError(f'Cutout shape {config["cutout"]["shape"]}')

    # CONNECTORS
    connector_pads = []
    connectors_count = 0
    for connector in config['connectors']:
        footprint = getattr(Footprints, connector['footprint'])()
        lst = []
        if connector['pattern_type'] == 'CIRCULAR':
            r = connector['pattern_diameter']/2
            for i in range(connector['pattern_count']):
                phi = 2*np.pi*(i+connector['pattern_relative_rotation'])/connector['pattern_count']
                pt = [pcb_center[0] + r*np.cos(phi), pcb_center[1] - r*np.sin(phi)]
                orientation = 180*phi/np.pi + 180
                lst.append((pt, orientation))
        elif connector['pattern_type'] == 'LINEAR':
            pt1 = np.array(connector['pattern_start_point'])
            pt2 = np.array(connector['pattern_end_point'])
            for i in range(connector['pattern_count']):
                pt = pt1 + i*(pt2-pt1)/(connector['pattern_count']-1)
                lst.append((pt+pcb_center, connector['footprint_orientation']))
        else:
            NotImplementedError(f'Pattern type {connector["pattern_type"]}')

        for pt, orientation in lst:
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


    # HOLES
    for hole in config['holes']:
        if 'pattern_type' in hole: # pattern of holes
            if hole['pattern_type'] == 'CIRCULAR':
                r = hole['pattern_diameter']/2
                for i in range(hole['pattern_count']):
                    phi = 2*np.pi*(i+hole['pattern_relative_rotation'])/hole['pattern_count']
                    pt = [pcb_center[0] + r*np.cos(phi), pcb_center[1] + r*np.sin(phi)]
                    board.addHole(pt, hole['drill_diameter'])
            elif hole['pattern_type'] == 'LINEAR':
                pt1 = np.array(hole['pattern_start_point'])
                pt2 = np.array(hole['pattern_end_point'])
                for i in range(hole['pattern_count']):
                    pt = pt1 + i*(pt2-pt1)/(hole['pattern_count']-1)
                    board.addHole(pt+pcb_center, hole['drill_diameter'])
            else:
                NotImplementedError(f'Pattern type {hole["pattern_type"]}')
        else: # single hole
            pt = np.array(hole['position'])
            board.addHole(pt+pcb_center, hole['drill_diameter'])

    # LAUNCHERS
    launcher_pads = []
    launcher_idx = 1

    # LAUNCHERS AROUND CUTOUT
    if 'cutout' in config:
        margin = config['cutout']['launcher_length'] / 2 + 1.1 * design_rules['min_copper_edge_clearance']
        X = config['cutout']['dimensions'][0] / 2 + margin
        Y = config['cutout']['dimensions'][1] / 2 + margin
        if 'center' in config['cutout']:
            x0, y0 = config['cutout']['center']
        else:
            x0, y0 = 0, 0
        launchers = KiCadStructure(name = 'module', content = ['Launchers'])

        for side, xP, sgn, flip, angle in [
                ('left',  -X, 1, True, 180),
                ('bottom', Y, 1, False, 270),
                ('right', X, -1, True, 0),
                ('top', -Y, -1, False, 90)]:

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
                    [pcb_center[0] + x0 + x, pcb_center[1] + y0 + y, angle],
                    launcher_idx,
                    config['cutout']['launcher_length'],
                    config['cutout']['launcher_width'],
                    ['F.Cu'], net, net_name)
                launcher_pads.append(pad)
                launchers.addAsChild(pad)
                launcher_idx += 1

        board.addAsChild(launchers)

    # ADDITIONAL LAUNCHERS
    if 'launchers' in config:
        for launcher in config['launchers']:
            margin = launcher['launcher_length'] / 2 + 1.1 * design_rules['min_copper_edge_clearance']
            eval_namespace = {}
            if config['board_shape'] == 'RECTANGULAR':
                eval_namespace = {
                    'left_edge': -config['board_width']/2 + margin,
                    'right_edge': config['board_width']/2 - margin,
                    'top_edge': -config['board_height']/2 + margin,
                    'bottom_edge': config['board_height']/2 - margin,
                    }
            launchers = KiCadStructure(name = 'module', content = ['Launchers'])
            if launcher['pattern_type'] == 'LINEAR':
                pt1 = np.array(getParam(launcher, 'pattern_start_point', eval_namespace))
                pt2 = np.array(getParam(launcher, 'pattern_end_point', eval_namespace))
                angle = launcher['launcher_orientation']
                for i in range(launcher['pattern_count']):
                    if str(launcher_idx) in config['launcher_to_connector_mapping']:
                        net = int(config['launcher_to_connector_mapping'][str(launcher_idx)][0]) + 1
                        net_name = f'NET_{net-1}'
                    else:
                        net = 0
                        net_name = '""'
                    pt = pt1 + i*(pt2-pt1)/(launcher['pattern_count']-1)
                    pad = KiCadStructure.pad_rect(
                        [pcb_center[0] + pt[0], pcb_center[1] + pt[1], angle],
                        launcher_idx,
                        launcher['launcher_length'],
                        launcher['launcher_width'],
                        ['F.Cu'], net, net_name)
                    launcher_pads.append(pad)
                    launchers.addAsChild(pad)
                    launcher_idx += 1
            else:
                NotImplementedError(f'Pattern type {launcher["pattern_type"]}')

            board.addAsChild(launchers)


    # =====================

    via_diameter = 0.5
    via_drill_diameter = 0.3


    # placing traces joining connectors with launchers
    for conn_pad_idx, pad in enumerate(connector_pads):
        key = str(conn_pad_idx + 1)
        if key in config['launcher_to_connector_mapping']:
            lncr_pad_idx = int(config['launcher_to_connector_mapping'][key][0]) - 1

            conn_pad = connector_pads[conn_pad_idx]
            lncr_pad = launcher_pads[lncr_pad_idx]

            conn_pad_T = conn_pad.getTransformation()
            lncr_pad_T = lncr_pad.getTransformation()

            pt1 = lncr_pad_T.translation_vector
            dir1 = lncr_pad_T.rotation_angle
            pt2 = conn_pad_T.translation_vector
            dir2 = conn_pad_T.rotation_angle

            bend_radius = config['trace_bend_radius']
            cmd = config['launcher_to_connector_mapping'][key][1]
            cmd = cmd.translate({114: 108, 108: 114}) # swap 'r' <-> 'l'
            route = RouteDescription.between_points(pt1, -dir1, pt2, -dir2,
                bend_radius, cmd = cmd)

            route.initial_position = pt1
            route.initial_direction_vector = np.array([np.cos(dir1), -np.sin(dir1)])

            poly = np.array(
                [route.point(s)[0] for s in np.linspace(1e-6, route.length()-1e-6, 101)])
            board.addPolySegment(poly, 0.17, 'F.Cu', int(key) + 1)
            plt.plot(poly[:, 0], -poly[:, 1], color = 'black')

            pts = np.concatenate((
                distributePointsOnPath(offsetPath(poly, +0.50), 0.65,
                    init_gap = 0.50, fin_gap = 2.50),
                distributePointsOnPath(offsetPath(poly, -0.50), 0.65,
                    init_gap = 0.50, fin_gap = 2.50)
                ))
            for pt in pts:
                board.addVia(*pt, via_diameter, via_drill_diameter, 1)

            plt.plot(pts[:, 0], -pts[:, 1], 'k.')

    for pad in connector_pads + launcher_pads:
        pad_T = pad.getTransformation()

        pt = pad_T.translation_vector
        dir = pad_T.rotation_angle
        R = _rotation_matrix(dir)
        pts = np.array([[2.0, -1.0, -1.0], [0., 0.5, -0.5]])
        pts = R.dot(pts).transpose() + pt
        plt.fill(pts[:, 0], -pts[:, 1], color = 'red')

    plt.show()


    ######################

    # creating polygon representations of objects on the PCB
    # for plotting and for via distribution
    polys = {
        'cutout': [shapely.geometry.Polygon(cutout_polygon)]
            if cutout_polygon is not None else [],
        'segment': [],
        'pad': [],
        'via': [],
        'footprint': []
        }

    # convert segments and pads
    for obj in board.filter(
            name_filter = (lambda s: s in ['segment', 'pad', 'via']),
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
    clearances = {
        'cutout': 0.1,
        'segment': 0.25,
        'pad': 0.25,
        'footprint': 0.1,
        'via': 0.1
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
        for r in np.arange(0, R - 0.5*dr, dr):
            if r == 0:
                phi_rng = [0]
            else:
                N = int(round(2*np.pi*r / dr))
                phi_rng = np.linspace(0, 2*np.pi, N+1)[:-1]
            for phi in phi_rng:
                pt = [pcb_center[0]+r*np.cos(phi), pcb_center[1]+r*np.sin(phi)]
                if not poly_buff_combined.contains(shapely.geometry.Point(*pt)):
                    plt.plot(pt[0], -pt[1], '.', color = 'green')
                    board.addVia(*pt, via_diameter, via_drill_diameter, 1)

    elif config['board_shape'] == 'RECTANGULAR':
        dy = dr * np.sqrt(3) / 2
        Ny = int(config['board_height'] / dy) - 1
        Nx = int(config['board_width'] / dr) - 1
        for i in range(Ny):
            y = dy * (i - (Ny-1)/2)
            for j in range(Nx):
                x = dr * (j - (Nx-1)/2) + 0.25 * dr * (-1)**(i % 2)
                pt = [pcb_center[0]+x, pcb_center[1]+y]
                if not poly_buff_combined.contains(shapely.geometry.Point(*pt)):
                    plt.plot(pt[0], -pt[1], '.', color = 'green')
                    board.addVia(*pt, via_diameter, via_drill_diameter, 1)

    plotShapelyPolyLike(ax, poly_buff_combined, facecolor=(0., 0., 1., 0.2))
    plotShapelyPolyLike(ax, poly_combined, facecolor=(0., 0., 0., 0.2))
    plt.show()

    # save KiCad PCB file
    board.toPCBfile(output_filename + '.kicad_pcb')
