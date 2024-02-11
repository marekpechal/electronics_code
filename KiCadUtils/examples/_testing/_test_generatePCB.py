import sys, os
filepath = os.path.dirname(os.path.realpath(__file__))
libpath = os.path.normpath(os.path.join(filepath, '..\\..'))
sys.path.insert(-1, libpath)

from KiCadStructure import KiCadStructure
from Footprints import SMP_19S102_40ML5_2layer_508um_AD1000
import numpy as np

template_filename = 'Template.kicad_pcb'
output_filename = '_test.kicad_pcb'

pcb_center = [100.0, 100.0]
pcb_radius = 25.0

board = KiCadStructure.fromPCBfile(template_filename)
gnd_net_idx = board.addNet('GND')

board.addEdgeCutCircle(pcb_center[0], pcb_center[1], pcb_radius)
for layer, net, net_name, clearance in [
        ('F.Cu', gnd_net_idx, 'GND', 0.05),
        ('B.Cu', gnd_net_idx, 'GND', 0.05),
        ('F.Mask', 0, '""', None),
        ('B.Mask', 0, '""', None)]:
    board.addAsChild(KiCadStructure.filledZoneCircle(
        net, net_name, layer, pcb_center[0], pcb_center[1], pcb_radius,
        clearance = clearance))

via_diameter = 0.5
via_drill_diameter = 0.3

# STITCHING VIAS
# dr = 1.0
# for r in np.arange(0, pcb_radius - 2*dr, dr):
#     if r == 0:
#         phi_rng = [0]
#     else:
#         N = int(round(2*np.pi*r / dr))
#         phi_rng = np.linspace(0, 2*np.pi, N+1)[:-1]
#     for phi in phi_rng:
#         pt = [pcb_center[0] + r*np.cos(phi), pcb_center[1] + r*np.sin(phi)]
#         board.addAsChild(KiCadStructure.via(pt, via_diameter, via_drill_diameter, ['F.Cu', 'B.Cu'], gnd_net_idx))


connector_footprint = SMP_19S102_40ML5_2layer_508um_AD1000()

r = 15.0
for i in range(8):
    phi = 2*np.pi*i/8
    pt = [pcb_center[0] + r*np.sin(phi), pcb_center[1] + r*np.cos(phi)]
    orientation = 180*phi/np.pi
    board.placeAt(connector_footprint, pt, orientation)



board.toPCBfile(output_filename)
