from KiCadStructure import KiCadStructure
import numpy as np

class SMP_19S102_40ML5_2layer_508um_AD1000(KiCadStructure):
    def __init__(self, **kwargs):

        KiCadStructure.__init__(self, name = 'footprint', content = [
            'SMP'
            ])

        self.addChild(cls = KiCadStructure, name = 'layer', content = ['F.Cu'])

        via_diameter = 0.5
        via_drill_diameter = 0.3
        gnd_net_idx = 1


        # adding vias
        footprint_via_positions = [
            [+2.0, -2.0], [+1.0, -2.0], [0.0, -2.0], [-1.0, -2.0], [-2.0, -2.0],
            [+2.0, +2.0], [+1.0, +2.0], [0.0, +2.0], [-1.0, +2.0], [-2.0, +2.0],
            [-2.0, -1.0], [-2.0, 0.0], [-2.0, 1.0],
            [+4.0, -0.45], [+3.0, -1.05], [+2.0, -1.05], [+1.0, -0.85], [0.0, -0.85],
            [+4.0, +0.45], [+3.0, +1.05], [+2.0, +1.05], [+1.0, +0.85], [0.0, +0.85],
            [-0.601, -0.601], [-0.85, 0.0], [-0.601, +0.601]
            ]
        for pt in footprint_via_positions:
            self.addAsChild(KiCadStructure.pad_th(
                pt, 0, via_diameter, via_drill_diameter, ['F.Cu', 'B.Cu'], gnd_net_idx, 'GND'))

        # pad
        self.addAsChild(KiCadStructure.pad_rect(
            [2.45, 0, 0], 1, 1.1, 0.44, ['F.Cu'], 0, '""'))

        # central keepout
        r = 0.6
        bridge_width = 0.9
        x1 = 1.45
        x2 = 3.3
        cutout_width = 1.4
        theta = np.arcsin(0.5*bridge_width/r)
        pts = [[r*np.cos(u), r*np.sin(u)]
            for u in np.linspace(theta, 2*np.pi-theta, 40)]
        pts = (
            [[x2, cutout_width/2], [x1, cutout_width/2], [x1, bridge_width/2]] +
            pts +
            [[x1, -bridge_width/2], [x1, -cutout_width/2], [x2, -cutout_width/2]]
            )
        self.addAsChild(KiCadStructure.ruleArea(
            'F.Cu', ['copperpour', 'footprints'], pts))

        # outline keepouts
        solder_area_size = 4.9
        keepout_width = 0.4
        bridge_width = 1.0
        pts = [
            [solder_area_size/2, solder_area_size/2],
            [bridge_width/2, solder_area_size/2],
            [bridge_width/2, solder_area_size/2+keepout_width],
            [solder_area_size/2+keepout_width, solder_area_size/2+keepout_width],
            [solder_area_size/2+keepout_width, bridge_width/2],
            [solder_area_size/2, bridge_width/2],
            ]
        self.addAsChild(KiCadStructure.ruleArea(
            'F.Cu', ['copperpour', 'footprints'], pts))
        self.addAsChild(KiCadStructure.ruleArea(
            'F.Cu', ['copperpour', 'footprints'], [[x, -y] for x, y in pts]))
        pts = [[-x, max(y, bridge_width+cutout_width/2)] for x, y in pts]
        self.addAsChild(KiCadStructure.ruleArea(
            'F.Cu', ['copperpour', 'footprints'], pts))
        self.addAsChild(KiCadStructure.ruleArea(
            'F.Cu', ['copperpour', 'footprints'], [[x, -y] for x, y in pts]))

        # END FOOTPRINT DEFINITION
