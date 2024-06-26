import numpy as np
from Structure import Structure
from helpers import rotation_matrix

# TODO: allow passing nets by name as well as index
# TODO: fix zone net assignment
# TODO: pass centers of circles, arcs etc as lists/tuples, not individual coordinates
# TODO: improve estimate of text bounding box

class Transformation:
    def __init__(self, translation_vector, rotation_angle):
        self.translation_vector = translation_vector
        self.rotation_angle = rotation_angle

    def __call__(self, x):
        return (rotation_matrix(self.rotation_angle).dot(x) +
            self.translation_vector)

    def __mul__(self, other):
        # d1 + R1.(d2 + R2.x)
        return Transformation(
            self(other.translation_vector),
            self.rotation_angle + other.rotation_angle
            )

    @classmethod
    def I(cls):
        return cls(np.zeros(2), 0)

def _segmentToPoly(pt1, pt2, w):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    e = pt2 - pt1
    if np.linalg.norm(e) == 0:
        return None
    e = e / np.linalg.norm(e)
    n = np.array([1., -1.]) * e[::-1]
    return np.array(
        [pt2 + 0.5 * w * (e * np.cos(u) + n * np.sin(u))
            for u in np.linspace(-np.pi/2, np.pi/2, 21)] +
        [pt1 - 0.5 * w * (e * np.cos(u) + n * np.sin(u))
            for u in np.linspace(-np.pi/2, np.pi/2, 21)]
        )


class KiCadStructure(Structure):
    def filter(self,
            name_filter = None,
            layer_filter = None,
            net_filter = None,
            content_filter = None,
            recursive = False,
            **kwargs):
        def apply(filter, obj):
            if filter is not None:
                return filter(obj)
            else:
                return True
        if recursive:
            objects = self.walk(**kwargs)
        else:
            objects = self.children
        for entry in objects:
            if apply(name_filter, entry.name):
                if apply(content_filter, entry.content):
                    if apply(layer_filter, entry.getLayers()):
                        if apply(net_filter, entry.getNet()):
                            yield entry

    def getLayers(self):
        return self.getChildrenContent('layer', 'layers')

    def getNet(self):
        return self.getChildrenContent('net')

    def getSegment(self):
        T = self.getTransformation()
        pt1 = T([float(s) for s in self['start'].content])
        pt2 = T([float(s) for s in self['end'].content])
        width = float(self['width'].content[0])
        return _segmentToPoly(pt1, pt2, width)

    def getPolygon(self):
        T = self.getTransformation()
        return np.array([T([float(s) for s in obj.content])
            for obj in self['polygon']['pts'].getChildren('xy')])

    def getRectangle(self):
        T = self.getTransformation()
        w, h = [float(s) for s in self['size'].content]
        pts = [[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]
        return np.array([T(x) for x in pts])

    def getCircle(self, Npoints = 40):
        T = self.getTransformation()
        if len(self['size'].content) == 1:
            r = float(self['size'].content[0])
        else:
            w, h = [float(s) for s in self['size'].content]
            r = (w + h) / 4
        return T.translation_vector + r*np.array([[np.cos(u), np.sin(u)]
            for u in np.linspace(0, 2*np.pi, Npoints+1)[:-1]])

    def getTextBoundingBox(self):
        T = self.getTransformation()
        text = self.content[-1].strip('"')
        fontSize = float(self['effects']['font']['size'].content[0])
        w = 1.2 * fontSize * len(text)
        h = 2 * fontSize
        pts = [[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]
        return np.array([T(x) for x in pts])

    def toPolygon(self):
        if self.name == 'segment':
            return self.getSegment()
        elif self.name == 'via':
            return self.getCircle()
        elif self.name == 'pad':
            if (self.content[1] in ['thru_hole', 'np_thru_hole'] and
                self.content[2] == 'circle'):
                return self.getCircle()
            elif self.content[1] == 'smd' and self.content[2] == 'rect':
                return self.getRectangle()
            else:
                raise NotImplementedError(f'pad {self.content[1:]}')
        elif self.name == 'zone':
            return self.getPolygon()
        elif self.name in ['gr_text', 'fp_text']:
            return self.getTextBoundingBox()
        else:
            raise NotImplementedError(f'{self.name}')


    # UNPACKING AND MODIFYING COORDINATES

    def getTransformation(self):
        T = Transformation.I()
        obj = self
        while obj is not None:
            lst = list(obj.getChildren('at'))
            if len(lst) == 1:
                params = [float(s) for s in lst[0].content]
                translation_vector = np.array([params[0], params[1]])
                if len(params) == 3:
                    rotation_angle = params[2] * np.pi/180
                else:
                    rotation_angle = 0
                T0 = Transformation(translation_vector, rotation_angle)
            elif len(lst) == 0:
                T0 = Transformation.I()
            elif len(lst) > 1:
                raise ValueError(f'object {obj.name} has multiple position entries')
            T = T0 * T
            obj = obj.parent
        return T

    def getCoordinateObjects(self):
        for obj in self.walk(check = lambda obj: obj.parent is None or len(list(obj.getChildren('at'))) == 0):
            yield from obj.getChildren('at')
            if obj.name in ['at', 'xy', 'start', 'end']:
                yield obj

    def boundingBox(self):
        pts = np.array([[float(obj.content[0]), float(obj.content[1])]
            for obj in self.getCoordinateObjects()])
        return np.min(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 0]), np.max(pts[:, 1])

    def move(self, dx, dy):
        for obj in self.getCoordinateObjects():
            obj.content[0] = str(float(obj.content[0]) + dx)
            obj.content[1] = str(float(obj.content[1]) + dy)

    # UNPACKING AND MODIFYING NETS

    def getNetObjects(self):
        for obj in self.walk():
            if obj.name in ['net', 'net_name', 'add_net', 'net_class']:
                yield obj

    def netNumbers(self):
        # cache this if speed becomes critical
        return set([int(obj.content[0]) for obj in self.getNetObjects()
            if obj.name == 'net'])

    def netDict_numToName(self):
        # cache this if speed becomes critical
        return {int(obj.content[0]): obj.content[1]
            for obj in self.getNetObjects()
            if obj.name == 'net' and len(obj.content) == 2}

    def netDict_nameToNum(self):
        # cache this if speed becomes critical
        return {obj.content[1]: int(obj.content[0])
            for obj in self.getNetObjects()
            if obj.name == 'net' and len(obj.content) == 2}

    def renameNets(self, numberMap, nameMap):
        for obj in self.getNetObjects():
            if obj.name == 'net':
                obj.content[0] = str(numberMap(int(obj.content[0])))
                i = 1
            else:
                i = 0

            if i < len(obj.content):
                if obj.content[i][0] == '"' and obj.content[i][-1] == '"':
                    obj.content[i] = '"' + nameMap(obj.content[i][1:-1]) + '"'
                else:
                    obj.content[i] = nameMap(obj.content[i])

    # UNPACKING AND MODIFYING EDGE CUTS

    def removeEdgeCuts(self):
        for obj in self.walk():
            children_to_keep = []
            for child in obj.children:
                if not (
                        child.name == 'gr_line' and
                        child['layer'].content[0] == '"Edge.Cuts"'):
                    children_to_keep.append(child)
            obj.children = children_to_keep

    def addNet(self, name):
        net_nums = self.netNumbers()
        idx = 1
        while idx in net_nums:
            idx += 1
        pos_list = [i for i, p in enumerate(self.children) if p.name == 'net']
        child = self.addChild(cls = KiCadStructure, name = 'net', content = [str(idx), name],
            position = (max(pos_list) if pos_list else None))
        return idx

    def addLine(self, line_token, x1, y1, x2, y2, width, layer):
        line_token = line_token.split(' ')
        child = self.addChild(cls = KiCadStructure, name = line_token[0],
            content = line_token[1:])
        child.addChild(cls = KiCadStructure, name = 'start', content = [str(x1), str(y1)])
        child.addChild(cls = KiCadStructure, name = 'end', content = [str(x2), str(y2)])
        child.addChild(cls = KiCadStructure, name = 'layer', content = [layer])
        child.addChild(cls = KiCadStructure, name = 'width', content = [str(width)])

    def addGraphicalLine(self, x1, y1, x2, y2, width, layer):
        self.addLine('gr_line', x1, y1, x2, y2, width, layer)
        # child = self.addChild(cls = KiCadStructure, name = 'gr_line')
        # child.addChild(cls = KiCadStructure, name = 'start', content = [str(x1), str(y1)])
        # child.addChild(cls = KiCadStructure, name = 'end', content = [str(x2), str(y2)])
        # child.addChild(cls = KiCadStructure, name = 'layer', content = [layer])
        # child.addChild(cls = KiCadStructure, name = 'width', content = [str(width)])

    def addFootprintLine(self, x1, y1, x2, y2, width, layer):
        self.addLine('fp_line', x1, y1, x2, y2, width, layer)

    def addArc(self, arc_token, ptStart, ptMid, ptEnd, width, layer):
        arc_token = arc_token.split(' ')
        child = self.addChild(cls = KiCadStructure, name = arc_token[0],
            content = arc_token[1:])
        child.addChild(cls = KiCadStructure, name = 'start', content = [str(ptStart[0]), str(ptStart[1])])
        child.addChild(cls = KiCadStructure, name = 'mid', content = [str(ptMid[0]), str(ptMid[1])])
        child.addChild(cls = KiCadStructure, name = 'end', content = [str(ptEnd[0]), str(ptEnd[1])])
        child.addChild(cls = KiCadStructure, name = 'layer', content = [layer])
        child.addChild(cls = KiCadStructure, name = 'width', content = [str(width)])

    def addGraphicalArc(self, ptStart, ptMid, ptEnd, width, layer):
        self.addArc('gr_arc', ptStart, ptMid, ptEnd, width, layer)

    def addFootprintArc(self, ptStart, ptMid, ptEnd, width, layer):
        self.addArc('fp_arc', ptStart, ptMid, ptEnd, width, layer)

    def addGraphicalCircle(self, xc, yc, radius, width, layer):
        child = self.addChild(cls = KiCadStructure, name = 'gr_circle')
        child.addChild(cls = KiCadStructure, name = 'center', content = [str(xc), str(yc)])
        child.addChild(cls = KiCadStructure, name = 'end', content = [str(xc + radius), str(yc)])
        child.addChild(cls = KiCadStructure, name = 'layer', content = [layer])
        child.addChild(cls = KiCadStructure, name = 'width', content = [str(width)])

    def addText(self, text_token, text, x, y, size, layer, orientation = None):
        text_token = text_token.split(' ')
        child = self.addChild(cls = KiCadStructure, name = text_token[0],
            content = text_token[1:] + [f'"{text}"'])
        child.addChild(cls = KiCadStructure, name = 'at',
            content = [str(x), str(y)] + ([] if orientation is None else [str(orientation)]))
        child.addChild(cls = KiCadStructure, name = 'layer', content = [layer])
        effects = child.addChild(cls = KiCadStructure, name = 'effects')
        font = effects.addChild(cls = KiCadStructure, name = 'font')
        font.addChild(cls = KiCadStructure, name = 'size', content = [str(size), str(size)])
        font.addChild(cls = KiCadStructure, name = 'thickness', content = [str(size/5)])

    def addGraphicalText(self, text, x, y, size, layer, orientation = None):
        self.addText('gr_text', text, x, y, size, layer, orientation = orientation)

    def addFootprintReferenceText(self, text, x, y, size, layer, orientation = None):
        self.addText('fp_text reference', text, x, y, size, layer, orientation = orientation)

    def addFootprintValueText(self, text, x, y, size, layer, orientation = None):
        self.addText('fp_text value', text, x, y, size, layer, orientation = orientation)

    def addSegment(self, pt1, pt2, width, layer, net):
        child = self.addChild(cls = KiCadStructure, name = 'segment')
        child.addChild(cls = KiCadStructure, name = 'start', content = [str(pt1[0]), str(pt1[1])])
        child.addChild(cls = KiCadStructure, name = 'end', content = [str(pt2[0]), str(pt2[1])])
        child.addChild(cls = KiCadStructure, name = 'layer', content = [layer])
        child.addChild(cls = KiCadStructure, name = 'width', content = [str(width)])
        child.addChild(cls = KiCadStructure, name = 'net', content = [str(net)])

    def addPolySegment(self, pts, width, layer, net):
        for i in range(len(pts) - 1):
            self.addSegment(pts[i], pts[i+1], width, layer, net)

    def addEdgeCutLine(self, x1, y1, x2, y2):
        self.addGraphicalLine(x1, y1, x2, y2, 0.05, 'Edge.Cuts')

    def addEdgeCutArc(self, ptStart, ptMid, ptEnd):
        self.addGraphicalArc(ptStart, ptMid, ptEnd, 0.05, 'Edge.Cuts')

    def addEdgeCutCircle(self, xc, yc, radius):
        self.addGraphicalCircle(xc, yc, radius, 0.05, 'Edge.Cuts')

    def addEdgeCutPolyLine(self, pts):
        for i in range(len(pts) - 1):
            self.addEdgeCutLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])

    def addEdgeCutRectangle(self, x1, y1, x2, y2):
        self.addEdgeCutLine(x1, y1, x2, y1)
        self.addEdgeCutLine(x2, y1, x2, y2)
        self.addEdgeCutLine(x2, y2, x1, y2)
        self.addEdgeCutLine(x1, y2, x1, y1)

    def addHole(self, pt, drill_diameter):
        child = self.addChild(cls = KiCadStructure, name = 'module', content = ['Hole'])
        child.addChild(cls = KiCadStructure, name = 'at', content = [str(pt[0]), str(pt[1])])
        child2 = child.addChild(cls = KiCadStructure, name = 'pad', content = ['1', 'np_thru_hole', 'circle'])
        child2.addChild(cls = KiCadStructure, name = 'at', content = ['0', '0'])
        child2.addChild(cls = KiCadStructure, name = 'size', content = [str(drill_diameter), str(drill_diameter)])
        child2.addChild(cls = KiCadStructure, name = 'drill', content = [str(drill_diameter)])
        child2.addChild(cls = KiCadStructure, name = 'layers', content = ['*.Cu', '*.Mask'])

    def addVia(self, x, y, diameter, drill_diameter, net):
        if not isinstance(net, int):
            net = self.netDict_nameToNum()[net]
        child = self.addChild(cls = KiCadStructure, name = 'via')
        child.addChild(cls = KiCadStructure, name = 'at', content = [str(x), str(y)])
        child.addChild(cls = KiCadStructure, name = 'size', content = [str(diameter)])
        child.addChild(cls = KiCadStructure, name = 'drill', content = [str(drill_diameter)])
        child.addChild(cls = KiCadStructure, name = 'layers', content = ['F.Cu', 'B.Cu'])
        child.addChild(cls = KiCadStructure, name = 'net', content = [str(net)])

    # (via (at 130.46 67.07) (size 0.8) (drill 0.4) (layers F.Cu B.Cu) (net 0))

    def addEdgeCutLozenge(self, x1, y1, x2, y2, width):
        pt1 = np.array([x1, y1])
        pt2 = np.array([x2, y2])
        e = pt2 - pt1
        e = e / np.linalg.norm(e)
        n = np.array([1, -1]) * e[::-1]
        self.addEdgeCutLine(*(list(pt1 + n * width / 2) + list(pt2 + n * width / 2)))
        self.addEdgeCutLine(*(list(pt1 - n * width / 2) + list(pt2 - n * width / 2)))
        self.addEdgeCutArc(pt1 + n * width / 2, pt1 - e * width / 2, pt1 - n * width / 2)
        self.addEdgeCutArc(pt2 + n * width / 2, pt2 + e * width / 2, pt2 - n * width / 2)

    def merge(self, other):
        Structure.merge(self, other,
            names_toMerge = ['gr_line', 'gr_poly', 'gr_text', 'module', 'segment', 'via', 'zone', 'net', 'net_class', 'footprint'],
            names_toCompare = ['general', 'host', 'layers', 'page', 'setup', 'version', 'generator', 'paper'])

    def toPCBstring(self, indent = ''):
        content_copy = self.content.copy()
        if 'hide' in content_copy:
            extra = ' hide'
            content_copy.remove('hide')
        else:
            extra = ''
        s = (indent + '(' + self.name + ' ' + ' '.join(content_copy) +
            ''.join(['\n' + child.toPCBstring(indent = '  '+indent)
            for child in self.children]) + extra + ')')
        return s

    def toPCBfile(self, filename):
        with open(filename, 'w') as f:
            f.write(self.toPCBstring())

    def placeAt(self, obj, pt, orientation):
        obj_copy = obj.copy()
        obj_copy.addChild(cls = KiCadStructure, name = 'at', content = [
            str(pt[0]), str(pt[1]), str(orientation)])
        self.addAsChild(obj_copy)
        return obj_copy

    @classmethod
    def footprint(cls, name, layer):
        obj = cls(name = 'footprint', content = [name])
        obj.addChild(cls = KiCadStructure, name = 'layer', content = [layer])
        return obj

    @classmethod
    def pad_th(cls, pt, idx, diameter, drill_diameter, layers, net, net_name):
        obj = cls(name = 'pad', content = [str(idx), 'thru_hole', 'circle'])
        obj.addChild(cls = KiCadStructure, name = 'at', content = [str(x) for x in pt])
        obj.addChild(cls = KiCadStructure, name = 'size', content = [str(diameter), str(diameter)])
        obj.addChild(cls = KiCadStructure, name = 'drill', content = [str(drill_diameter)])
        obj.addChild(cls = KiCadStructure, name = 'layers', content = layers)
        obj.addChild(cls = KiCadStructure, name = 'net', content = [str(net), net_name])
        return obj

    @classmethod
    def pad_rect(cls, pt, idx, width, height, layers, net, net_name):
        obj = cls(name = 'pad', content = [str(idx), 'smd', 'rect'])
        obj.addChild(cls = KiCadStructure, name = 'at', content = [str(x) for x in pt])
        obj.addChild(cls = KiCadStructure, name = 'size', content = [str(width), str(height)])
        obj.addChild(cls = KiCadStructure, name = 'layers', content = layers)
        obj.addChild(cls = KiCadStructure, name = 'net', content = [str(net), net_name])
        return obj

    @classmethod
    def via(cls, pt, diameter, drill_diameter, layers, net):
        obj = cls(name = 'via')
        obj.addChild(cls = KiCadStructure, name = 'at', content = [str(x) for x in pt])
        obj.addChild(cls = KiCadStructure, name = 'size', content = [str(diameter)])
        obj.addChild(cls = KiCadStructure, name = 'drill', content = [str(drill_diameter)])
        obj.addChild(cls = KiCadStructure, name = 'layers', content = [str(layers[0]), str(layers[1])])
        obj.addChild(cls = KiCadStructure, name = 'net', content = [str(net)])
        return obj

    @classmethod
    def filledZone(cls, net, net_name, layer, pts, clearance = None):
        obj = cls(name = 'zone')
        obj.addChild(cls = KiCadStructure, name = 'net', content = [str(net)])
        obj.addChild(cls = KiCadStructure, name = 'net_name', content = [str(net_name)])
        obj.addChild(cls = KiCadStructure, name = 'layer', content = [str(layer)])
        child = obj.addChild(cls = KiCadStructure, name = 'connect_pads', content = ['yes']) # TODO: make this an argument
        if clearance is not None:
            child.addChild(cls = KiCadStructure, name = 'clearance', content = [str(clearance)])
        obj.addChild(cls = KiCadStructure, name = 'fill', content = ['yes'])
        obj.addAsChild(cls.polygon(pts))
        return obj

    @classmethod
    def ruleArea(cls, layer, keepout, pts):
        obj = cls(name = 'zone')
        obj.addChild(cls = KiCadStructure, name = 'net', content = ['0'])
        obj.addChild(cls = KiCadStructure, name = 'net_name', content = ['""'])
        obj.addChild(cls = KiCadStructure, name = 'layer', content = [str(layer)])
        child = obj.addChild(cls = KiCadStructure, name = 'keepout')
        for obj_name in ['tracks', 'vias', 'pads', 'copperpour', 'footprints']:
            child.addChild(cls = KiCadStructure, name = obj_name, content = [
                'not_allowed' if obj_name in keepout else 'allowed'
                ])
        # (keepout (tracks allowed) (vias allowed) (pads allowed) (copperpour not_allowed) (footprints not_allowed))
        obj.addAsChild(cls.polygon(pts))
        return obj

    @classmethod
    def filledZoneCircle(cls, net, net_name, layer, xc, yc, radius, Npts = 100, **kwargs):
        pts = [[xc + radius*np.cos(u), yc + radius*np.sin(u)]
            for u in np.linspace(0, 2*np.pi, Npts+1)[:-1]]
        return cls.filledZone(net, net_name, layer, pts, **kwargs)

    @classmethod
    def filledZoneRectangle(cls, net, net_name, layer, x1, y1, x2, y2, **kwargs):
        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return cls.filledZone(net, net_name, layer, pts, **kwargs)

    @classmethod
    def polygon(cls, pts):
        obj = cls(name = 'polygon')
        child = obj.addChild(cls = KiCadStructure, name = 'pts')
        for x, y in pts:
            child.addChild(cls = KiCadStructure, name = 'xy', content = [str(x), str(y)])
        return obj

    @classmethod
    def fromPCBfile(cls, filename):
        root = cls(content = [''])
        current_node = root
        parsing_string = False

        with open(filename, 'r') as f:
            for line in f:
                for ch in line:
                    if ch == '"':
                        parsing_string = not parsing_string
                    if ch == '(' and not parsing_string:
                        # opening node
                        current_node = current_node.addChild(content = [''])
                    elif ch == ')' and not parsing_string:
                        # closing node
                        if not current_node.content[-1]:
                            current_node.content = current_node.content[:-1]
                        current_node.name = current_node.content[0]
                        current_node.content = current_node.content[1:]
                        current_node = current_node.parent
                    else:
                        # whitespace -> new entry if previous is non-empty
                        if ch in [' ', '\n'] and not parsing_string:
                            if current_node.content[-1]:
                                current_node.content.append('')
                        else:
                            current_node.content[-1] += ch
        return root.children[0]
