import sys, os
filepath = os.path.dirname(os.path.realpath(__file__))
libpath = os.path.normpath(os.path.join(filepath, '..\\..'))
sys.path.insert(-1, libpath)

from KiCadStructure import KiCadStructure
import os

input_filename = 'DemoPCB.kicad_pcb'
output_filename = '_copy.kicad_pcb'

board = KiCadStructure.fromPCBfile(input_filename)
board.addNet('GND')

for i in range(10):
    for j in range(10):
        board.addVia(100.0 + 1.0*i, 100.0 + 1.0*j, 0.5, 0.3, 'GND')
board.toPCBfile(output_filename)
