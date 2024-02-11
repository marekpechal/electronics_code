import sys, os
filepath = os.path.dirname(os.path.realpath(__file__))
libpath = os.path.normpath(os.path.join(filepath, '..\\..'))
sys.path.insert(-1, libpath)

from KiCadStructure import KiCadStructure
import os

def simplify_string(s):
    # remove \n and \t
    s = s.replace('\t', '').replace('\n', '')
    # strip multiple spaces
    while True:
        s_new = s.replace('  ', ' ')
        if s == s_new: break
        s = s_new
    # remove extra space between brackets
    while True:
        s_new = s.replace('( (', '((')
        if s == s_new: break
        s = s_new
    while True:
        s_new = s.replace(') )', '))')
        if s == s_new: break
        s = s_new
    return s

input_filename = 'E:\\electronics\\DemoPCB\\DemoPCB.kicad_pcb'
output_filename = '_copy.kicad_pcb'

board = KiCadStructure.fromPCBfile(input_filename)
board.toPCBfile(output_filename)

with open(input_filename, 'r') as f:
    s_input = f.read()

with open(output_filename, 'r') as f:
    s_output = f.read()

os.remove(output_filename)

s_input = simplify_string(s_input)
s_output = simplify_string(s_output)

print(s_input == s_output)
