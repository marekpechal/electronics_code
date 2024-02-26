import sys, os
filepath = os.path.dirname(os.path.realpath(__file__))
libpath = os.path.normpath(os.path.join(filepath, '..\\..'))
sys.path.insert(-1, libpath)

import helpers
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize = (6, 6))

path = np.array([[0.0, 0.0], [100.0, 0.0], [150.0, 50.], [50.0, 50.0]])
plt.plot(path[:, 0], path[:, 1], '.-')

points = helpers.distributePointsOnPath(path,
    5.0, init_gap = 20.0, fin_gap = 50.0)
plt.plot(points[:, 0], points[:, 1], '.')

points = helpers.distributePointsOnPath(helpers.offsetPath(path, 10.0),
    5.0, init_gap = 20.0, fin_gap = 50.0)
plt.plot(points[:, 0], points[:, 1], '.')

points = helpers.distributePointsOnPath(helpers.offsetPath(path, -10.0),
    5.0, init_gap = 20.0, fin_gap = 50.0)
plt.plot(points[:, 0], points[:, 1], '.')

plt.grid()
plt.xlim((-15.0, 185.0))
plt.ylim((-75.0, 125.0))

plt.show()
