from pathlib import Path

import numpy as np
from stl import mesh  # numpy-stl

from mpl_toolkits import mplot3d
from matplotlib import pyplot


#FILENAME="models/sauce_ramp_v2.stl"
#FILENAME="models/LRW pick (plate stiffener).stl"
#FILENAME="models/gyroid_20mm.stl"
#FILENAME="models/modern_hexagon_revisited.stl"
FILENAME="models/bar_6mm.stl"


def bbox(my_mesh):
    min_x = my_mesh.x.min_val()
    max_x = my_mesh.x.max_val()
    min_y = my_mesh.y.min_val()
    max_y = my_mesh.y.max_val()
    min_z = my_mesh.z.min_val()
    max_z = my_mesh.z.max_val()
    return ((min_x, min_y, min_z), (max_x, max_y, max_z))

def scale(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


stl_filename = Path(FILENAME)

my_mesh = mesh.Mesh.from_file(stl_filename)

points = my_mesh.points.shape[0]

print(f'my_mesh num_triangles={my_mesh.points.shape[0]},  shape={my_mesh.points.shape}')

b = bbox(my_mesh)
print(f'my_mesh bbox={b}')

# v0 = my_mesh.points[0][0:3]
v0 = my_mesh.points[0][0:3].tolist()
v1 = my_mesh.points[0][3:6].tolist()
v2 = my_mesh.points[0][6:9].tolist()
print(f'my_mesh triangle_1={(v0,v1,v2)}')

# scale
new_mesh_data = scale(my_mesh, -1, 1)
new_mesh = mesh.Mesh(new_mesh_data)
b = bbox(new_mesh)
print(f'new_mesh bbox={b}')

figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(my_mesh.vectors))

scale = my_mesh.points.flatten()
axes.auto_scale_xyz(scale,scale,scale)

pyplot.show()  # doesn't work in pycharm -- gives mac() arg is an empty sequence
