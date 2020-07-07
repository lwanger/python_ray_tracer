"""
Get the bbox and triangle count for an STL mesh
"""

from pathlib import Path
import sys

from PIL import Image
from stl import mesh  # numpy-stl

from geometry_classes import Vec3
from primitives_classes import STLMesh
from texture_classes import SolidColor
from material_classes import Lambertian


if __name__ == '__main__':
    try:
        filename = sys.argv[1]
    except:
        print(f'usage: {sys.argv[0]} <stl filename>')

    matl = Lambertian(SolidColor(Vec3(0,0,0)))
    stl_filename = Path(filename)
    my_mesh = mesh.Mesh.from_file(stl_filename)
    stl_mesh = STLMesh(my_mesh, matl, name="stl_mesh")
    print(f'stl_mesh {stl_filename} -- bbox={stl_mesh.bounding_box(None, None)}, num_triangles={stl_mesh.num_triangles}')
