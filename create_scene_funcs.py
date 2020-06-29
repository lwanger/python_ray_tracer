"""
Scene creation functions

Listing 60 from Pete Shirley's Ray Tracing in a Weekend:

https://raytracing.github.io/books/RayTracingInOneWeekend.html

Len Wanger -- 2020
"""

import math
from random import random, uniform

from pathlib import Path
from PIL import Image

from stl import mesh  # numpy-stl

from geometry_classes import Vec3, GeometryList, Scene, Camera
from geometry_classes import random_on_unit_sphere
from material_classes import Lambertian, Metal, Dielectric
from texture_classes import SolidColor, CheckerBoard, ImageTexture
from primitives_classes import Sphere, Plane, Triangle, STLMesh


def create_simple_world():
    color_1 = SolidColor(Vec3(0.7, 0.3, 0.3))
    color_2 = SolidColor(Vec3(0.8, 0.8, 0))
    color_3 = SolidColor(Vec3(0.8,0.6,0.2))

    diffuse_1 = Lambertian(color_1, name="diffuse_1")
    diffuse_2 = Lambertian(color_2, name="diffuse_2")
    metal_1 = Metal(color_3, fuzziness=0.3, name="metal_1")
    dielectric_1 = Dielectric(1.5, name="dielectric_1")

    world = GeometryList()
    world.add(Sphere(Vec3(0,0,-1), 0.5, diffuse_1))
    world.add(Sphere(Vec3(0,-100.5,-1), 100, diffuse_2))
    world.add(Sphere(Vec3(1,0,-1), 0.5, metal_1))
    world.add(Sphere(Vec3(-1,0,-1),0.5, dielectric_1))
    world.add(Sphere(Vec3(-1,0,-1),-0.45, dielectric_1))  # hollow sphere

    scene = Scene(world)
    camera = Camera(look_from=Vec3(-0.5, 1, 5), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.1,
                    focus_dist=20)
    return {'scene': scene, 'camera': camera}


def create_simple_world_2():
    # use a plane instead of a big sphere!
    color_3 = SolidColor(Vec3(0.2, 0.2, 0.7))
    color_4 = SolidColor(Vec3(0.8,0.6,0.2))
    color_5 = SolidColor(Vec3(0.4,0.4,0.4))

    diffuse_3 = Lambertian(color_3, name="diffuse_3")
    metal_1 = Metal(color_4, fuzziness=0.3, name="metal_1")
    metal_2 = Metal(color_5, fuzziness=0.0, name="metal_2")

    world = GeometryList()

    world.add(Sphere(Vec3(0,0,-1), 1.5, metal_1))

    plane_1 = Plane.plane_from_point_and_normal(pt=Vec3(0,-3,0), normal=Vec3(0,1,0), material=diffuse_3)
    plane_2 = Plane.plane_from_point_and_normal(pt=Vec3(0,0,-10), normal=Vec3(0,0,1), material=metal_2)

    world.add(plane_1)
    world.add(plane_2)

    scene = Scene(world)
    camera = Camera(look_from=Vec3(-0.5, 1, 10), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.1,
                    focus_dist=20)
    return {'scene': scene, 'camera': camera}


def create_simple_world_3():
    # add triangles
    color_1 = SolidColor(Vec3(0.7, 0.3, 0.3))
    color_2 = SolidColor(Vec3(0.2, 0.2, 0.7))
    color_3 = SolidColor(Vec3(0.4,0.4,0.4))

    diffuse_1 = Lambertian(color_1)
    diffuse_3 = Lambertian(color_2)
    metal_2 = Metal(color_3, fuzziness=0.0)
    dielectric_1 = Dielectric(1.5)

    world = GeometryList()

    world.add(Sphere(Vec3(0,0,-1), 1.5, metal_2))

    v0 = Vec3(-1.8, -0.5, 1.5)
    v1 = Vec3(-1.0, 0.5, 1.5)
    v2 = Vec3(-0.2, -0.5, 1.5)
    world.add(Triangle(v0,v1,v2,diffuse_1))

    v0 = Vec3(1.8, -0.5, 1.5)
    v1 = Vec3(1.0, 0.5, 1.5)
    v2 = Vec3(0.2, -0.5, 1.5)
    world.add(Triangle(v0, v1, v2, metal_2))

    v0 = Vec3(-1.0, 0.8, 1.5)
    v1 = Vec3(0.0, 2.5, 0.75)
    v2 = Vec3(1.0, 0.8, 1.5)
    world.add(Triangle(v0, v1, v2, dielectric_1))

    plane_1 = Plane.plane_from_point_and_normal(pt=Vec3(0,-3,0), normal=Vec3(0,1,0), material=diffuse_3)
    world.add(plane_1)

    scene = Scene(world)
    # camera = Camera(look_from=Vec3(-0.5, 1, 13), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.1, focus_dist=20)
    camera = Camera(look_from=Vec3(1, 1, 13), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.1, focus_dist=20)
    return {'scene': scene, 'camera': camera}


def create_random_world():
    world = GeometryList()

    color_1 = SolidColor(Vec3(0.5,0.5,0.5))

    ground_material = Lambertian(color_1)
    glass_material = Dielectric(1.5)
    center_offset = Vec3(4, 0.2, 9)

    world.add(Sphere(Vec3(0,-1000,0), 1000, ground_material))

    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random()
            center = Vec3(a+0.9*random(), 0.2, b+0.9*random())

            if (center - center_offset).length() > 0.9:
                if choose_mat < 0.8:  # diffuse
                    r = random()*random()
                    g = random()*random()
                    b = random()*random()
                    albedo = SolidColor(Vec3(r,g,b))
                    sphere_material = Lambertian(albedo)
                elif choose_mat < 0.95:  # metal
                    a = uniform(0.5, 1.0)
                    fuzz = uniform(0.0, 0.5)
                    albedo = SolidColor(Vec3(a,a,a))

                    sphere_material = Metal(albedo, fuzz)
                else:  # glass
                    sphere_material = glass_material

                world.add(Sphere(center, 0.2, sphere_material))

    material_1 = Dielectric(1.5)
    world.add(Sphere(Vec3(0,1,0), 1.0, material_1))

    color_2 = SolidColor(Vec3(0.4, 0.2, 0.1))
    material_2 = Lambertian(color_2)
    world.add(Sphere(Vec3(-4, 1, 0), 1.0, material_2))

    color_3 = SolidColor(Vec3(0.7,0.6,0.5))
    material_3 = Metal(color_3, 0.0)
    world.add(Sphere(Vec3(4, 1, 0), 1.0, material_3))

    scene = Scene(world)
    camera = Camera(look_from=Vec3(13, 2, 3), look_at=Vec3(0, 0, 0), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.1, focus_dist=20)
    return {'scene': scene, 'camera': camera}


def create_random_world2():
    def random_material():
        choose_mat = random()

        if choose_mat < 0.8:  # diffuse
            r = random() * random()
            g = random() * random()
            b = random() * random()
            albedo = SolidColor(Vec3(r, g, b))
            material = Lambertian(albedo)
        elif choose_mat < 0.95:  # metal
            a = uniform(0.5, 1.0)
            albedo = SolidColor(Vec3(a,a,a))
            fuzz = uniform(0.0, 0.5)
            material = Metal(albedo, fuzz)
        else:  # glass
            material = glass_material

        return material

    # a ground plane, a metal sphere and random triangles...
    world = GeometryList()

    ground_color = SolidColor(Vec3(0.2,0.6,0.2))
    ground_material = Lambertian(ground_color)

    metal_1 = Metal(SolidColor(Vec3(0.7,0.6,0.5)), fuzziness=0.0)
    metal_2 = Metal(SolidColor(Vec3(0.4,0.4,0.4)), fuzziness=0.3)
    glass_material = Dielectric(1.5)
    center_offset = Vec3(4, 0.2, 9)

    plane_1 = Plane.plane_from_point_and_normal(pt=Vec3(0, -3, 0), normal=Vec3(0, 1, 0), material=ground_material)
    world.add(plane_1)

    world.add(Sphere(Vec3(0,0,-1), 1.5, metal_1))

    for a in range(-12, 12):
        for b in range(-12, 12):
            center = Vec3(a+0.9*random(), 3*random()+0.3, b+0.9*random())

            if (center - center_offset).length() > 0.9:
                material = random_material()
                v0 = random_on_unit_sphere().mul_val(0.7) + center
                v1 = random_on_unit_sphere().mul_val(0.7) + center
                v2 = random_on_unit_sphere().mul_val(0.7) + center
                triangle = Triangle(v0,v1,v2, material)
                world.add(triangle)

    scene = Scene(world)
    camera = Camera(look_from=Vec3(13, 2, 3), look_at=Vec3(0, 0, 0), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.1, focus_dist=20)
    return {'scene': scene, 'camera': camera}


def create_checkerboard_world():
    color_1 = SolidColor(Vec3(0.7, 0.3, 0.3))
    color_2 = SolidColor(Vec3(0.8,0.6,0.2))

    odd_color = SolidColor(Vec3(0.2,0.3,0.1))
    even_color = SolidColor(Vec3(0.9,0.9,0.9))
    checker_board = CheckerBoard(even_color, odd_color, spacing=3)

    other_odd_color = SolidColor(Vec3(0.1, 0.1, 0.1))
    other_even_color = SolidColor(Vec3(0.9, 0.1, 0.1))
    checker_board_2 = CheckerBoard(other_even_color, other_odd_color, spacing=8)

    diffuse_1 = Lambertian(color_1, name="diffuse_1")
    diffuse_2 = Lambertian(checker_board, name="diffuse_checkerboard")
    diffuse_3 = Lambertian(checker_board_2, name="diffuse_checkerboard_2")
    metal_1 = Metal(color_2, fuzziness=0.3, name="metal_1")
    dielectric_1 = Dielectric(1.5, name="dielectric_1")

    world = GeometryList()
    world.add(Sphere(Vec3(0,0,-1), 0.5, diffuse_1))

    v0 = Vec3(-2, 0.1, -2.5)
    v1 = Vec3(2, 0.1, -2.5)
    v2 = Vec3(0, 1.5, -2.0)

    world.add(Triangle(v0, v1, v2, diffuse_3, uv0=(0.5,1), uv1=(1,0), uv2=(0,0)))

    plane_1 = Plane.plane_from_point_and_normal(pt=Vec3(0, -3, 0), normal=Vec3(0, 1, 0), material=diffuse_2)
    world.add(plane_1)

    world.add(Sphere(Vec3(1.2,0,-1), 0.5, metal_1))
    world.add(Sphere(Vec3(-1.2,0,-1),0.5, dielectric_1))
    world.add(Sphere(Vec3(-1.2,0,-1),-0.45, dielectric_1))  # hollow sphere

    scene = Scene(world)
    camera = Camera(look_from=Vec3(-0.5, 1, 5), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=30)
    return {'scene': scene, 'camera': camera}


def create_checkerboard_world_2():
    odd_color = SolidColor(Vec3(0.2,0.3,0.1))
    even_color = SolidColor(Vec3(0.9,0.9,0.9))
    checker_board = CheckerBoard(even_color, odd_color, spacing=3)

    diffuse_1 = Lambertian(checker_board, name="diffuse_checkerboard")
    metal_1 = Metal(checker_board, fuzziness=0.1, name="metal_checkerboard")

    world = GeometryList()
    world.add(Sphere(Vec3(0,-10,0), 10, diffuse_1))
    world.add(Sphere(Vec3(0,10,0), 10, metal_1))

    scene = Scene(world)
    camera = Camera(look_from=Vec3(-0.5, 1, 15), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=20)

    return {'scene': scene, 'camera': camera}


def create_image_texture_world():
    silver = SolidColor(Vec3(0.7, 0.7, 0.7))

    image_1 = Image.open(Path("./textures/earthlights_dmsp_big.jpg"))
    image_2 = Image.open(Path("./textures/george harrison (1 bit).bmp"))
    image_texture_1 = ImageTexture(image_1, "earthlights")

    odd_color = ImageTexture(image_2)
    even_color = SolidColor(Vec3(0.1, 0.1, 0.1))
    checker_board = CheckerBoard(even_color, odd_color, spacing=3)

    diffuse_1 = Lambertian(image_texture_1, name="diffuse_1")
    diffuse_2 = Lambertian(checker_board, name="checkerboard")
    metal_1 = Metal(silver, name="metal_1")

    world = GeometryList()

    world.add(Sphere(Vec3(-0.75, 0, -1), 1.0, diffuse_1))
    world.add(Sphere(Vec3(1.5, 0, -2.25), 1.0, metal_1))

    plane_1 = Plane.plane_from_point_and_normal(pt=Vec3(0, -3, 0), normal=Vec3(0, 1, 0), material=diffuse_2)
    world.add(plane_1)

    scene = Scene(world)
    camera = Camera(look_from=Vec3(-0.5, 1, 7), look_at=Vec3(0, 0, -0.5), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.1, focus_dist=20)

    return {'scene': scene, 'camera': camera}


def create_canonical_1():
    """
    debug:
        - only half of checkerboard showingup
        - no io_logo image showing up on checkerboard
        X reflection on sphere in wrong place?
    """
    silver = SolidColor(Vec3(0.7, 0.7, 0.7))

    # image_1 = Image.open(Path("./textures/earthlights_dmsp_big.jpg"))
    image_2 = Image.open(Path("./textures/IO_logo.png"))
    # image_texture_1 = ImageTexture(image_1, "earthlights")

    logo = ImageTexture(image_2)  # images are across the entire checkerboard, not a single square?

    # odd_color = ImageTexture(image_2)  # images are across the entire checkerboard, not a single square?
    odd_color = SolidColor(Vec3(0.2, 0.75, 0.2))
    even_color = SolidColor(Vec3(0.1, 0.1, 0.1))
    checker_board = CheckerBoard(even_color, odd_color, spacing=2.0)

    if False:  # use checkerboard vs image texture
        diffuse_2 = Lambertian(checker_board, name="checkerboard")
    else:
        diffuse_2 = Lambertian(logo, name="io_logo'")

    metal_1 = Metal(silver, name="metal_1")

    world = GeometryList()

    world.add(Sphere(Vec3(0, 1.25, 0.35), 1.0, metal_1))

    if False:  # use plane vs triangles
        plane_1 = Plane.plane_from_point_and_normal(pt=Vec3(0, -1, 0), normal=Vec3(0, 1, 0), material=diffuse_2)
        world.add(plane_1)
    else:
        plane_x = 2
        plane_y = -1
        back_plane_z = -2
        front_plane_z = 3
        v0 = Vec3(-plane_x, plane_y, front_plane_z)
        uv0 = (0,0)
        v1 = Vec3(-plane_x ,plane_y,back_plane_z)
        uv1 = (0, 1)
        v2 = Vec3(plane_x, plane_y, front_plane_z)
        uv2 = (1, 0)
        v3 = Vec3(plane_x, plane_y,back_plane_z)
        uv3 = (1, 1)
        triangle = Triangle(v0, v1, v2, diffuse_2, uv0, uv1, uv2)
        world.add(triangle)

        triangle = Triangle(v1, v2, v3, diffuse_2, uv1, uv2, uv3)
        world.add(triangle)

    scene = Scene(world)
    camera = Camera(look_from=Vec3(8.5, 4, 0), look_at=Vec3(0, 1, 0), vup=Vec3(0, 1, 0), vert_fov=25)

    return {'scene': scene, 'camera': camera}


# def scale(np, x, out_range=(-1, 1), axis=None):
#     domain = np.min(x, axis), np.max(x, axis)
#     y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
#     return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def create_stl_mesh():
    """
    TODO:
        scale the model
    """
    silver = SolidColor(Vec3(0.7, 0.7, 0.7))
    green = SolidColor(Vec3(0.1, 0.5, 0.1))
    blue = SolidColor(Vec3(0.1, 0.1, 0.5))
    red = SolidColor(Vec3(0.5, 0.1, 0.1))
    purple = SolidColor(Vec3(0.4, 0.1, 0.4))
    gray = SolidColor(Vec3(0.2, 0.2, 0.2))
    light_gray = SolidColor(Vec3(0.9, 0.9, 0.9))
    dark_gray = SolidColor(Vec3(0.1, 0.1, 0.1))
    black = SolidColor(Vec3(0.0, 0.0, 0.0))

    # stl_filename = "models/sauce_ramp_v2.stl"  # vmin=(-1.60, -0.69, 0.0), vmax=(0.97, 0.69, 1.19)
    # settings = {'look_from': Vec3(0.0, 2, 5), 'look_at': Vec3(0, 0.0, 0),
    #     'plane_x': 5, 'plane_y': -2, 'back_plane_z': -3, 'front_plane_z': 3,
    #     'rot_axis': [0.0, 0.5, 0.0], 'rot_rads': math.radians(90), 'show_walls': False}

    # stl_filename ="models/LRW pick (plate stiffener).stl" # vmin=(-365.174, -471.391, 12.700), vmax=(365.313, 341.873, 50.800)
    # setting not right yet!
    # settings = {'look_from': Vec3(0.0, 2, 300), 'look_at': Vec3(0, 0.0, 0),
    #             'plane_x': 400, 'plane_y': -500, 'back_plane_z': 0, 'front_plane_z': 60,
    #             'rot_axis': None, 'rot_rads': None, 'show_walls': False}

    stl_filename ="models/gyroid_20mm.stl"
    # settings = {'look_from': Vec3(0.0, 4, 60), 'look_at': Vec3(0, 0.0, 0),
    settings = {'look_from': Vec3(-10.0, 25, 60), 'look_at': Vec3(0, 0.0, 0),
                # 'plane_x': 15, 'plane_y': -13, 'back_plane_z': -15, 'front_plane_z': 15,
                'plane_x': 25, 'plane_y': -15, 'back_plane_z': -20, 'front_plane_z': 15,
                'rot_axis': None, 'rot_rads': None, 'show_walls': True}

    # stl_filename ="models/modern_hexagon_revisited.stl"
    # stl_filename ="models/bar_6mm.stl"
    # settings = {'look_from': Vec3(50, 25, 50), 'look_at': Vec3(0, 0.5, 0),
    #             'plane_x': 20, 'plane_y': -5, 'back_plane_z': -50, 'front_plane_z': 50, }

    image_2 = Image.open(Path("./textures/IO_logo.png"))
    logo = ImageTexture(image_2)  # images are across the entire checkerboard, not a single square?

    checked = CheckerBoard(dark_gray, light_gray, spacing=0.5)

    metal_1 = Metal(silver, name="metal_1")

    # diffuse_2 = Lambertian(logo, name="io_logo'")
    diffuse_2 = Lambertian(gray, name="gray'")

    object_matl = metal_1

    # ground = Lambertian(green, name="green'")
    # ground = Lambertian(logo, name="gray'")
    ground = Lambertian(checked, name="gray'")
    # right_wall = Lambertian(red, name="red'")
    right_wall = Lambertian(logo, name="red'")
    # right_wall = metal_1
    left_wall = Lambertian(blue, name="blue'")
    # left_wall = metal_1
    # back_wall = Lambertian(purple, name="purple'")
    back_wall = metal_1

    world = GeometryList()

    # world.add(Sphere(Vec3(0, 1.25, 0.35), 1.0, metal_1))

    if True:
        plane_x = settings['plane_x']
        plane_y = settings['plane_y']
        back_plane_z = settings['back_plane_z']
        front_plane_z = settings['front_plane_z']
        v0 = Vec3(-plane_x, plane_y, front_plane_z)
        uv0 = (0,0)
        v1 = Vec3(-plane_x ,plane_y,back_plane_z)
        uv1 = (0, 1)
        v2 = Vec3(plane_x, plane_y, front_plane_z)
        uv2 = (1, 0)
        v3 = Vec3(plane_x, plane_y,back_plane_z)
        uv3 = (1, 1)
        triangle = Triangle(v0, v1, v2, ground, uv0, uv1, uv2)
        world.add(triangle)

        triangle = Triangle(v1, v2, v3, ground, uv1, uv2, uv3)
        world.add(triangle)

        height = 2 * plane_x

        if settings['show_walls'] is True:
            # right wall
            v0 = Vec3(plane_x, plane_y, front_plane_z)
            uv0 = (0, 0)
            v1 = Vec3(plane_x, plane_y, back_plane_z)
            uv1 = (0, 1)
            v2 = Vec3(plane_x, plane_y+height, back_plane_z)
            uv2 = (1, 0)
            v3 = Vec3(plane_x, plane_y+height, front_plane_z)
            uv3 = (1, 1)
            triangle = Triangle(v0, v1, v2, right_wall, uv0, uv1, uv2)
            world.add(triangle)

            triangle = Triangle(v0, v2, v3, right_wall, uv1, uv2, uv3)
            world.add(triangle)

            # left wall
            v0 = Vec3(-plane_x, plane_y, front_plane_z)
            uv0 = (0, 0)
            v1 = Vec3(-plane_x, plane_y, back_plane_z)
            uv1 = (0, 1)
            v2 = Vec3(-plane_x, plane_y + height, back_plane_z)
            uv2 = (1, 0)
            v3 = Vec3(-plane_x, plane_y + height, front_plane_z)
            uv3 = (1, 1)
            triangle = Triangle(v0, v1, v2, left_wall, uv0, uv1, uv2)
            world.add(triangle)

            triangle = Triangle(v0, v2, v3, left_wall, uv1, uv2, uv3)
            world.add(triangle)

            # back wall
            v0 = Vec3(-plane_x, plane_y, back_plane_z)
            uv0 = (0, 0)
            v1 = Vec3(-plane_x, plane_y + height, back_plane_z)
            uv1 = (0, 1)
            v2 = Vec3(plane_x, plane_y + height, back_plane_z)
            uv2 = (1, 0)
            v3 = Vec3(plane_x, plane_y, back_plane_z)
            uv3 = (1, 1)
            triangle = Triangle(v0, v1, v2, back_wall, uv0, uv1, uv2)
            world.add(triangle)

            triangle = Triangle(v0, v2, v3, back_wall, uv1, uv2, uv3)
            world.add(triangle)

    my_mesh = mesh.Mesh.from_file(stl_filename)

    if 'rot_axis' in settings and settings['rot_axis'] is not None:
        rot_axis = settings['rot_axis']
        rot_rads = settings['rot_rads']
        my_mesh.rotate(rot_axis, rot_rads)

    # stl_mesh = STLMesh(my_mesh, metal_1, name="mesh_1")
    stl_mesh = STLMesh(my_mesh, object_matl, name="mesh_1")
    print(f'stl_mesh {stl_filename} -- bbox={stl_mesh.bounding_box(None, None)}, num_triangles={stl_mesh.num_triangles}')
    world.add(stl_mesh)

    scene = Scene(world)

    # camera = Camera(look_from=Vec3(8.5, 4, 0), look_at=Vec3(0, 1, 0), vup=Vec3(0, 1, 0), vert_fov=25)
    # camera = Camera(look_from=Vec3(12, 3, 0), look_at=Vec3(0, 0.5, 0), vup=Vec3(0, 1, 0), vert_fov=25)
    look_from = settings['look_from']
    look_at = settings['look_at']
    camera = Camera(look_from=look_from, look_at=look_at, vup=Vec3(0, 1, 0), vert_fov=25)

    return {'scene': scene, 'camera': camera}