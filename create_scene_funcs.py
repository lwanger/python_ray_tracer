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

import colorcet as cc
from stl import mesh  # numpy-stl

from geometry_classes import Vec3, GeometryList, Camera
from geometry_classes import random_on_unit_sphere, get_color
from material_classes import Lambertian, Metal, Dielectric
from texture_classes import SolidColor, CheckerBoard, ImageTexture, NoiseTexture
from primitives_classes import Sphere, Plane, Triangle, Disc, STLMesh
from light_classes import PointLight, AreaLight
from scene import Scene
from perlin import value_noise, turbulent_noise, fractal_noise, wood_pattern, marble_pattern


def create_simple_world(settings=None):
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
    world.add(Sphere(Vec3(1.25,0,-1), 0.5, metal_1))
    world.add(Sphere(Vec3(-1.25,0,-1),0.5, dielectric_1))
    world.add(Sphere(Vec3(-1.25,0,-1),-0.45, dielectric_1))  # hollow sphere

    ambient = Vec3(0.7, 0.7, 0.7)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))
    # light_1 = PointLight(pos=Vec3(0, 10, 0.35), color=Vec3(0.25, 0.25, 0.25))  # light directly above sphere
    light_1 = PointLight(pos=Vec3(-1, 10, 0.35), color=Vec3(0.25, 0.25, 0.25))
    # light_2 = PointLight(pos=Vec3(0, 10, 5.0), color=Vec3(0.1, 0.1, 0.25))  # blue light to the left
    lights = [light_1]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)

    #camera = Camera(look_from=Vec3(-0.5, 1, 5), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.1,
    camera = Camera(look_from=Vec3(-0.5, 1, 5), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.0,
                    focus_dist=20)
    return {'scene': scene, 'camera': camera}


def create_simple_world_2(settings=None):
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

    ambient = Vec3(0.6, 0.6, 0.6)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))
    light_1 = PointLight(pos=Vec3(-2, 10, 0.35), color=Vec3(0.25, 0.25, 0.25))  # light directly above sphere
    lights = [light_1]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)
    camera = Camera(look_from=Vec3(-0.5, 1, 10), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.0,
                    focus_dist=20)
    return {'scene': scene, 'camera': camera}

def create_simple_world_3(settings=None):
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

    ambient = Vec3(0.6, 0.6, 0.6)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))
    light_1 = PointLight(pos=Vec3(-2, 10, 0.35), color=Vec3(0.25, 0.25, 0.25))  # light directly above sphere
    lights = [light_1]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)
    # camera = Camera(look_from=Vec3(-0.5, 1, 13), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.1, focus_dist=20)
    camera = Camera(look_from=Vec3(1, 1, 13), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.0, focus_dist=20)
    return {'scene': scene, 'camera': camera}


def create_random_world(settings=None):
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

    ambient = Vec3(0.6, 0.6, 0.6)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))
    light_1 = PointLight(pos=Vec3(18,12,5), color=Vec3(0.25, 0.25, 0.25))  # light directly above sphere
    lights = [light_1]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)
    camera = Camera(look_from=Vec3(13, 2, 3), look_at=Vec3(0, 0, 0), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.1, focus_dist=20)
    return {'scene': scene, 'camera': camera}


def create_random_world2(settings=None):
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

    ambient = Vec3(0.6, 0.6, 0.6)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))
    light_1 = PointLight(pos=Vec3(18, 10, 5), color=Vec3(0.25, 0.25, 0.25))  # light directly above sphere
    lights = [light_1]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)
    camera = Camera(look_from=Vec3(13, 2, 3), look_at=Vec3(0, 0, 0), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.1, focus_dist=20)
    return {'scene': scene, 'camera': camera}


def create_checkerboard_world(settings=None):
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

    ambient = Vec3(0.6, 0.6, 0.6)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))
    light_1 = PointLight(pos=Vec3(-2, 10, 0.35), color=Vec3(0.25, 0.25, 0.25))  # light directly above sphere
    lights = [light_1]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)
    camera = Camera(look_from=Vec3(-0.5, 1, 5), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=30)
    return {'scene': scene, 'camera': camera}


def create_checkerboard_world_2(settings=None):
    odd_color = SolidColor(Vec3(0.2,0.3,0.1))
    even_color = SolidColor(Vec3(0.9,0.9,0.9))
    checker_board = CheckerBoard(even_color, odd_color, spacing=3)

    diffuse_1 = Lambertian(checker_board, name="diffuse_checkerboard")
    metal_1 = Metal(checker_board, fuzziness=0.1, name="metal_checkerboard")

    world = GeometryList()
    world.add(Sphere(Vec3(0,-10,0), 10, diffuse_1))
    world.add(Sphere(Vec3(0,10,0), 10, metal_1))

    ambient = Vec3(0.6, 0.6, 0.6)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))
    light_1 = PointLight(pos=Vec3(-2, 10, 0.35), color=Vec3(0.25, 0.25, 0.25))  # light directly above sphere
    lights = [light_1]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)
    camera = Camera(look_from=Vec3(-0.5, 1, 15), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=20)

    return {'scene': scene, 'camera': camera}


def create_image_texture_world(settings=None):
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

    ambient = Vec3(0.6, 0.6, 0.6)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))
    light_1 = PointLight(pos=Vec3(-2, 10, 0.35), color=Vec3(0.25, 0.25, 0.25))  # light directly above sphere
    lights = [light_1]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)
    camera = Camera(look_from=Vec3(-0.5, 1, 7), look_at=Vec3(0, 0, -0.5), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.1, focus_dist=20)

    return {'scene': scene, 'camera': camera}


def create_canonical_1(settings=None):
    # sphere over a checkerboard!
    silver = SolidColor(Vec3(0.7, 0.7, 0.7))

    # image_1 = Image.open(Path("./textures/earthlights_dmsp_big.jpg"))
    image_2 = Image.open(Path("./textures/IO_logo.png"))
    # image_texture_1 = ImageTexture(image_1, "earthlights")

    logo = ImageTexture(image_2)  # images are across the entire checkerboard, not a single square?

    # odd_color = ImageTexture(image_2)  # images are across the entire checkerboard, not a single square?
    odd_color = SolidColor(Vec3(0.2, 0.75, 0.2))
    even_color = SolidColor(Vec3(0.1, 0.1, 0.1))
    checker_board = CheckerBoard(even_color, odd_color, spacing=2.0)

    if True:  # use checkerboard vs image texture
        diffuse_2 = Lambertian(checker_board, name="checkerboard")
        # diffuse_2 = Lambertian(odd_color, name="odd_color")
    else:
        diffuse_2 = Lambertian(logo, name="io_logo'")

    metal_1 = Metal(silver, name="metal_1")

    world = GeometryList()

    world.add(Sphere(Vec3(0, 1.25, 0.35), 1.0, metal_1))
    # world.add(Sphere(Vec3(0, 1.0001, 0.35), 1.0, metal_1))

    if True:  # use plane vs triangles
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

    ambient = Vec3(0.5,0.5,0.5)
    # ambient = Vec3(0.3,0.3,0.3)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))
    # background = SolidColor(Vec3(0,0,0))
    # light_1 = PointLight(pos=Vec3(11,10,3), color=Vec3(0.25, 0.25, 0.25))

    geom = Disc(center=Vec3(11,10,3), normal=Vec3(0,-1,0), radius=1.5, material=SolidColor(Vec3(0.7, 0.7, 0.7)))
    if settings and 'SAMPLES_PER_LIGHT' in settings:
        samples = settings['SAMPLES_PER_LIGHT']
    else:
        samples = 25

        light_2 = AreaLight(geom=geom, color=Vec3(0.6, 0.6, 0.6), num_samples=samples)

    lights = [light_2]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)
    camera = Camera(look_from=Vec3(8.5, 4, 0), look_at=Vec3(0, 1, 0), vup=Vec3(0, 1, 0), vert_fov=25)

    return {'scene': scene, 'camera': camera}


def create_canonical_2(settings=None):
    """
    teapot time!

    TEAPOT_thingiverse.stl -- bbox=AABB(vmin=(-15.000, -10.005, -9.088), vmax=(16.371, 10.005, 7.162)), num_triangles=87298
    """
    spacing=0.5
    stl_filename = Path("models/TEAPOT_thingiverse.stl")
    rot_axis = [1, 0, 0]
    rot_rads = math.pi / 2.0  # 45 deg
    look_from = Vec3(0, 15, 60)
    look_at = Vec3(0, -1.5, 0)
    plane_y = -9.1
    plane_x = 25
    back_plane_z = -25
    front_plane_z = 15
    light1_pos = Vec3(11, 10, 3)

    silver = SolidColor(Vec3(0.7, 0.7, 0.7))
    light_gray = SolidColor(Vec3(0.85, 0.85, 0.85))

    odd_color = SolidColor(Vec3(0.2, 0.75, 0.2))
    even_color = SolidColor(Vec3(0.1, 0.1, 0.1))

    checker_board = CheckerBoard(even_color, odd_color, spacing=spacing)
    diffuse_1 = Lambertian(checker_board, name="checkerboard")

    # diffuse_2 = Lambertian(silver, name="silver_matte")
    diffuse_2 = Lambertian(light_gray, name="silver_matte")
    fuzz = 0.2
    metal_1 = Metal(silver, fuzziness=fuzz, name="chrome")

    world = GeometryList()

    my_mesh = mesh.Mesh.from_file(stl_filename)
    my_mesh.rotate(rot_axis, rot_rads)

    # teapot_matl = metal_1
    teapot_matl = diffuse_2

    stl_mesh = STLMesh(my_mesh, teapot_matl, name="teapot")
    print(f'stl_mesh {stl_filename} -- bbox={stl_mesh.bounding_box(None, None)}, num_triangles={stl_mesh.num_triangles}')
    world.add(stl_mesh)

    if False:  # use plane vs triangles
        plane_1 = Plane.plane_from_point_and_normal(pt=Vec3(0, plane_y, 0), normal=Vec3(0, 1, 0), material=diffuse_1)
        world.add(plane_1)
    else:
        v0 = Vec3(-plane_x, plane_y, front_plane_z)
        uv0 = (0,0)
        v1 = Vec3(-plane_x ,plane_y,back_plane_z)
        uv1 = (0, 1)
        v2 = Vec3(plane_x, plane_y, front_plane_z)
        uv2 = (1, 0)
        v3 = Vec3(plane_x, plane_y,back_plane_z)
        uv3 = (1, 1)
        triangle = Triangle(v0, v1, v2, diffuse_1, uv0, uv1, uv2)
        world.add(triangle)

        triangle = Triangle(v1, v2, v3, diffuse_1, uv1, uv2, uv3)
        world.add(triangle)

    # ambient = Vec3(0.6,0.6,0.6)
    ambient = Vec3(0.5,0.5,0.5)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))
    # light_1 = PointLight(pos=light1_pos, color=Vec3(0.35, 0.35, 0.35))
    light_1 = PointLight(pos=light1_pos, color=Vec3(0.5, 0.5, 0.5))
    lights = [light_1]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)
    camera = Camera(look_from=look_from, look_at=look_at, vup=Vec3(0, 1, 0), vert_fov=25)

    return {'scene': scene, 'camera': camera}


def create_perlin_1(settings=None):
    # sphere over a plane!
    # dragon over a plane!

    green = SolidColor(Vec3(0.2,0.7,0.2))
    brown = SolidColor(Vec3(0.7,0.5,0.3))

    # point_scale = 1.0
    # wood_point_scale = 100.0
    wood_point_scale = 20.0

    wood, wood_name = (cc.CET_D6[155:240], 'wood3')
    wood_colormap = [get_color(i, wood) for i in range(len(wood))]
    kwargs = {'frequency': 0.01, 'frequency_mult': 10, }
    translate = 1.0
    scale = 0.5
    wood_texture = NoiseTexture(wood_colormap, point_scale=wood_point_scale, translate=translate, scale=scale,
                              name=wood_name, eval_func=wood_pattern, eval_kwargs=kwargs)

    jade, jade_name = (cc.CET_D13[135:240], 'jade2')
    jade_colormap = [get_color(i, jade) for i in range(len(jade))]
    kwargs = {'frequency': 0.024, 'frequency_mult': 2.5, 'amplitude_mult': 0.5, 'layers': 7, 'displace_x': 200}
    translate = 0.20
    scale = 1.0
    # jade_point_scale = 600.0
    jade_point_scale = 6.0
    jade_texture = NoiseTexture(jade_colormap, point_scale=jade_point_scale, translate=translate, scale=scale,
                              name=jade_name, eval_func=marble_pattern, eval_kwargs=kwargs)

    # diffuse_1 = Lambertian(wood_texture, name="wood'")
    diffuse_2 = Lambertian(jade_texture, name="jade")
    diffuse_3 = Lambertian(green, name="solid green")
    diffuse_4 = Lambertian(brown, name="solid brown")

    metal_1 = Metal(wood_texture, name="shiny wood", fuzziness=0.2)
    metal_2 = Metal(jade_texture, name="metal_1", fuzziness=0.3)

    ground_matl = metal_1
    # ground_matl = diffuse_4

    object_matl = diffuse_2
    # object_matl = metal_2
    # object_matl = diffuse_3

    world = GeometryList()

    if False:
    # if True:
        world.add(Sphere(Vec3(0, 0.0, 0.0), 8.0, object_matl))
        settings = {'look_from': Vec3(0.0, 10, 40), 'look_at': Vec3(0, 0.25, 0),
                    'plane_x': 24, 'plane_y': -8.0, 'back_plane_z': -25, 'front_plane_z': 20,
                    'rot_axis': [1, 0, 0], 'rot_rads': math.pi / 2, 'translate': [0, 0, -12.5], 'show_walls': True}
    else:
        stl_filename = "models/dragon_65.stl"
        settings = {'look_from': Vec3(0.0, 10, 40), 'look_at': Vec3(0, 0.25, 0),
        'plane_x': 24, 'plane_y': -7.221, 'back_plane_z': -25, 'front_plane_z': 20,
        'rot_axis': [1,0,0], 'rot_rads': math.pi/2, 'translate': [0, 0, -12.5], 'show_walls': True}

        # if False:
        if True:
            my_mesh = mesh.Mesh.from_file(stl_filename)

            if 'translate' in settings:
                settings['translate'][0]
                my_mesh.translate([settings['translate'][0], settings['translate'][1], settings['translate'][2]])

            if 'rot_axis' in settings and settings['rot_axis'] is not None:
                rot_axis = settings['rot_axis']
                rot_rads = settings['rot_rads']
                my_mesh.rotate(rot_axis, rot_rads)

            stl_mesh = STLMesh(my_mesh, object_matl, name="mesh_1")
            print(f'stl_mesh {stl_filename} -- bbox={stl_mesh.bounding_box(None, None)}, num_triangles={stl_mesh.num_triangles}')
            world.add(stl_mesh)

    if True:
        # if True:  # use plane vs triangles
        if False:
            plane_1 = Plane.plane_from_point_and_normal(pt=Vec3(0, -1, 0), normal=Vec3(0, 1, 0), material=ground_matl)
            world.add(plane_1)
        else:
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
            triangle = Triangle(v0, v1, v2, ground_matl, uv0, uv1, uv2)
            world.add(triangle)

            triangle = Triangle(v1, v2, v3, ground_matl, uv1, uv2, uv3)
            world.add(triangle)

    ambient = Vec3(0.6,0.6,0.6)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))

    geom = Disc(center=Vec3(3,10,-3), normal=Vec3(0,-1,0), radius=1.5, material=SolidColor(Vec3(0.7, 0.7, 0.7)))
    if settings and 'SAMPLES_PER_LIGHT' in settings:
        samples = settings['SAMPLES_PER_LIGHT']
    else:
        samples = 25

        # light_1 = PointLight(pos=Vec3(-10.0, 100, 80), color=Vec3(0.2, 0.3, 0.2))
        light_1 = PointLight(pos=Vec3(-10.0, 100, 80), color=Vec3(0.6, 0.6, 0.6))
        light_2 = AreaLight(geom=geom, color=Vec3(0.6, 0.6, 0.6), num_samples=samples)

    # lights = [light_1]
    lights = [light_2]
    # lights = [light_1, light_2]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)
    # camera = Camera(look_from=Vec3(8.5, 4, 0), look_at=Vec3(0, 1, 0), vup=Vec3(0, 1, 0), vert_fov=25)
    camera = Camera(look_from=settings['look_from'], look_at=settings['look_at'], vup=Vec3(0, 1, 0), vert_fov=25)

    return {'scene': scene, 'camera': camera}



def create_stl_mesh(settings=None):
    """
    TODO:
        scale the model
    """
    silver = SolidColor(Vec3(0.7, 0.7, 0.7))
    green = SolidColor(Vec3(0.1, 0.5, 0.1))
    blue = SolidColor(Vec3(0.1, 0.1, 0.5))
    red = SolidColor(Vec3(0.5, 0.2, 0.2))
    purple = SolidColor(Vec3(0.4, 0.1, 0.4))
    gray = SolidColor(Vec3(0.2, 0.2, 0.2))
    med_gray = SolidColor(Vec3(0.4, 0.4, 0.4))
    light_gray = SolidColor(Vec3(0.9, 0.9, 0.9))
    dark_gray = SolidColor(Vec3(0.1, 0.1, 0.1))
    black = SolidColor(Vec3(0.0, 0.0, 0.0))

    # rotated by 90 deg on the X axis... bbox=AABB(vmin=(-48.551, 5.275, -45.792), vmax=(59.196, 113.167, 42.010)), num_triangles=112402
    stl_filename = "models/Bunny.stl"
    settings = {'look_from': Vec3(0.0, 100, 350), 'look_at': Vec3(0, 50.0, 0),
    'plane_x': 120, 'plane_y': 5.275, 'back_plane_z': -85, 'front_plane_z': 350,
    'rot_axis': [1,0,0], 'rot_rads': math.pi/2, 'translate': [-25, 0, 0], 'show_walls': True}

    image_2 = Image.open(Path("./textures/IO_logo.png"))
    logo = ImageTexture(image_2)  # images are across the entire checkerboard, not a single square?

    checked = CheckerBoard(dark_gray, light_gray, spacing=0.1)

    diffuse_red = Lambertian(red, name="red'")
    diffuse_blue = Lambertian(blue, name="blue'")
    diffuse_gray = Lambertian(gray, name="gray'")
    diffuse_med_gray = Lambertian(med_gray, name="med_gray'")
    diffuse_light_gray = Lambertian(light_gray, name="light_gray'")
    metal_1 = Metal(silver, name="metal_1")
    logo_matl = Lambertian(logo, name="logo")
    # dielectric_1 = Dielectric(1.5, name="dielectric_1")
    checkerboard = Lambertian(checked, name="gray'")
    dielectric = Dielectric(1.0, "dielectric")

    # object_matl = metal_1
    # object_matl = diffuse_gray
    object_matl = diffuse_light_gray
    # object_matl = dielectric
    ground_matl = diffuse_gray
    # ground_matl = checkerboard
    # ground_matl = logo_matl
    # right_wall_matl = diffuse_red
    # right_wall_matl = metal_1
    right_wall_matl = diffuse_light_gray
    # left_wall_matl = diffuse_blue
    # left_wall_matl = metal_1
    left_wall_matl = diffuse_light_gray
    # back_wall_matl = logo_matl

    # back_wall_matl = metal_1
    back_wall_matl = checkerboard

    world = GeometryList()

    if True:
        plane_x = settings['plane_x']
        plane_y = settings['plane_y']
        back_plane_z = settings['back_plane_z']
        front_plane_z = settings['front_plane_z']

        if True:
            # ground plane
            v0 = Vec3(-plane_x, plane_y, front_plane_z)
            uv0 = (0,1)
            v1 = Vec3(-plane_x ,plane_y,back_plane_z)
            uv1 = (0,0)
            v2 = Vec3(plane_x, plane_y, front_plane_z)
            uv2 = (1,1)
            v3 = Vec3(plane_x, plane_y,back_plane_z)
            uv3 = (1,0)
            triangle = Triangle(v0, v1, v2, ground_matl, uv0, uv1, uv2)
            world.add(triangle)
            triangle = Triangle(v1, v2, v3, ground_matl, uv1, uv2, uv3)
            world.add(triangle)

        height = 2 * plane_x

        if settings['show_walls'] is True:
            # right wall
            v0 = Vec3(plane_x, plane_y, front_plane_z)
            uv0 = (1, 1)
            v1 = Vec3(plane_x, plane_y, back_plane_z)
            uv1 = (0, 1)
            v2 = Vec3(plane_x, plane_y+height, back_plane_z)
            uv2 = (0,0)
            v3 = Vec3(plane_x, plane_y+height, front_plane_z)
            uv3 = (1, 0)
            triangle = Triangle(v0, v1, v2, right_wall_matl, uv0, uv1, uv2)
            world.add(triangle)
            triangle = Triangle(v0, v2, v3, right_wall_matl, uv0, uv2, uv3)
            world.add(triangle)

            # left wall
            v0 = Vec3(-plane_x, plane_y, front_plane_z)
            uv0 = (0, 1)
            v1 = Vec3(-plane_x, plane_y, back_plane_z)
            uv1 = (1, 1)
            v2 = Vec3(-plane_x, plane_y + height, back_plane_z)
            uv2 = (1, 0)
            v3 = Vec3(-plane_x, plane_y + height, front_plane_z)
            uv3 = (0, 0)
            triangle = Triangle(v0, v1, v2, left_wall_matl, uv0, uv1, uv2)
            world.add(triangle)
            triangle = Triangle(v0, v2, v3, left_wall_matl, uv0, uv2, uv3)
            world.add(triangle)

            # back wall
            v0 = Vec3(-plane_x, plane_y, back_plane_z)
            uv0 = (0, 1)
            v1 = Vec3(-plane_x, plane_y + height, back_plane_z)
            uv1 = (0, 0)
            v2 = Vec3(plane_x, plane_y + height, back_plane_z)
            uv2 = (1, 0)
            v3 = Vec3(plane_x, plane_y, back_plane_z)
            uv3 = (1, 1)
            triangle = Triangle(v0, v1, v2, back_wall_matl, uv0, uv1, uv2)
            world.add(triangle)
            triangle = Triangle(v0, v2, v3, back_wall_matl, uv0, uv2, uv3)
            world.add(triangle)

    my_mesh = mesh.Mesh.from_file(stl_filename)

    if 'translate' in settings:
        settings['translate'][0]
        my_mesh.translate([settings['translate'][0], settings['translate'][1], settings['translate'][2]])

    if 'rot_axis' in settings and settings['rot_axis'] is not None:
        rot_axis = settings['rot_axis']
        rot_rads = settings['rot_rads']
        my_mesh.rotate(rot_axis, rot_rads)

    stl_mesh = STLMesh(my_mesh, object_matl, name="mesh_1")
    print(f'stl_mesh {stl_filename} -- bbox={stl_mesh.bounding_box(None, None)}, num_triangles={stl_mesh.num_triangles}')
    world.add(stl_mesh)

    ambient = Vec3(0.5, 0.5, 0.5)
    # ambient = Vec3(0.3, 0.3, 0.3)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))
    light_1 = PointLight(pos=Vec3(-10.0, 100, 80), color=Vec3(0.2, 0.2, 0.2))
    # disc_1 = Disc(center=Vec3(-15,110,20), normal=Vec3(-1,-20,-0.5), radius=5.0, material=diffuse_gray)
    # disc_1 = Disc(center=Vec3(-15,110,20), normal=Vec3(0,-1,0), radius=15.0, material=diffuse_gray)
    disc_1 = Disc(center=Vec3(-30,130,25), normal=Vec3(0.15, -1, 0.15), radius=8.0, material=diffuse_med_gray)
    light_2 = AreaLight(geom=disc_1, color=Vec3(0.5, 0.5, 0.5))
    # lights = [light_1]
    lights = [light_2]
    # lights = [light_1, light_2]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)

    # camera = Camera(look_from=Vec3(8.5, 4, 0), look_at=Vec3(0, 1, 0), vup=Vec3(0, 1, 0), vert_fov=25)
    # camera = Camera(look_from=Vec3(12, 3, 0), look_at=Vec3(0, 0.5, 0), vup=Vec3(0, 1, 0), vert_fov=25)
    look_from = settings['look_from']
    look_at = settings['look_at']
    camera = Camera(look_from=look_from, look_at=look_at, vup=Vec3(0, 1, 0), vert_fov=25)

    return {'scene': scene, 'camera': camera}


def create_quad_world(settings=None):
    color_1 = SolidColor(Vec3(0.7, 0.3, 0.3))
    color_2 = SolidColor(Vec3(0.8, 0.8, 0))
    color_3 = SolidColor(Vec3(0.8,0.6,0.2))

    image_1 = Image.open(Path("./textures/earthlights_dmsp_big.jpg"))
    image_texture_1 = ImageTexture(image_1, "earthlights")
    texture_1 = Lambertian(image_texture_1, name="texture_1")

    diffuse_1 = Lambertian(color_1, name="diffuse_1")
    diffuse_2 = Lambertian(color_2, name="diffuse_2")
    diffuse_3 = Lambertian(color_3, name="diffuse_3")
    # metal_1 = Metal(color_3, fuzziness=0.3, name="metal_1")
    # dielectric_1 = Dielectric(1.5, name="dielectric_1")

    world = GeometryList()

    v0 = Vec3(-1,0,0)
    v1 = Vec3(-1,1,0)
    v2 = Vec3(1,1,0)
    # world.add(Quad(v0,v1,v2, diffuse_1))  # XY
    world.add(Quad(v0,v1,v2, texture_1))  # XY

    v0 = Vec3(-1, 0, -1)  # XZ
    v1 = Vec3(-1, 0, -2)
    v2 = Vec3(1, 0, -1)
    world.add(Quad(v0, v1, v2, diffuse_2))

    v0 = Vec3(-0.75, 3, -0.75)
    v1 = Vec3(-1, 2, -0.35)
    v2 = Vec3(0.3, 2.75, -0.5)
    world.add(Quad(v0, v1, v2, diffuse_3))

    ambient = Vec3(0.7, 0.7, 0.7)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))
    light_1 = PointLight(pos=Vec3(-1, 10, 0.35), color=Vec3(0.25, 0.25, 0.25))
    lights = [light_1]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)
    camera = Camera(look_from=Vec3(1.5, 3, 5), look_at=Vec3(0, 0, -1), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.0,
    # camera = Camera(look_from=Vec3(0, 30, 0), look_at=Vec3(0, 0, 0), vup=Vec3(-1, 0, 0), vert_fov=20, aperature=0.0,
                    focus_dist=20)
    return {'scene': scene, 'camera': camera}


def create_disc_test_world(settings=None):
    color_1 = SolidColor(Vec3(0.7, 0.3, 0.3))
    color_2 = SolidColor(Vec3(0.8, 0.8, 0))
    color_3 = SolidColor(Vec3(0.8,0.6,0.2))

    # image_1 = Image.open(Path("./textures/earthlights_dmsp_big.jpg"))
    image_1 = Image.open(Path("./textures/george harrison (1 bit).bmp"))
    image_texture_1 = ImageTexture(image_1, "earthlights")

    diffuse_1 = Lambertian(color_1, name="diffuse_1")
    diffuse_2 = Lambertian(color_2, name="diffuse_2")
    diffuse_3 = Lambertian(image_texture_1, name="diffuse_3")
    metal_1 = Metal(color_3, fuzziness=0.0, name="metal_1")
    dielectric_1 = Dielectric(1.5, name="dielectric_1")

    world = GeometryList()

    world.add(Disc(center=Vec3(-1.5,1.5,0), normal=Vec3(0,0,1), radius=0.5, material=diffuse_1))
    world.add(Disc(center=Vec3(1.5,1.5,0), normal=Vec3(0,0,1), radius=1.0, material=diffuse_2))
    world.add(Disc(center=Vec3(0,1.0,0), normal=Vec3(0,-1,1), radius=0.75, material=diffuse_3))
    world.add(Disc(center=Vec3(0,0,0), normal=Vec3(0,1,0), radius=5.0, material=metal_1))

    ambient = Vec3(0.7, 0.7, 0.7)
    background = SolidColor(Vec3(0.5, 0.7, 1.0))
    light_1 = PointLight(pos=Vec3(-1, 10, 0.35), color=Vec3(0.25, 0.25, 0.25))
    lights = [light_1]
    scene = Scene(world, ambient=ambient, lights=lights, background=background)
    camera = Camera(look_from=Vec3(-0.5, 3, 10), look_at=Vec3(0, 1.0, 0), vup=Vec3(0, 1, 0), vert_fov=20, aperature=0.0, focus_dist=20)
    return {'scene': scene, 'camera': camera}
