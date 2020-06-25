"""
Scene creation functions

Listing 60 from Pete Shirley's Ray Tracing in a Weekend:

https://raytracing.github.io/books/RayTracingInOneWeekend.html

Len Wanger -- 2020
"""

from random import random, uniform
from geometry_classes import Vec3, GeometryList, Scene
from geometry_classes import random_on_unit_sphere
from material_classes import Lambertian, Metal, Dielectric
from primitives_classes import Sphere, Plane, Triangle


def create_simple_world():
    diffuse_1 = Lambertian(Vec3(0.7, 0.3, 0.3), name="diffuse_1")
    diffuse_2 = Lambertian(Vec3(0.8, 0.8, 0), name="diffuse_2")
    metal_1 = Metal(Vec3(0.8,0.6,0.2), fuzziness=0.3, name="metal_1")
    dielectric_1 = Dielectric(1.5, name="dielectric_1")

    world = GeometryList()
    world.add(Sphere(Vec3(0,0,-1), 0.5, diffuse_1))
    world.add(Sphere(Vec3(0,-100.5,-1), 100, diffuse_2))
    world.add(Sphere(Vec3(1,0,-1), 0.5, metal_1))
    world.add(Sphere(Vec3(-1,0,-1),0.5, dielectric_1))
    world.add(Sphere(Vec3(-1,0,-1),-0.45, dielectric_1))  # hollow sphere
    # return world
    return Scene(world)

def create_simple_world_2():
    # use a plane instead of a big sphere!
    # diffuse_1 = Lambertian(Vec3(0.7, 0.3, 0.3), name="diffuse_1")
    diffuse_2 = Lambertian(Vec3(0.8, 0.8, 0), name="diffuse_2")
    diffuse_3 = Lambertian(Vec3(0.2, 0.2, 0.7), name="diffuse_3")
    metal_1 = Metal(Vec3(0.8,0.6,0.2), fuzziness=0.3, name="metal_1")
    metal_2 = Metal(Vec3(0.4,0.4,0.4), fuzziness=0.0, name="metal_2")
    # dielectric_1 = Dielectric(1.5, name="dielectric_1")

    world = GeometryList()

    world.add(Sphere(Vec3(0,0,-1), 1.5, metal_1))

    plane_1 = Plane.plane_from_point_and_normal(pt=Vec3(0,-3,0), normal=Vec3(0,1,0), material=diffuse_3)
    plane_2 = Plane.plane_from_point_and_normal(pt=Vec3(0,0,-10), normal=Vec3(0,0,1), material=metal_2)
    # plane_3 = Plane.plane_from_point_and_normal(pt=Vec3(0,5,0), normal=Vec3(0.3,-1,0), material=diffuse_2)

    world.add(plane_1)
    world.add(plane_2)
    # world.add(plane_3)

    return Scene(world)



def create_simple_world_3():
    # add triangles
    diffuse_1 = Lambertian(Vec3(0.7, 0.3, 0.3))
    diffuse_2 = Lambertian(Vec3(0.8, 0.8, 0))
    diffuse_3 = Lambertian(Vec3(0.2, 0.2, 0.7))
    metal_1 = Metal(Vec3(0.8,0.6,0.2), fuzziness=0.3)
    metal_2 = Metal(Vec3(0.4,0.4,0.4), fuzziness=0.0)
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
    # world.add(Triangle(v0, v1, v2, diffuse_1))

    plane_1 = Plane.plane_from_point_and_normal(pt=Vec3(0,-3,0), normal=Vec3(0,1,0), material=diffuse_3)

    world.add(plane_1)

    return Scene(world)


def create_random_world():
    world = GeometryList()
    ground_material = Lambertian(Vec3(0.5,0.5,0.5))
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
                    albedo = Vec3(r,g,b)
                    sphere_material = Lambertian(albedo)
                elif choose_mat < 0.95:  # metal
                    albedo = uniform(0.5, 1.0)
                    fuzz = uniform(0.0, 0.5)
                    sphere_material = Metal(Vec3(albedo, albedo, albedo), fuzz)
                else:  # glass
                    sphere_material = glass_material

                world.add(Sphere(center, 0.2, sphere_material))

    material_1 = Dielectric(1.5)
    world.add(Sphere(Vec3(0,1,0), 1.0, material_1))
    material_2 = Lambertian(Vec3(0.4, 0.2, 0.1))
    world.add(Sphere(Vec3(-4, 1, 0), 1.0, material_2))
    material_3 = Metal(Vec3(0.7,0.6,0.5), 0.0)
    world.add(Sphere(Vec3(4, 1, 0), 1.0, material_3))
    return Scene(world)


def create_random_world2():
    def random_material():
        choose_mat = random()

        if choose_mat < 0.8:  # diffuse
            r = random() * random()
            g = random() * random()
            b = random() * random()
            albedo = Vec3(r, g, b)
            material = Lambertian(albedo)
        elif choose_mat < 0.95:  # metal
            albedo = uniform(0.5, 1.0)
            fuzz = uniform(0.0, 0.5)
            material = Metal(Vec3(albedo, albedo, albedo), fuzz)
        else:  # glass
            material = glass_material

        return material

    # a ground plane, a metal sphere and random triangles...
    world = GeometryList()

    ground_material = Lambertian(Vec3(0.2,0.6,0.2))
    metal_1 = Metal(Vec3(0.7,0.6,0.5), fuzziness=0.0)
    metal_2 = Metal(Vec3(0.4,0.4,0.4), fuzziness=0.3)
    glass_material = Dielectric(1.5)
    center_offset = Vec3(4, 0.2, 9)

    plane_1 = Plane.plane_from_point_and_normal(pt=Vec3(0, -3, 0), normal=Vec3(0, 1, 0), material=ground_material)
    world.add(plane_1)

    world.add(Sphere(Vec3(0,0,-1), 1.5, metal_1))

    for a in range(-12, 12):
        for b in range(-12, 12):
            # center = Vec3(a+0.9*random(), 0.2, b+0.9*random())
            center = Vec3(a+0.9*random(), 3*random()+0.3, b+0.9*random())

            if (center - center_offset).length() > 0.9:
                material = random_material()
                v0 = random_on_unit_sphere().mul_val(0.7) + center
                v1 = random_on_unit_sphere().mul_val(0.7) + center
                v2 = random_on_unit_sphere().mul_val(0.7) + center
                triangle = Triangle(v0,v1,v2, material)
                world.add(triangle)

    return Scene(world)