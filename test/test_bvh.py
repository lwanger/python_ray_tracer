"""
Test BVHNode building and hits
"""

import math

from geometry_classes import Vec3, GeometryList, Ray
from primitives_classes import Sphere, Plane, Triangle
from material_classes import Lambertian
from scene import Scene


if __name__ == '__main__':
    diffuse_1 = Lambertian(Vec3(0.5,0.5,0.5), "diffuse_1")
    ray = Ray(origin=Vec3(-1.8, 0., 3.), direction=Vec3(0., 0., -1.))
    ray2 = Ray(origin=Vec3(0., 0., 3.), direction=Vec3(0., 0., -1.))
    g_list = GeometryList()

    try:
        scene = Scene(g_list)
    except RuntimeError:
        pass

    g_list.add( Sphere(Vec3(0,0,0), 1.0, diffuse_1) )

    scene = Scene(g_list)
    v = scene.bvh.hit(ray)
    v = scene.bvh.hit(ray2)

    g_list.add( Sphere(Vec3(-2,-2,-2), 1.0, diffuse_1) )

    scene = Scene(g_list)
    v = scene.bvh.hit(ray)
    v = scene.bvh.hit(ray2)

    g_list.add( Sphere(Vec3(3,3,3), 1.0, diffuse_1) )

    plane = Plane.plane_from_point_and_normal(Vec3(0,0,0), Vec3(0,1,0), diffuse_1)
    g_list.add( plane )

    plane = Plane.plane_from_point_and_normal(Vec3(0, 0, 1), Vec3(3, 0, 0), diffuse_1)
    g_list.add(plane)

    triangle_1 = Triangle(Vec3(-3,0,0),Vec3(-2,1,0),Vec3(-1,0,0),diffuse_1)
    g_list.add( triangle_1 )

    triangle_2 = Triangle(Vec3(0, 0, 0), Vec3(1, 1, 0), Vec3(2, 0, 0), diffuse_1)
    g_list.add(triangle_1)

    triangle_3 = Triangle(Vec3(0.5, 0.5, 0.5), Vec3(1.5, 0.5, 0), Vec3(2.5, -0.5, -0.5), diffuse_1)
    g_list.add(triangle_1)

    scene = Scene(g_list)
    v = scene.bvh.hit(ray)
    v = scene.bvh.hit(ray2)

    # ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, 1))
    v = scene.bvh.hit(ray)
    v = scene.bvh.hit(ray2)

    # ray = Ray(origin=Vec3(-10, 0, 0), direction=Vec3(0, 0, 1))
    v = scene.bvh.hit(ray)
    # v = scene.bvh.hit(ray2)

    print('done')