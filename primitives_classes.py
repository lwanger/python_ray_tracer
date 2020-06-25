"""
Primitive Classes

based on Pete Shirley's Ray Tracing in a Weekend (https://raytracing.github.io/books/RayTracingInOneWeekend.html)
Len Wanger, Copyright 2020

removed isinstance calls from Vec3 dunder methods (was very slow!)

"""

import math
from typing import Optional

from geometry_classes import Vec3, Ray, Geometry, AABB, dot, cross
from material_classes import Material

EPSILON = 1e-15
NEG_EPSILON = -1e-15


def surrounding_box(box1: "AABB", box2: "AABB") -> "AABB":
    # return the bounding box surrounding two bounding boxes (i.e. the union)
    if box1 is None:
        return box2
    if box2 is None:
        return box1

    b1_min = box1.vmin
    b2_min = box2.vmin
    b1_max = box1.vmax
    b2_max = box2.vmax

    new_x_min = min(b1_min.x, b2_min.x)
    new_y_min = min(b1_min.y, b2_min.y)
    new_z_min = min(b1_min.z, b2_min.z)
    new_min = Vec3(new_x_min, new_y_min, new_z_min)

    new_x_max = max(b1_max.x, b2_max.x)
    new_y_max = max(b1_max.y, b2_max.y)
    new_z_max = max(b1_max.z, b2_max.z)
    new_max = Vec3(new_x_max, new_y_max, new_z_max)

    return AABB(new_min, new_max)


class HitRecord():
    # object passed back from a ray intersection
    def __init__(self, point: Vec3, normal: Vec3, t: float, material: Material):
        self.point = point
        self.normal = normal
        self.t = t
        self.material = material
        self.front_face = None

    def __repr__(self):
        return f'HitRecord(point={self.point}, normal={self.normal}, t={self.t}, ...)'

    def set_face_normal(self, ray: Ray, outward_normal: Vec3):
        dp = dot(ray.direction, outward_normal)
        if dp < 0:
            self.front_face = True
            self.normal = outward_normal
        else:
            self.front_face = False
            self.normal = -outward_normal


class Sphere(Geometry):
    def __init__(self, center: Vec3, radius: float, material: Material):
        super().__init__(material)
        self.center = center
        self.radius = radius

    def __repr__(self):
        return f'Sphere(center={self.center}, radius={self.radius}, material={self.material})'

    def has_bbox(self) -> bool:
        return True

    def bounding_box(self, t0: float, t1: float) -> AABB:
        c = self.center
        r = self.radius
        vmin = c - Vec3(r,r,r)
        vmax = c + Vec3(r,r,r)
        aabb = AABB(vmin, vmax)
        return aabb

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        hr = None
        oc = ray.origin - self.center
        a = ray.direction.squared_length()
        half_b = dot(oc, ray.direction)
        c = oc.squared_length() - self.radius ** 2
        discriminant = half_b ** 2 - a * c

        if discriminant > 0:
            root = math.sqrt(discriminant)
            t = (-half_b - root) / a
            p = None

            if t_min < t < t_max:
                p = ray.at(t)
                # n = (p - self.center) / self.radius
                n = (p - self.center).div_val(self.radius)
            else:
                t = (-half_b + root) / a

                if t_min < t < t_max:
                    p = ray.at(t)
                    # n = (p - self.center) / self.radius
                    n = (p - self.center).div_val(self.radius)

            if p is not None:
                hr = HitRecord(point=p, normal=n, t=t, material=self.material)
                # outward_normal = (hr.point - self.center) / self.radius
                outward_normal = (hr.point - self.center).div_val(self.radius)
                hr.set_face_normal(ray, outward_normal)

        return hr


class Plane(Geometry):
    def __init__(self, a: float, b: float, c: float, d: float, material: Material):
        # Ax + By + cz +d = 0
        super().__init__(material)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.normal = Vec3(a,b,c).unit_vector()
        self.inverse_normal = -self.normal

    def __repr__(self):
        return f'Plane(a={self.a}, b={self.b}, c={self.c}, d={self.d},  material={self.material})'

    def has_bbox(self) -> bool:
        return False

    def bounding_box(self, t0: float, t1: float) -> AABB:
        return None

    @classmethod
    def plane_from_three_points(cls, a: Vec3, b: Vec3, c: Vec3, material: Material) -> "Plane":
        ab = b - a
        ac = c - b
        normal = cross(ab, ac)
        d = dot(normal, -a)
        return cls(normal.x, normal.y, normal.z, d, material)

    @classmethod
    def plane_from_point_and_normal(cls, pt: Vec3, normal: Vec3, material: Material) -> "Plane":
        d = dot(normal, -pt)
        return cls(normal.x, normal.y, normal.z, d, material)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        hr = None
        vd = dot(self.normal, ray.direction)

        # if vd == 0:  # ray is parallel to the plane -- no hit
        if abs(vd) < EPSILON:  # ray is parallel to the plane -- no hit
            return hr
        elif vd > 0:  # normal is pointing away from the plane -- no hit for 1-sided plane
            return hr

        vo = -(dot(self.normal, ray.origin) + self.d)
        t = vo/vd

        if t < 0:  # intersection behind origin
            return hr

        ri = ray.origin + ray.direction.mul_val(t)

        if vd < 0:
            rn = self.normal
        else:
            rn = self.inverse_normal

        if t_min < t < t_max:
            hr = HitRecord(ri, rn, t, self.material)
            hr.set_face_normal(ray, rn)

        return hr


class Triangle(Geometry):
    def __init__(self, v0: Vec3, v1: Vec3, v2: Vec3, material: Material):
        # Ax + By + cz +d = 0
        super().__init__(material)
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        self.normal = cross(v0v1, v0v2).normalize()
        self.inverse_normal = -self.normal

    def __repr__(self):
        return f'Triangle(v0={self.v0}, v1={self.v1}, v2={self.v2},  material={self.material})'

    def has_bbox(self) -> bool:
        return True

    def bounding_box(self, t0: float, t1: float) -> AABB:
        v0 = self.v0
        v1 = self.v1
        v2 = self.v2

        x_min = min(v0.x, v1.x, v2.x)
        x_max = max(v0.x, v1.x, v2.x)
        y_min = min(v0.y, v1.y, v2.y)
        y_max = max(v0.y, v1.y, v2.y)
        z_min = min(v0.z, v1.z, v2.z)
        z_max = max(v0.z, v1.z, v2.z)
        vmin = Vec3(x_min,y_min,z_min)
        vmax = Vec3(x_max,y_max,z_max)
        aabb = AABB(vmin, vmax)
        return aabb

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        # Moller Trumbore method -- https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        hr = None

        v0 = self.v0
        v1 = self.v1
        v2 = self.v2
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = cross(ray.direction, edge2)
        a = dot(edge1, h)

        if NEG_EPSILON < a < EPSILON:  # ray parallel to triangle
            return hr

        f = 1 / a
        s = ray.origin - v0
        u = f * dot(s, h)

        if u < 0 or u > 1:
            return hr

        q = cross(s, edge1)
        v = f * dot(ray.direction, q)

        if v < 0 or u + v > 1:
            return hr

        # point is in triangle... return t
        t = f * dot(edge2, q)

        if t < EPSILON:  # intersects with line, not ray
            return hr

        ri = ray.origin + ray.direction.mul_val(t)
        vd = dot(self.normal, ray.direction) # faster way? Already computed?

        if vd < 0:
            rn = self.normal
        else:
            rn = self.inverse_normal

        if t_min < t < t_max:
            hr = HitRecord(ri, rn, t, self.material)
            hr.set_face_normal(ray, rn)

        return hr


if __name__ == '__main__':
    v1 = Vec3(1.0, 2.0, 3.0)
    v2 = Vec3(4.0, 5.0, 6.0)
    ray1 = Ray(v1, v2, 0.0, 1.0)

    print(f'v1={v1}, repr={repr(v1)}')
    print(f'v2={v2}, repr={repr(v2)}')

    print(f'v2 x,y,z={v2.x},{v2.y},{v2.z}')
    print(f'v2 x,y,z={v2.r},{v2.g},{v2.b}')

    print(v2)
    v2.x = 4.0
    print(v2)

    print(f'-v1 = {-v1}')
    print(f'v1+v2 = {v1+v2}')
    print(f'v1-v2 = {v1-v2}')
    print(f'v1*3.0 = {v1*3.0}')
    print(f'v1/3.0 = {v1/3.0}')
    print(f'v1*v2 = {v1*v2}')
    print(f'v1/v2 = {v1/v2}')

    print(f'v1.length = {v1.length()}')
    print(f'v1.squared_length = {v1.squared_length()}')

    print(f'dot(v1,v2) = {dot(v1,v2)}')
    print(f'v1.normalize = {v1.normalize()}')
    print(f'v1.unit_vector = {v1.unit_vector()}')
    print(f'cross(v1,v2) = {cross(v1,v2)}')

    print(f'ray1={ray1}, repr={repr(ray1)}')
    print(f'ray1 repr={repr(ray1)}')
    print(f'ray1.at(0.0)={ray1.at(0.0)}')
    print(f'ray1.at(0.5)={ray1.at(0.5)}')
    print(f'ray1.at(1.0)={ray1.at(1.0)}')

