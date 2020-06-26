"""
Primitive Classes

based on Pete Shirley's Ray Tracing in a Weekend (https://raytracing.github.io/books/RayTracingInOneWeekend.html)
Len Wanger, Copyright 2020

removed isinstance calls from Vec3 dunder methods (was very slow!)

TODO:
    - add tiling pattern for plane textures (divmod on 1.0 and do different things on odd vs even? remainder is 0-1 val)

"""

import math
from typing import Optional

from geometry_classes import Vec3, Ray, Geometry, HitRecord, AABB, dot, cross
from material_classes import Material

EPSILON = 1e-15
NEG_EPSILON = -1e-15
TWO_PI = 2*math.pi
HALF_PI = math.pi/2


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

    def get_uv(self, p: Vec3):
        # find u,v of point hit -- returns tuple of (u:float, v: float)
        phi = math.atan2(p.z, p.x)
        theta = math.asin(p.y)
        u = 1 - (phi + math.pi) / TWO_PI
        v = (theta + HALF_PI) / math.pi
        return (u,v)

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
                n = (p - self.center).div_val(self.radius)
            else:
                t = (-half_b + root) / a

                if t_min < t < t_max:
                    p = ray.at(t)
                    n = (p - self.center).div_val(self.radius)

            if p is not None:
                if self.material.uv_used:
                    uv_pt = (p-self.center).div_val(self.radius)
                    u,v = self.get_uv(uv_pt)
                else:
                    u = v = None

                hr = HitRecord(point=p, normal=n, t=t, material=self.material, u=u, v=v)
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
        self.inverse_normal = -self.normal  # pre-compute to speed up hit testing

        # pre-calculate two basis vectors on the plane (for u,v calculation)
        self._basis_vec_1 = cross(self.normal, Vec3(1, 0, 0)).normalize()
        if self._basis_vec_1.length() < EPSILON:  # normal is parallel to (1,0,0), so use a different vector
            self._basis_vec_1 = cross(self.normal, Vec3(0, 0, 1)).normalize()

        self._basis_vec_2 = cross(self.normal, self._basis_vec_1).normalize()

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

    def get_uv(self, p: Vec3):
        # based on: https://gamedev.stackexchange.com/questions/136652/uv-world-mapping-in-shader-with-unity/136720#136720
        u = dot(self._basis_vec_1, p)

        if u < 0 or u > 1:  # make sure u is between 0 and 1
            _, u = divmod(u, 1.0)

        v = dot(self._basis_vec_2, p)
        if v < 0 or v > 1:  # make sure v is between 0 and 1
            _, v = divmod(u, 1.0)

        return (u,v)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        hr = None
        vd = dot(self.normal, ray.direction)

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
            if self.material.uv_used:
                u, v = self.get_uv(ri)
            else:
                u = v = None

            hr = HitRecord(ri, rn, t, self.material, u, v)
            hr.set_face_normal(ray, rn)

        return hr


class Triangle(Geometry):
    def __init__(self, v0: Vec3, v1: Vec3, v2: Vec3, material: Material, uv0=None, uv1=None, uv2=None):
        """

        :param v0: point 0 of the triangle (Vec3)
        :param v1: point 1 of the triangle (Vec3)
        :param v2: point 2 of the triangle (Vec3)
        :param material: material for the triangle (Material)
        :param uv0: u,v coordinates for v0 (only used if material.uv_used is True)
        :param uv1: u,v coordinates for v1 (only used if material.uv_used is True)
        :param uv2: u,v coordinates for v2 (only used if material.uv_used is True)
        """
        super().__init__(material)
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        self.normal = cross(v0v1, v0v2).normalize()
        self.inverse_normal = -self.normal  # pre-compute for hit testing

        if material.uv_used:
            use_uv0 = uv0 if uv0 is not None else (1,0.5)
            use_uv1 = uv1 if uv1 is not None else (0,1)
            use_uv2 = uv2 if uv2 is not None else (0,0)
            self.u = Vec3(use_uv0[0], use_uv1[0], use_uv2[0])
            self.v = Vec3(use_uv0[1], use_uv1[1], use_uv2[1])

            # pre-compute the following for u,v calculations
            self.v0v1 = v0v1  # pre-compute for barycentric calc to get uv.
            self.v0v2 = v0v2  # pre-compute for barycentric calc to get uv.
            self.d00 = dot(v0v1, v0v1)
            self.d01 = dot(v0v1, v0v2)
            self.d11 = dot(v0v2, v0v2)
            self.inv_denom = 1 / ((self.d00 * self.d11) - (self.d01 * self.d01))

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

    def get_uv(self, p: Vec3):
        # based on barycentric coords of the triangle.
        # from: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates

        # calculate barycentric coordinates a,b,c for the point
        v2 = p - self.v0
        d20 = dot(v2, self.v0v1)
        d21 = dot(v2, self.v0v2)
        b = (self.d11*d20 - self.d01*d21) * self.inv_denom
        c = (self.d00*d21 - self.d01*d20) * self.inv_denom
        a = 1.0 - b - c

        # now calculate u,v
        pu = a*self.u.x + b*self.u.y + c*self.u.z
        pv = a*self.v.x + b*self.v.y + c*self.v.z

        return (pu, pv)

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
            if self.material.uv_used:
                u, v = self.get_uv(ri)
            else:
                u = v = None

            hr = HitRecord(ri, rn, t, self.material, u, v)
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

