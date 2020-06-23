"""
Geometry Classes and utility functions

based on Pete Shirley's Ray Tracing in a Weekend (https://raytracing.github.io/books/RayTracingInOneWeekend.html)
Len Wanger, Copyright 2020

removed isinstance calls from Vec3 dunder methods (was very slow!)

"""

from abc import ABC, abstractmethod
from collections import namedtuple
import math
from random import uniform
from typing import Optional

_TINY = 1e-15
DEFAULT_ASPECT_RATIO = 16.0/9.0
DEFAULT_FOV = 90.0


def degrees_to_radians(degrees: float):
    return degrees * math.pi / 180

def clamp(x: float, min: float, max: float) -> float:
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x


class Vec3():
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f'Vec3({self.x}, {self.y}, {self.z})'

    def __str__(self):
        return f'({self.x}, {self.y}, {self.z})'

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __add__(self, other):
        # if isinstance(other, numbers.Number):
        #     return Vec3(self.x+other, self.y+other, self.z+other)
        return Vec3(self.x+other.x, self.y+other.y, self.z+other.z)

    def add_val(self, value):
        return Vec3(self.x+value, self.y+value, self.z+value)

    def __sub__(self, other):
        # if isinstance(other, numbers.Number):
        #     return Vec3(self.x-other, self.y-other, self.z-other)
        return Vec3(self.x-other.x, self.y-other.y, self.z-other.z)

    def sub_val(self, value):
        return Vec3(self.x-value, self.y-value, self.z-value)

    def __mul__(self, other):
        # if isinstance(other, numbers.Number):
        #     return Vec3(self.x*other, self.y*other, self.z*other)
        return Vec3(self.x*other.x, self.y*other.y, self.z*other.z)

    def mul_val(self, value):
        return Vec3(self.x*value, self.y*value, self.z*value)

    def __truediv__(self, other):
        # if isinstance(other, numbers.Number):
        #     return Vec3(self.x/other, self.y/other, self.z/other)
        return Vec3(self.x/other.x, self.y/other.y, self.z/other.z)

    def div_val(self, value):
        return Vec3(self.x/value, self.y/value, self.z/value)

    def get_x(self):
        return self.x

    def set_x(self, v):
        self[0] = v

    r = property(get_x, set_x)

    def get_y(self):
        return self.y

    def set_y(self, v):
        self[1] = v

    g = property(get_y, set_y)

    def get_z(self):
        return self.z

    def set_z(self, v):
        self[2] = v

    b = property(get_z, set_z)

    def length(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def squared_length(self):
        return self.x*self.x + self.y*self.y + self.z*self.z

    def normalize(self):
        k = 1.0 / math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
        return Vec3(self.x*k, self.y*k, self.z*k)

    def unit_vector(self):
        return self.normalize()

    def get_unscaled_color(self):
        return self.x, self.y, self.z

    def get_color(self):
        return 255.999 * self.x, 255.999 * self.y, 255.999 * self.z


def squared_length(v: Vec3):
    x,y,z = v.x, v.y, v.z
    # return v.x*v.x + v.y*v.y + v.z*v.z
    return x*x + y*y + z*z

def cross(a: Vec3, b: Vec3):
    return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)

def dot(a: Vec3, b: Vec3):
    return a.x*b.x + a.y*b.y + a.z*b.z


def random_vec3(min: float, max: float) -> Vec3:
    x = uniform(min, max)
    y = uniform(min, max)
    z = uniform(min, max)
    return Vec3(x, y, z)


def random_unit_vec3() -> Vec3:
    a = uniform(0, 2*math.pi)
    z = uniform(-1, 1)
    r = math.sqrt(1 - z**2)
    return Vec3(r*math.cos(a), r*math.sin(a), z)

def random_in_unit_sphere() -> Vec3:
    # pick a point in a unit sphere
    while True:
        p = random_vec3(-1, 1)
        if p.squared_length() < 1:
            break
    return p

def random_in_hemisphere(normal: Vec3) -> Vec3:
    # pick a point in the hemisphere
    in_unit_sphere = random_in_unit_sphere()
    if dot(in_unit_sphere, normal) > 0.0:
        return in_unit_sphere
    else:
        return -in_unit_sphere

def random_in_unit_disc() -> Vec3:
    # pick a point in a unit disc
    while True:
        p = Vec3(uniform(-1, 1), uniform(-1, 1), 0)
        if p.squared_length() < 1:
            break
    return p


class Camera():
    def __init__(self, look_from: Vec3, look_at: Vec3, vup: Vec3, vert_fov: float=DEFAULT_FOV,
                 # aspect_ratio: float=DEFAULT_ASPECT_RATIO, aperature:float=0.0, focus_dist:float=10):
                 aspect_ratio: float=DEFAULT_ASPECT_RATIO, aperature:float=0.0, focus_dist:float=math.inf):
        self.origin = look_from
        self.look_at = look_at
        self.look_vup = vup

        self.w = (look_from - look_at).unit_vector()
        self.u = cross(vup, self.w).unit_vector()
        self.v = cross(self.w, self.u)

        theta = degrees_to_radians(vert_fov)
        h = math.tan(theta/2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        if focus_dist == math.inf:
            # self.horizontal = viewport_width * self.u
            self.horizontal = self.u.mul_val(viewport_width)
            # self.vertical = viewport_height * self.v
            self.vertical =  self.v.mul_val(viewport_height)
            # self.lower_left_corner = self.origin - self.horizontal/2 - self.vertical/2 - self.w
            self.lower_left_corner = self.origin - self.horizontal.div_val(2) - self.vertical.div_val(2) - self.w
            self.lens_radius = aperature / 2
        else:
            # self.horizontal = self.u * viewport_width * focus_dist
            self.horizontal = self.u.mul_val(viewport_width * focus_dist)
            # self.vertical  = self.v * viewport_height * focus_dist
            self.vertical  = self.v.mul_val(viewport_height * focus_dist)
            # self.lower_left_corner = self.origin - self.horizontal/2 - self.vertical/2 - self.w * focus_dist
            self.lower_left_corner = self.origin - self.horizontal.div_val(2) - self.vertical.div_val(2) - self.w.mul_val(focus_dist)
            self.lens_radius = aperature / 2

    def get_ray(self, s: float, t: float):
        if self.lens_radius < 0.01:
            origin = self.origin
            # direction = self.lower_left_corner + s*self.horizontal + t*self.vertical - self.origin
            direction = self.lower_left_corner + self.horizontal.mul_val(s) + self.vertical.mul_val(t) - self.origin
        else:
            # rd = random_in_unit_disc() * self.lens_radius
            rd = random_in_unit_disc().mul_val(self.lens_radius)
            # offset = self.u * rd.x + self.v * rd.y
            offset = self.u.mul_val(rd.x) + self.v.mul_val(rd.y)
            origin = self.origin + offset
            # direction = self.lower_left_corner + self.horizontal*s + self.vertical*t - self.origin - offset
            direction = self.lower_left_corner + self.horizontal.mul_val(s) + self.vertical.mul_val(t) - self.origin - offset
        return Ray(origin, direction)


    def __repr__(self):
        return f'Camera(origin={self.origin}, ...)'


class Ray():
    def __init__(self, origin: Vec3, direction: Vec3, tmin: float = None, tmax: float = None):
        self.origin = Vec3(origin.x, origin.y, origin.z)
        self.direction = Vec3(direction.x, direction.y, direction.z)
        self.tmin = tmin
        self.tmax = tmax

    def __repr__(self):
        return f'Ray(origin={self.origin}, direction={self.direction}, tmin={self.tmin}, tmax={self.tmax})'

    def at(self, t: float):
        # return self.origin + self.direction * t
        return self.origin + self.direction.mul_val(t)


MaterialReturn = namedtuple("MaterialReturn", "more scattered attenuation")


class Material(ABC):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'Material(name={self.name})'

    @abstractmethod
    def scatter(self, ray_in: Ray, hr: "HitRecord") -> MaterialReturn:
        pass


class HitRecord():

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


class Geometry(ABC):
    # abstract base class for hittable geometry
    def __init__(self, material: Material):
        self.material = material

    @abstractmethod
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        pass


class GeometryList():
    def __init__(self, initial_list=None):
        if initial_list is None:
            self.list = []
        else:
            self.list = initial_list

    def __iter__(self):
        return self.list.__iter__()

    def __repr__(self):
        return f'GeometryList(list={len(self.list)} items)'

    def add(self, geom: Geometry):
        self.list.append(geom)

    def clear(self):
        self.list = []

    def hit(self, ray: Ray,t_min: float, t_max: float):
        closest_so_far = t_max
        hr = None

        for geom in self.list:
            new_hr = geom.hit(ray, t_min, closest_so_far)
            if new_hr is not None:
                closest_so_far = new_hr.t
                hr = new_hr

        return hr


class Sphere(Geometry):
    def __init__(self, center: Vec3, radius: float, material: Material):
        super().__init__(material)
        self.center = center
        self.radius = radius

    def __repr__(self):
        return f'Sphere(center={self.center}, radius={self.radius}, material={self.material})'

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

