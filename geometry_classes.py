"""
Geometry Classes and utility functions

based on Pete Shirley's Ray Tracing in a Weekend (https://raytracing.github.io/books/RayTracingInOneWeekend.html)
Len Wanger, Copyright 2020

"""

from abc import ABC, abstractmethod
from collections import namedtuple
import numbers
import numpy
import math
from random import uniform
from typing import Optional

_TINY = 1e-15


def _args2tuple(funcname, args):
    narg = len(args)
    if narg == 0:
        data = 3 * (0,)
    elif narg == 1:
        data = args[0]
        if len(data) != 3:
            raise TypeError('vec3.%s() takes sequence with 3 elements '
                            '(%d given),\n\t   when 1 argument is given' %
                            (funcname, len(data)))
    elif narg == 3:
        data = args
    else:
        raise TypeError('vec3.%s() takes 0, 1 or 3 arguments (%d given)' %
                        (funcname, narg))
    assert len(data) == 3
    try:
        return tuple(map(float, data))
    except (TypeError, ValueError):
        raise TypeError("vec3.%s() can't convert elements to float" % funcname)


def degrees_to_radians(degrees: float):
    return degrees * math.pi / 180

def clamp(x: float, min: float, max: float) -> float:
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x

class Vec3(numpy.ndarray):
    def __new__(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], Vec3):
                return args[0].copy()
            if isinstance(args[0], numpy.matrix):
                return Vec3(args[0].flatten().tolist()[0])
        data = _args2tuple('__new__', args)
        arr = numpy.array(data, dtype=numpy.float, copy=True)
        return numpy.ndarray.__new__(cls, shape=(3,), buffer=arr)

    def __repr__(self):
        return f'Vec3({self[0]}, {self[1]}, {self[2]})'

    def __str__(self):
        return f'({self[0]}, {self[1]}, {self[2]})'

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return Vec3(self.x*other, self.y*other, self.z*other)

        return Vec3(self.x*other.x, self.y*other.y, self.z*other.z)

    def get_x(self):
        return self[0]

    def set_x(self, v):
        self[0] = v

    x = property(get_x, set_x)
    r = property(get_x, set_x)

    def get_y(self):
        return self[1]

    def set_y(self, v):
        self[1] = v

    y = property(get_y, set_y)
    g = property(get_y, set_y)

    def get_z(self):
        return self[2]

    def set_z(self, v):
        self[2] = v

    z = property(get_z, set_z)
    b = property(get_z, set_z)

    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def squared_length(self):
        return self.x**2 + self.y**2 + self.z**2

    def normalize(self):
        k = 1.0 / math.sqrt(self.x**2 + self.y**2 + self.z**2)
        return Vec3(self.x*k, self.y*k, self.z*k)

    def unit_vector(self):
        return self.normalize()

    def dot(self, other):
        return numpy.dot(self, other)

    def cross(self, v):
        return Vec3(numpy.cross(self, v))

    def get_color(self):
        return 255.999 * self.r, 255.999 * self.g, 255.999 * self.b


def cross(a: Vec3, b: Vec3):
    return Vec3(numpy.cross(a, b))

def dot(a: Vec3, b: Vec3):
    return numpy.dot(a, b)

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

class Camera():
    def __init__(self):
        aspect_ratio = 16.0 / 9.0
        viewport_height = 2.0
        viewport_width = aspect_ratio * viewport_height
        focal_length = 1.0

        self.origin = Vec3(0, 0, 0)
        self.horizontal = Vec3(viewport_width, 0, 0)
        self.vertical = Vec3(0, viewport_height, 0)
        self.lower_left_corner = self.origin - self.horizontal / 2 - self.vertical / 2 - Vec3(0, 0, focal_length)

    def get_ray(self, u: float, v: float):
        return Ray(self.origin, self.lower_left_corner + u*self.horizontal + v*self.vertical - self.origin)

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

    def at(self, t):
        return self.origin + t * self.direction


MaterialReturn = namedtuple("MaterialReturn", "more scattered attenuation")


class Material(ABC):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'Material(name={self.name})'

    @abstractmethod
    def scatter(self, ray_in: Ray, hr: "HitRecord") -> MaterialReturn:
        pass


def reflect(v: Vec3, n: Vec3):
    return v - 2 * dot(v,n) * n


class Lambertian(Material):
    def __init__(self, color: Vec3):
        self.albedo = color

    def scatter(self, ray_in: Ray, hr: "HitRecord") -> MaterialReturn:
        scatter_direction = hr.normal + random_unit_vec3()
        scattered = Ray(hr.point, scatter_direction)
        attenuation = self.albedo
        return MaterialReturn(True, scattered, attenuation)


class Metal(Material):
    def __init__(self, color: Vec3, fuzziness:float=1.0):
        self.albedo = color
        if fuzziness > 1.0:
            self.fuzziness = 1.0
        else:
            self.fuzziness = fuzziness

    def scatter(self, ray_in: Ray, hr: "HitRecord") -> MaterialReturn:
        unit_vector = ray_in.direction.unit_vector()
        reflected = reflect(unit_vector, hr.normal)
        scattered = Ray(hr.point, reflected + self.fuzziness * random_in_unit_sphere())
        attenuation = self.albedo
        more = dot(scattered.direction, hr.normal) > 0
        return MaterialReturn(more, scattered, attenuation)


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
                n = (p - self.center) / self.radius
                # hr = HitRecord(point=p, normal=n, t=t, material=self.material)
            else:
                t = (-half_b + root) / a

                if t_min < t < t_max:
                    p = ray.at(t)
                    n = (p - self.center) / self.radius
                    # hr = HitRecord(point=p, normal=n, t=t, material=self.material)

            # if hr is not None:
            if p is not None:
                hr = HitRecord(point=p, normal=n, t=t, material=self.material)
                outward_normal = Vec3(hr.point - self.center) / self.radius
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

    v2[0] = 4.1
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

    print(f'v1.dot(v2) = {v1.dot(v2)}')
    print(f'v1.normalize = {v1.normalize()}')
    print(f'v1.unit_vector = {v1.unit_vector()}')
    print(f'v1.cross(v2) = {v1.cross(v2)}')

    print(f'ray1={ray1}, repr={repr(ray1)}')
    print(f'ray1 repr={repr(ray1)}')
    print(f'ray1.at(0.0)={ray1.at(0.0)}')
    print(f'ray1.at(0.5)={ray1.at(0.5)}')
    print(f'ray1.at(1.0)={ray1.at(1.0)}')

