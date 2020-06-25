"""
Material Classes and utility functions

based on Pete Shirley's Ray Tracing in a Weekend (https://raytracing.github.io/books/RayTracingInOneWeekend.html)
Len Wanger, Copyright 2020

"""

from abc import ABC, abstractmethod
from collections import namedtuple
import math
from random import random

from geometry_classes import Vec3, Geometry, Ray, dot, random_unit_vec3, random_in_unit_sphere


_TINY = 1e-15


MaterialReturn = namedtuple("MaterialReturn", "more scattered attenuation")


class Material(ABC):
    def __init__(self, name: str='unnamed'):
        self.name = name

    def __repr__(self):
        return f'Material(name={self.name})'

    @abstractmethod
    def scatter(self, ray_in: Ray, hr: "HitRecord") -> MaterialReturn:
        pass


def reflect(v: Vec3, n: Vec3):
    # return v - n*2*dot(v,n)
    return v - n.mul_val(2*dot(v,n))


def schlick(cosine: float, refraction_idx: float):
    # Schlick approximation for changing refraction with angle
    r0 = ((1 - refraction_idx) / (1 + refraction_idx)) ** 2
    return r0 + (1-r0) * math.pow((1 -cosine), 5)


def refract(uv: Vec3, n: Vec3, etai_over_etat: float):
    cos_theta = dot(-uv, n)
    # r_out_parallel = (uv + n*cos_theta) * etai_over_etat
    r_out_parallel = (uv + n.mul_val(cos_theta)).mul_val(etai_over_etat)
    # r_out_perp = n * (-math.sqrt(1.0 - r_out_parallel.squared_length()))
    r_out_perp = n.mul_val((-math.sqrt(1.0 - r_out_parallel.squared_length())))
    return r_out_parallel + r_out_perp


class Lambertian(Material):
    def __init__(self, color: Vec3, name=None):
        super().__init__(name)
        self.albedo = color

    def __repr__(self):
        return f'Lambertian(name={self.name}, albedo={self.albedo})'

    def scatter(self, ray_in: Ray, hr: "HitRecord") -> MaterialReturn:
        scatter_direction = hr.normal + random_unit_vec3()
        scattered = Ray(hr.point, scatter_direction)
        attenuation = self.albedo
        return MaterialReturn(True, scattered, attenuation)


class Metal(Material):
    def __init__(self, color: Vec3, fuzziness:float=1.0, name=None):
        super().__init__(name)
        self.albedo = color
        if fuzziness > 1.0:
            self.fuzziness = 1.0
        else:
            self.fuzziness = fuzziness

    def __repr__(self):
        return f'Metal(name={self.name}, albedo={self.albedo}, fuzziness={self.fuzziness})'

    def scatter(self, ray_in: Ray, hr: "HitRecord") -> MaterialReturn:
        unit_vector = ray_in.direction.unit_vector()
        reflected = reflect(unit_vector, hr.normal)
        # scattered = Ray(hr.point, reflected + random_in_unit_sphere()*self.fuzziness)
        scattered = Ray(hr.point, reflected + random_in_unit_sphere().mul_val(self.fuzziness))
        attenuation = self.albedo
        more = dot(scattered.direction, hr.normal) > 0
        return MaterialReturn(more, scattered, attenuation)


class Dielectric(Material):
    # transparent material
    def __init__(self, refractive_index: float, name=None):
        super().__init__(name)
        self.refraction_idx = refractive_index

    def __repr__(self):
        return f'Dielectric(name={self.name}, refraction_idx={self.refraction_idx})'

    def scatter(self, ray_in: Ray, hr: "HitRecord") -> MaterialReturn:
        attenuation = Vec3(1,1,1)

        if hr.front_face is True:
            etai_over_etat = 1 / self.refraction_idx
        else:
            etai_over_etat = self.refraction_idx

        unit_direction = ray_in.direction.unit_vector()
        cos_theta = min( dot(-unit_direction, hr.normal), 1.0 )
        sin_theta = math.sqrt(1.0 - cos_theta ** 2)

        if (etai_over_etat * sin_theta) > 1.0:
            reflected = reflect(unit_direction, hr.normal)
            scattered = Ray(hr.point, reflected)
        else:
            reflect_prob = schlick(cos_theta, etai_over_etat)
            if (random() < reflect_prob):
                reflected = reflect(unit_direction, hr.normal)
                scattered = Ray(hr.point, reflected)
            else:
                refracted = refract(unit_direction, hr.normal, etai_over_etat)
                scattered = Ray(hr.point, refracted)

        return MaterialReturn(True, scattered, attenuation)


def ray_color(ray: Ray, world: Geometry, depth=1):
    if depth < 1:
        return Vec3(0, 0, 0)

    hr = world.hit(ray, 0.001, math.inf)

    if hr is not None:
        matl_record = hr.material.scatter(ray, hr)
        if matl_record.more:
            return matl_record.attenuation * ray_color(matl_record.scattered, world, depth-1)
        else:
            return Vec3(0,0,0)

    unit_direction = ray.direction.unit_vector()
    t = 0.5 * (unit_direction.y + 1.0)
    # return Vec3(1.0, 1.0, 1.0)*(1-t) + Vec3(0.5, 0.7, 1.0)*t
    return Vec3(1.0, 1.0, 1.0).mul_val(1-t) + Vec3(0.5, 0.7, 1.0).mul_val(t)