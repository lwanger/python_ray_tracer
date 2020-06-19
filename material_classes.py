"""
Material Classes and utility functions

based on Pete Shirley's Ray Tracing in a Weekend (https://raytracing.github.io/books/RayTracingInOneWeekend.html)
Len Wanger, Copyright 2020

"""

from abc import ABC, abstractmethod
from collections import namedtuple
import math
from random import random

from geometry_classes import Vec3, Ray, dot, random_unit_vec3, random_in_unit_sphere

_TINY = 1e-15


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


def schlick(cosine: float, refraction_idx: float):
    # Schlick approximation for changing refraction with angle
    r0 = ((1 - refraction_idx) / (1 + refraction_idx)) ** 2
    return r0 + (1-r0) * math.pow((1 -cosine), 5)


def refract(uv: Vec3, n: Vec3, etai_over_etat: float):
    cos_theta = dot(-uv, n)
    r_out_parallel = etai_over_etat * (uv + cos_theta*n)
    r_out_perp = -math.sqrt(1.0 - r_out_parallel.squared_length()) * n
    return r_out_parallel + r_out_perp


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


class Dielectric(Material):
    # transparent material
    def __init__(self, refractive_index: float):
        self.refraction_idx = refractive_index

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