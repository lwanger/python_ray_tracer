"""
Scene class

"""

from abc import ABC, abstractmethod
from collections import namedtuple
import math
from random import uniform, randint
from typing import Optional


from geometry_classes import GeometryList, BVHNode, HitRecord, Vec3, Ray
from texture_classes import Texture, SolidColor


class Scene():
    def __init__(self, geometry: GeometryList, lights=None, ambient=None, background: Texture=None):
        """
        Class the represent a scene. Includes the geometry and lighting.

        ambient: ambient lighting
        background: background texture
        TODO:
            add lighting
            add start and end time
            add rendering?
            add camera?
        """
        # self.geometry = geometry  # not needed if you have bvh...
        # self.ambient = ambient_light  # Vec3
        self.bvh = BVHNode(geometry)

        if lights is None:
            self.lights = []
        else:
            self.lights = lights

        if ambient is None:
            self.ambient = Vec3(.1,.1,.1)
        else:
            self.ambient = ambient

        if background is None:
            self.background = SolidColor(Vec3(0.5, 0.7, 1.0))
        else:
            self.background = background


    def hit(self, ray: Ray, tmin: float, tmax: float) -> HitRecord:
        return self.bvh.hit(ray, tmin, tmax)

