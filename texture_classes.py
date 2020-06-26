"""
Texture mapping classes

based on Pete Shirley's Ray Tracing the next Weekend (https://raytracing.github.io/books/RayTracingTheNextWeek.html#boundingvolumehierarchies)

Len Wanger, Copyright 2020
"""

from abc import ABC, abstractmethod
import math

from geometry_classes import Vec3


class Texture(ABC):
    def __init__(self, uv_used=False, name='unnamed'):
        self.name = name
        self.uv_used = uv_used

    def __repr__(self):
        return f'Texture(name={self.name})'

    @abstractmethod
    def value(self, u: float, v: float) -> Vec3:
        # return color (Vec3) for the u,v point
        pass


class SolidColor(Texture):
    # a solid color texture
    def __init__(self, color: Vec3, name='unnamed_solid_color'):
        super().__init__(uv_used=False, name=name)
        self.color = color
        self.name = name

    def __repr__(self):
        return f'SolidColor(color={self.color}, name={self.name})'

    def value(self, u: float, v: float, p: Vec3) -> Vec3:
        return self.color


class CheckerBoard(Texture):
    # a checkerboard texture -- even and odd are themselves textures
    def __init__(self, even: Texture, odd: Texture, spacing: int=10, name='checker_board'):
        super().__init__(uv_used=False, name=name)
        self.even = even
        self.odd = odd
        self.spacing = spacing

    def __repr__(self):
        return f'CheckerBoard(even={self.even.name}, odd={self.odd.name}, name={self.name})'

    def value(self, u: float, v: float, p: Vec3) -> Vec3:
        spacing = self.spacing
        sines = math.sin(spacing*p.x) * math.sin(spacing*p.y) * math.sin(spacing*p.z)
        if sines < 0:
            return self.odd.value(u,v,p)
        else:
            return self.even.value(u,v,p)

        return self.color