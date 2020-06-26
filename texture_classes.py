"""
Texture mapping classes

based on Pete Shirley's Ray Tracing the next Weekend (https://raytracing.github.io/books/RayTracingTheNextWeek.html#boundingvolumehierarchies)

Len Wanger, Copyright 2020
"""

from abc import ABC, abstractmethod
import math

from geometry_classes import Vec3, lerp


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
        super().__init__(uv_used=True, name=name)
        self.even = even
        self.odd = odd
        self.spacing = spacing

    def __repr__(self):
        return f'CheckerBoard(even={self.even.name}, odd={self.odd.name}, name={self.name})'

    def value(self, u: float, v: float, p: Vec3) -> Vec3:
        # each square of the checkerboard pi/spacing pixels wide and high, a even/odd pair is 2*pi/spacing wide/high.
        spacing = self.spacing
        sines = math.sin(spacing*p.x) * math.sin(spacing*p.y) * math.sin(spacing*p.z)
        if sines < 0:
            return self.odd.value(u,v,p)
        else:
            return self.even.value(u,v,p)

        return self.color


class ImageTexture(Texture):
    # a texture from a bitmap -- pass in a PIL Image for the image
    def __init__(self, pil_im, name='unnamed_image_texture'):
        super().__init__(uv_used=True, name=name)

        # convert the image to RGB if not already...
        if pil_im.mode == "RGB":
            self.im = pil_im
        else:
            self.im = pil_im.convert(mode="RGB")

        self.name = name

    def __repr__(self):
        return f'ImageTexture(name={self.name})'

    def value(self, u: float, v: float, p: Vec3) -> Vec3:
        x = int(lerp(0, self.im.width-1, u))
        y = int(lerp(0, self.im.height-1, v))
        try:
            pixel = self.im.getpixel((x,y))
        except IndexError:
            RuntimeError(f'ImageTexture::value: getpixel index error (x={x}, y={y})')

        return Vec3(pixel[0]/255.999, pixel[1]/255.999, pixel[2]/255.999)
