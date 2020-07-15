"""
Texture mapping classes

based on Pete Shirley's Ray Tracing the next Weekend (https://raytracing.github.io/books/RayTracingTheNextWeek.html#boundingvolumehierarchies)

Len Wanger, Copyright 2020
"""

from abc import ABC, abstractmethod
import math

from geometry_classes import Vec3, clamp, lerp
from perlin import ValueNoise3D


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

        return Vec3(pixel.r/255.999, pixel.g/255.999, pixel.b/255.999)


class NoiseTexture(Texture):
    """
    Perlin noise texture:

    point_scale scales the X,Y,Z of the point being examined. This can scale a part over a larger or smaller
        portion of the noise map.

    colormap allows setting nice color maps for the rendering. For instance, using colorcet to get perceptually
    linear/uniform colormaps. e.g.

        import colorcet as cc
        palette = cc.fire
        colormap = [get_color(i,palette) for i in range(len(palette))]

    To use a subset of the palette using a portion of the colormap, e.g.

        colormap = [get_color(i,palette) for i in range(10,150)]

    Some nice colormaps to work with include:

        fire, coolwarm, dimgray, kgy  # jade, kbc, blues  # clouds?, rainbow, CET_CBC1

    The colorcet documentation for a full list of colormaps.
    """

    def __init__(self, colormap=None, point_scale=1.0, translate=0.0, scale=1.0,
                 eval_func=None, eval_args=None, eval_kwargs=None, name='noise_texture'):
        """
        Noise texture:
        :param colormap: a list of colormap values -- either hex (#FFFFFF) or float (0.0-1.0) for each element
        :param point_scale: multiple point by point_scale (scalar value). e.g. if point_scale=10 and p=Vec3(0.1,0.2,0.3)*10
            -> Vec3(1, 2, 3)
        :param translate: add translate to the result of the eval_func
        :param scale: multiple result of the eval_func by scale
        :param eval_func: the function to call for evaluating a point in the noise function (e.g. perlin::fractal_noise)
        :param eval_args: arguments to pass to the eval_func
        :param eval_kwargs: keyword arguments to pass to the eval_func (e.g. {'frequency': 0.5}
        :param name: name for the noise texture
        """
        super().__init__(uv_used=False, name=name)
        self.noise = ValueNoise3D()
        self.colormap = colormap
        self.point_scale = point_scale
        self.eval_func = eval_func
        self.translate = translate
        self.scale = scale

        if eval_args is None:
            self.eval_args = []
        else:
            self.eval_args = eval_args

        if eval_kwargs is None:
            self.eval_kwargs = {}
        else:
            self.eval_kwargs = eval_kwargs


    def __repr__(self):
        return f'NoiseTexture(name={self.name})'


    def raw_value(self, u: float, v: float, p: Vec3) -> Vec3:
        # for debugging or creating noise_maps
        if self.eval_func is None:
            val = self.noise.eval(self.point_scale*p.x, self.point_scale*p.y, self.point_scale*p.z)
        else:
            val = self.eval_func(self.noise, self.point_scale*p.x, self.point_scale*p.y,
                                 self.point_scale*p.z, *(self.eval_args), **(self.eval_kwargs))
        return val


    def get_color(self, val):
        if self.colormap is None:
            return Vec3(val, val, val)
        else:
            idx = int(val * (len(self.colormap) - 0.001))
            idx = min(idx, len(self.colormap) - 1)
            idx = max(idx, 0)
            color = self.colormap[idx]
            return Vec3(*color)


    def value(self, u: float, v: float, p: Vec3) -> Vec3:
        if self.eval_func is None:
            val = self.noise.eval(self.point_scale*p.x, self.point_scale*p.y, self.point_scale*p.z)
        else:
            val = self.eval_func(self.noise, self.point_scale*p.x, self.point_scale*p.y,
                                 self.point_scale*p.z, *(self.eval_args), **(self.eval_kwargs))

        val = (val + self.translate) * self.scale  # min is -0.06 - 0.06 map to 0.0-1.0
        val = clamp(val, 0.0, 1.0)

        return self.get_color(val)

