"""
Light classes

Len Wanger, copyright, 2020

Hit testing is used for shadow rays. For point lights, a shadow ray. origin is the point hit on the primitive.
world is the Scene. It returns the light contribution. For PointLight this is 0,0,0 (something obstructed the
ray before the light) or self.color. For area lights multiple samples are taken and the contribution is the percent
of the light  seen * the color

For area lights random points are picked using the point_on method on the geometry. For now the material on the
geometry is ignored. In the future this can be used for texture mapping -- allowing for uneven light emissions.
"""

from abc import ABC, abstractmethod

from geometry_classes import Vec3, HitRecord, Ray, Geometry
from scene import Scene


class LightBase(ABC):
    def __init__(self):
        pass

    def __repr__(self):
        pass

    @abstractmethod
    def hit(self, origin, world: Scene) -> Vec3:
        pass

    @abstractmethod
    def value(self) -> Vec3:
        pass


class PointLight(LightBase):
    def __init__(self, pos: Vec3, color: Vec3):
        super().__init__()
        self.color = color
        self.pos = pos

    def __repr__(self):
        return f'PointLight(color={self.color}, pos={self.pos})'

    def value(self) -> Vec3:
        return self.color

    def hit(self, origin, world: Scene) -> Vec3:
        """
        test a shadow ray. origin is the point hit on the primitive. world is the Scene.
        returns the light contribution. For PointLight this is 0,0,0 or self.color. For
        area lights multiple samples are taken and the contribution is the percent of the light
        seen * the color

        origin is the point hit in the scene (the point the shadow ray originates from)
        """
        direction = self.pos - origin
        unit_direction = direction.normalize()
        tmin = 0
        tmax = direction.length()
        ray = Ray(origin, unit_direction)
        hr = world.hit(ray, tmin, tmax)

        if hr is None:  # hit the light
            return self.value()
        else:  # hit something
            return Vec3(0,0,0)


class AreaLight(LightBase):
    def __init__(self, geom: Geometry, color: Vec3, num_samples=10):
        super().__init__()
        self.color = color
        self.geom = geom
        self.num_samples = num_samples  # number of samples to take

    def __repr__(self):
        return f'AreaLight(geom={self.geom}, color={self.color}, num_samples={self.num_samples})'

    def value(self) -> Vec3:
        return self.color

    def hit(self, origin, world: Scene) -> Vec3:
        """
        test a shadow ray. origin is the point hit on the primitive. world is the Scene.
        returns the light contribution. For AreaLight random points on the light are sampled
        with each returning 0,0,0 or self.color, and the value returned being the sum of the
        values divided by the number of samples.

        origin is the point hit in the scene (the point the shadow ray originates from)
        """
        value = Vec3(0,0,0)

        for sample in range(self.num_samples):
            pos = self.geom.point_on()
            direction = pos - origin
            unit_direction = direction.normalize()
            tmin = 0
            tmax = direction.length()
            ray = Ray(origin, unit_direction)
            hr = world.hit(ray, tmin, tmax)

            if hr is None:  # hit the light
                value += self.value()

        ret_val = value.div_val(self.num_samples)
        return ret_val