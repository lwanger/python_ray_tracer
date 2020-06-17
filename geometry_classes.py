"""
Geometry Classes

TODO:
    - create pytest tests for Vec3 and Ray
"""
import numbers
import numpy
import math

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
        # return numpy.dot(self, other)
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

    # def __len__(self):
    #     return math.sqrt(self.x**2 + self.y**2 + self.z**2)

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

