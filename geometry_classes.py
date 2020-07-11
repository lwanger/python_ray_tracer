"""
Geometry Classes and utility functions

based on Pete Shirley's Ray Tracing in a Weekend (https://raytracing.github.io/books/RayTracingInOneWeekend.html)
Len Wanger, Copyright 2020

removed isinstance calls from Vec3 dunder methods (was very slow!)

"""

from abc import ABC, abstractmethod
from collections import namedtuple
import math
from random import uniform, randint, choice
from typing import Optional


EPSILON = 1e-15
NEG_EPSILON = -1e-15

DEFAULT_ASPECT_RATIO = 16.0/9.0
DEFAULT_FOV = 90.0


MaterialReturn = namedtuple("MaterialReturn", "more scattered attenuation")


def lerp(low_val, high_val, a):
    # linear interpolation -- a (0.0-1.0) * interval[low_val, high_val]
    return a*(high_val-low_val) + low_val

def degrees_to_radians(degrees: float):
    return degrees * math.pi / 180


def clamp(x: float, min: float, max: float) -> float:
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x


def squared_length(v: "Vec3"):
    x,y,z = v.x, v.y, v.z
    return x*x + y*y + z*z


def cross(a: "Vec3", b: "Vec3"):
    return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)


def dot(a: "Vec3", b: "Vec3"):
    return a.x*b.x + a.y*b.y + a.z*b.z


def random_vec3(min: float, max: float) -> "Vec3":
    x = uniform(min, max)
    y = uniform(min, max)
    z = uniform(min, max)
    return Vec3(x, y, z)


def random_unit_vec3() -> "Vec3":
    a = uniform(0, 2*math.pi)
    z = uniform(-1, 1)
    r = math.sqrt(1 - z**2)
    return Vec3(r*math.cos(a), r*math.sin(a), z)

def random_in_unit_sphere() -> "Vec3":
    # pick a point in a unit sphere
    while True:
        p = random_vec3(-1, 1)
        if p.squared_length() < 1:
            break
    return p

def random_on_unit_sphere() -> "Vec3":
    # pick a point on the surface of a unit sphere
    return Vec3(uniform(-1,1), uniform(-1,1), uniform(-1,1)).normalize()


def random_in_hemisphere(normal: "Vec3") -> "Vec3":
    # pick a point in the hemisphere
    in_unit_sphere = random_in_unit_sphere()
    if dot(in_unit_sphere, normal) > 0.0:
        return in_unit_sphere
    else:
        return -in_unit_sphere

def random_in_unit_disc() -> "Vec3":
    # pick a point in a unit disc
    while True:
        p = Vec3(uniform(-1, 1), uniform(-1, 1), 0)
        if p.squared_length() < 1:
            break
    return p


def surrounding_box(box1: "AABB", box2: "AABB") -> "AABB":
    # return the bounding box surrounding two bounding boxes (i.e. the union)
    if box1 is None:
        return box2
    if box2 is None:
        return box1

    b1_min = box1.vmin
    b2_min = box2.vmin
    b1_max = box1.vmax
    b2_max = box2.vmax

    new_x_min = min(b1_min.x, b2_min.x)
    new_y_min = min(b1_min.y, b2_min.y)
    new_z_min = min(b1_min.z, b2_min.z)
    new_min = Vec3(new_x_min, new_y_min, new_z_min)

    new_x_max = max(b1_max.x, b2_max.x)
    new_y_max = max(b1_max.y, b2_max.y)
    new_z_max = max(b1_max.z, b2_max.z)
    new_max = Vec3(new_x_max, new_y_max, new_z_max)

    return AABB(new_min, new_max)

def hex_to_rgb(hex):
    # convert from hex string ("#FFFFFF") to rgb (1.0,1.0,1.0)
    return ( int(hex[1:3], 16) / 255.999, int(hex[3:5], 16) / 255.999, int(hex[5:], 16) / 255.999 )


def get_color(val, colormap):
    # get rgb color from a color map, colormap is a list/tuple, where each entry is either a
    # single float value 0.0-1.0, or a list of strings of hex values for the color. Can use
    # Compatible with colorcet colormaps (e.g colormap==cc.fire)
    colormap_val = colormap[val]

    if isinstance(colormap_val, str):  # colormap is in hex
        color = hex_to_rgb(colormap_val)
        return color
    else:
        return colormap_val


class Vec3():
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f'Vec3({self.x:0.3f}, {self.y:0.3f}, {self.z:0.3f})'

    def __str__(self):
        return f'({self.x:0.3f}, {self.y:0.3f}, {self.z:0.3f})'

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


class Camera():
    def __init__(self, look_from: Vec3, look_at: Vec3, vup: Vec3, vert_fov: float=DEFAULT_FOV,
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
            # direction = direction.normalize()  # Lenw... DELETE?
        else:
            # rd = random_in_unit_disc() * self.lens_radius
            rd = random_in_unit_disc().mul_val(self.lens_radius)
            # offset = self.u * rd.x + self.v * rd.y
            offset = self.u.mul_val(rd.x) + self.v.mul_val(rd.y)
            origin = self.origin + offset
            # direction = self.lower_left_corner + self.horizontal*s + self.vertical*t - self.origin - offset
            direction = self.lower_left_corner + self.horizontal.mul_val(s) + self.vertical.mul_val(t) - self.origin - offset
            # direction = direction.normalize()  # Lenw... DELETE?
        return Ray(origin, direction)


    def __repr__(self):
        return f'Camera(origin={self.origin}, ...)'


class Ray():
    def __init__(self, origin: Vec3, direction: Vec3, tmin: float = None, tmax: float = None):
        self.origin = Vec3(origin.x, origin.y, origin.z)
        self.direction = Vec3(direction.x, direction.y, direction.z)
        self.inv_direction = -self.direction  # inverse direction useful to be pre-computed for hit test.
        self.tmin = tmin
        self.tmax = tmax

    def __repr__(self):
        return f'Ray(origin={self.origin}, direction={self.direction}, tmin={self.tmin}, tmax={self.tmax})'

    def at(self, t: float):
        # return self.origin + self.direction * t
        return self.origin + self.direction.mul_val(t)


class HitRecord():

    def __init__(self, point: Vec3, normal: Vec3, t: float, material: "Material", u: float=None, v: float=None):
        self.point = point
        self.normal = normal
        self.t = t
        self.u = u
        self.v = v
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


class AABB():
    # axis-aligned bounding box -- used for ray intersection speedup.
    def __init__(self, vmin: Vec3, vmax: Vec3):
        self.vmin = vmin
        self.vmax = vmax

    def __repr__(self):
        return f'AABB(vmin={self.vmin}, vmax={self.vmax})'

    def hit(self, ray: Ray, tmin: float, tmax: float) -> bool:
        ro = ray.origin
        rd = ray.direction
        vmin = self.vmin
        vmax = self.vmax

        # check x slab
        if rd.x == 0:
            if ro.x < vmin.x or ro.x > vmax.x:
                return False
        else:
            inv_d = 1 / rd.x
            t0 = (vmin.x - ro.x) * inv_d
            t1 = (vmax.x - ro.x) * inv_d
            t0, t1 = (t1, t0) if inv_d < 0.0 else (t0, t1)
            tmin = t0 if t0 > tmin else tmin
            tmax = t1 if t1 < tmax else tmax

            if tmax < tmin:
                return False

        # check y slab
        if rd.y == 0:
            if ro.y < vmin.y or ro.y > vmax.y:
                return False
        else:
            inv_d = 1 / rd.y
            t0 = (vmin.y - ro.y) * inv_d
            t1 = (vmax.y - ro.y) * inv_d
            t0, t1 = (t1, t0) if inv_d < 0.0 else (t0, t1)
            tmin = t0 if t0 > tmin else tmin
            tmax = t1 if t1 < tmax else tmax

            if tmax < tmin:
                return False

        # check z slab
        if rd.z == 0:
            if ro.z < vmin.z or ro.z > vmax.z:
                return False
        else:
            inv_d = 1 / rd.z
            t0 = (vmin.z - ro.z) * inv_d
            t1 = (vmax.z - ro.z) * inv_d
            t0, t1 = (t1, t0) if inv_d < 0.0 else (t0, t1)
            tmin = t0 if t0 > tmin else tmin
            tmax = t1 if t1 < tmax else tmax

            if tmax < tmin:
                return False

        return True


class Geometry(ABC):
    # abstract base class for hittable geometry
    def __init__(self, material: "Material"):
        self.material = material

    @abstractmethod
    def has_bbox(self) -> bool:
        # returns True if the primitive has a bounding box, False otherwise (e.g. planes)
        pass

    @abstractmethod
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        # ray / geometry intersection method. Returns a HitRecord of the closest hit or None (no intersection)
        pass

    @abstractmethod
    def bounding_box(self, t0: float, t1: float) -> AABB:
        # returns axis-aligned bounding box (AABB) for the geometry, or None if there is no AABB (e.g. for a plane)
        # t0 and t1 are used for start and stop time (not used for stationary objects)
        pass

    @abstractmethod
    def point_on(self):
        # return a uniformly distributed random point on the primitive. Used for sampling lights
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

    def __len__(self):
        return len(self.list)

    def add(self, geom: Geometry):
        self.list.append(geom)

    def clear(self):
        self.list = []

    def no_has_bbox_list(self):
        return [g for g in self.list if g.has_bbox() is False]

    def has_bbox_list(self):
        return [g for g in self.list if g.has_bbox() is True]

    def bounding_box(self, t0: float, t1: float) -> AABB:
        if len(self.list) == 0:
            return None

        temp_box = None

        for geom in self.list:
            bbox = geom.bounding_box(t0, t1)

            if temp_box is None:
                output_box = bbox
            else:
                output_box = surrounding_box(output_box, temp_box)

    def hit(self, ray: Ray,t_min: float, t_max: float):
        closest_so_far = t_max
        hr = None

        for geom in self.list:
            new_hr = geom.hit(ray, t_min, closest_so_far)
            if new_hr is not None:
                closest_so_far = new_hr.t
                hr = new_hr

        return hr


def box_x_compare(a, time0=0, time1=0):
    # returns x component of the Geometry (or other Hittable)... used for list sorting key
    box = a.bounding_box(time0, time1)
    return box.vmin.x

def box_y_compare(a, time0=0, time1=0):
    # returns y component of the Geometry (or other Hittable)... used for list sorting key
    box = a.bounding_box(time0, time1)
    return box.vmin.y

def box_z_compare(a, time0=0, time1=0):
    # returns z component of the Geometry (or other Hittable)... used for list sorting key
    box = a.bounding_box(time0, time1)
    return box.vmin.z


class BVHNode(Geometry):
    def __init__(self, geometry_list: GeometryList, time0: float=0, time1: float=0):
        """
        :param geometry_list: list of Geometry objects to add the the BVH
        :param time0: the start time (for moving scenes and motion blur)
        :param time1: the end time  (for moving scenes and motion blur)

        each node has three elements:
            - left geometry or BVHNode for the left half of the tree
            - right geometry or BVHNode for the right half of the tree
            - bbox - the bounding box (AABB) for this node

        Builds the BVH (bounding volume hierarchy).BVHNode is a node in a binary tree.
        Each left/right pair (node) splits the scene by a random axis at
        each level of the hierarchy:

            1) randomly choose an axis
            2) sort the primitives (using std::sort)
            3) put half in each subtree)

        When the list has only one or two elements, put one in each subtree and end the recursion.

        Finally set a bbox for the node

        note: If there are any primitives with no bbox (i.e. has_bbox==False), then
        put them all in a left node with math.-inf and math.inf for the bbox. If all are no bbox
        then split between left and right. This will force a hit on them.

            def has_bbox(self) -> bool:
        """

        if len(geometry_list) == 0:
            raise RuntimeError("BVHNode.__init__: Empty geometry list")

        no_bbox_list = geometry_list.no_has_bbox_list()
        has_bbox_list = geometry_list.has_bbox_list()

        if len(no_bbox_list) != 0:
            vmin = Vec3(-math.inf, -math.inf, -math.inf)
            vmax = Vec3(math.inf, math.inf, math.inf)
            self.bbox = AABB(vmin, vmax)

            if len(has_bbox_list) == 0:
                if len(no_bbox_list) == 1:
                    self.left = no_bbox_list[0]
                    self.right = None
                elif len(no_bbox_list) == 2:
                    self.left = no_bbox_list[0]
                    self.right = no_bbox_list[1]
                else:
                    mid = len(no_bbox_list) // 2
                    geom_list = GeometryList(no_bbox_list[:mid])
                    self.left = BVHNode(geom_list, time0, time1)
                    geom_list = GeometryList(no_bbox_list[mid:])
                    self.right = BVHNode(geom_list, time0, time1)
            else:  # len(has_bbox_list) != 0
                if len(no_bbox_list) == 1:
                    self.left = no_bbox_list[0]
                else:
                    geom_list = GeometryList(no_bbox_list)
                    self.left = BVHNode(geom_list, time0, time1)

                if len(has_bbox_list) == 1:
                    self.right = has_bbox_list[0]
                else:
                    geom_list = GeometryList(has_bbox_list)
                    self.right = BVHNode(geom_list, time0, time1)
        else:  # len(no_bbox_list) == 0
            if len(has_bbox_list) > 1:
                axis = randint(0,2)
                comparators = {0: box_x_compare, 1: box_y_compare, 2: box_z_compare}
                comparator = comparators[axis]

            if len(has_bbox_list) == 1:
                self.left = has_bbox_list[0]
                self.right = None
            elif len(has_bbox_list) == 2:
                obj0 = has_bbox_list[0]
                obj1 = has_bbox_list[1]

                if comparator(obj0) < comparator(obj1):
                    self.left = obj0
                    self.right = obj1
                else:
                    self.left = obj1
                    self.right = obj0
            else:
                has_bbox_list.sort(key=comparator)
                mid = len(has_bbox_list) // 2
                geom_list_left = GeometryList(has_bbox_list[:mid])
                self.left = BVHNode(geom_list_left, time0, time1)
                geom_list_right = GeometryList(has_bbox_list[mid:])
                self.right = BVHNode(geom_list_right, time0, time1)

            right_bbox = None
            left_bbox = self.left.bounding_box(time0, time1)
            if self.right is not None:
                right_bbox = self.right.bounding_box(time0, time1)

            self.bbox = surrounding_box(left_bbox, right_bbox)

    def __repr__(self):
        return(f'BVHNode(left={type(self.left)}, right={type(self.right)}, bbox={self.bbox})')

    def has_bbox(self) -> bool:
        return True

    def bounding_box(self, t0: float, t1: float) -> AABB:
        return self.bbox

    def point_on(self):
        # pick a random primitive in the BVH and return point_on for it.
        def pick_node(node: BVHNode):
            if node.right is None:
                return self.left
            dir = choice([0,1])
            if dir==0:  # left
                pick = node.left
            else:
                pick = node.right

            if isinstance(pick, BVHNode):
                r = pick_node(pick)
            else:
                return pick

            return r

        prim = pick_node(self)
        p = prim.point_on()
        return p

    def hit(self, ray: Ray,t_min: float=0.0, t_max: float=math.inf) -> HitRecord:
        hr = self.bbox.hit(ray, t_min, t_max)

        if hr is False:
            return None

        hit_left = self.left.hit(ray, t_min, t_max)

        if self.right is None:
            return hit_left

        hit_right = self.right.hit(ray, t_min, t_max)

        # return [lst for lst in (hit_left, hit_right) if lst is not None]
        if hit_left is None:
            return hit_right
        elif hit_right is None:
            return hit_left
        else: # return lowest t_min
            if hit_left.t < hit_right.t:
                return hit_left
            else:
                return hit_right


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

