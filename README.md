# CS_292

In 1980 Ed Catmull and a number of the computer graphics wizards from Lucasfilm (Jim Blinn, Loren Carpenter, and Alvy Ray Smith) taught a computer graphics class at Berkeley. THis class is on Youtube at:https://www.youtube.com/channel/UCNXre0qpHjdhC29xH8WkKnw.

I thought it would be fun to try to write code for the concepts there. This repository is my attempt.

Our modern tools are very luxurious compared to what they had available. Modern hardware and software tools like Python and NumPy mad this very easy in comparison to C on PDP 11's.

First is a Python/NumPy class to act as a framebuffer. The FrameBuffer class has some simple methods:

__init__(x, y, dtype) -- x size, y size and data type (see numpy for types). defaults to type uint8 for ints 0..256
get_x_size() -- returns the X size of the framebuffer
get_y_size() -- returns the Y size of the framebuffer
set_pixel(x, y, val) -- set the pixel at location x, y to the value (val)
get_pixel(x,y) -- return the value of the pixel
make_image(mode) -- creates an image using PIL. For list of modes see pillow documentation. Defaults to "L" - 0..256 luminosity value. There are two functions (show_image and save_image) that can take the return value and show it on screen or save it to disk.

Ray Tracing:
# Also in here is ray tracing from Pete Shirley's _Ray Tracing in One Weekend_ book:

for example, run: python listing_60.py

optional environment variables:

    - SAMPLES_PER_PIXEL (int e.g. 50)
    - USE_RES: low', 'med', 'high', or 'ultra'. Sets to the settings in res_settings. Each can be overwritten by the
        variables below.
    - X_SIZE: x size of the rendered image
    - ASPECT_RATIO: aspect ratio of the rendered image -- used to calculate y size (default is 16:9)
    - SAMPLES_PER_PIXEL: samples per pixel
    - MAX_DEPTH: maximum depth of bounces per pixel
    - CHUNK_SIZE: size of chunks to calculate (e.g. value of 10 is 10x10 pixel blocks)
    - RANDOM_CHUNKS: whether rendered chunks are in order or random (True - default)
    - IMAGE_FILENAME: the file name to use to save the image

Lots of fun stuff to do on the ray tracing:

- clean up:
    - GUI - set params
    - update material to: Ka, Kd, Ks
    - better creator function handling

- new features:
    - ray casting
    - use trimesh for mesh i/o?
    - Area lights - triangle, disc, sphere (attenuate contribution) -- random sampling on primitive (point_on?)
        - random point in triangle (https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle)
    - Disc primitive
    - parallelogram primitive? - planar quad from two vectors... easy hit and UV? [rectangle a sub-class]
        - intersection - https://math.stackexchange.com/questions/2461034/raytracing-a-parallelogram-ray-parallelogram-intersection
            or https://stackoverflow.com/questions/59128744/raytracing-ray-vs-parallelogram
            or rectangle: https://stackoverflow.com/questions/21114796/3d-ray-quad-intersection-test-in-java
        - point on - https://math.stackexchange.com/questions/3537762/random-point-in-a-triangle
    - profile / optimize
        - nuitka? numba? C routine for hits? (ctypes?)
        - NVidia Optix (https://developer.nvidia.com/optix & https://github.com/ozen/pyoptix)
        - Embree/pyembree (https://www.embree.org/ & http://embree.github.com)    
    - multi-processing
    - de-noising
    - more scenes and models (bunnies, dragons), sphereflake
    - quads
    - run batch / animation
    - solid textures & Perlin noise
         - https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/procedural-patterns-noise-part-1/creating-simple-1D-noise
         
         # use settable grid size (256 or 512 pts) -- test w/ 2D plots -- if power of two can use bit & instead of modulo
         # mask = 256-1,  b = a & MASK
         # random number (0..1) on grid points. LERP between them, LERP between last pt and first point for repeating.
         float smoothstepRemap(const float &a, const float &b, const float &t) 
            { 
                float tRemapSmoothstep = t * t * (3 - 2 * t); 
                return Mix(a, b, tRemapSmoothstep); 
            } 
         # or better: 6t^5−15t^4+10t^3
         # offset by phase (x+) and frequency (x*)
    - importance sampling
    - preview mode (https://github.com/snavely/pyrender ?)
    - OpenSurfaces (http://opensurfaces.cs.cornell.edu/)
    - reaction/diffusion textures
    - texture repeat/tiling/mirroring

Implemented Features:
    - Primitives: Spheres, Triangles, Planes, STL Files (triangle meshes)
    - Texture mapping: SolidColor, CheckerBoard, ImageTexture
    - Lighting: Point lights
    - Shadow rays for shadowing
    - BVH (bounding volume hieraerchy)
    - models: teapots, bunnies, etc.

Len Wanger -- 2020
