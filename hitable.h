#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

class material;

struct hit_record
{
    // p = point of impact
    
    float t;
    vec3 p;
    vec3 normal;
    material *mat;
    float light_index;
};

class hitable  {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
        __device__ virtual bool hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const { return false; };
};

#endif