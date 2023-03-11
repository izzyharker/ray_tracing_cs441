#ifndef RECTH
#define RECTH

#include "hitable.h"

class rect: public hitable  {
    public:
        __device__ rect() {}
        __device__ rect(vec3 norm, float r, material *m) : normal(norm), radius(r), mat_ptr(m)  {};
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        vec3 normal;
        float radius;
        material *mat_ptr;
};

__device__ bool rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    // TODO

    return false;
}


#endif