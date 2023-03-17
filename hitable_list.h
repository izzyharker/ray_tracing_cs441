#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"
#include "material.h"

class material;

class hitable_list: public hitable  {
    public:
        __device__ hitable_list() {}
        __device__ hitable_list(hitable **l, int n) {list = l; list_size = n; }
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        __device__ bool hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const;
        hitable **list;
        int list_size;
};

__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
}

__device__ bool hitable_list::hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const {
    hit_record hit_rec;
    rec.light_index = 1.0f;
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, tmin, tmax, hit_rec)) {
            if (hit_rec.mat->get_id() != DIELECTRIC) return false;
            else {
                rec.light_index = abs(dot(unit_vector(hit_rec.normal), unit_vector(r.B)));
                rec.mat = hit_rec.mat;
                rec.p = hit_rec.p;
                rec.t = hit_rec.t;
                rec.normal = hit_rec.normal;
            }
        }
    }
    return true;
}

#endif