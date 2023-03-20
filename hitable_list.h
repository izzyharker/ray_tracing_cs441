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
    rec.light_index = 2.0f;
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, tmin, tmax, rec)) {
            float c = min(1.0f, abs(dot(unit_vector(rec.normal), unit_vector(r.B))));
            if (rec.mat->get_id() != DIELECTRIC) return false;
            // if (c > 1.0) rec.light_index = 1.0f;
            // else rec.light_index = 0.5f;
            if (c >= M_PI/30 && c <= M_PI/15) {
                rec.light_index = (c-M_PI/30)*30/M_PI;
            }
            else rec.light_index = 1.5;
        }
    }
    return true;
}

#endif