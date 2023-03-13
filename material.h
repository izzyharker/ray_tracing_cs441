#ifndef MATERIALH
#define MATERIALH

#include <curand_kernel.h>
#include "ray.h"
#include "hitable.h"

struct hit_record;

enum MAT_TYPE {
    METAL,
    DIELECTRIC,
    LAMBERTIAN,
    NONE
};

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - 2.0f*dot(v,n)*n;
}

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f + ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow(1 - cosine, 5.0f);
}

__device__ bool refract(const vec3& v, const vec3 & n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float disc = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (disc > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(disc);
        return true;
    }
    else return false;
}

class material  {
    public:
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const = 0;

    MAT_TYPE id;
    __device__ virtual MAT_TYPE get_id() { return NONE; }
};

class lambertian : public material {
    public:
        __device__ lambertian(const vec3& a) : albedo(a), id(LAMBERTIAN) {}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
             vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
             scattered = ray(rec.p, target-rec.p);
             attenuation = albedo;

             return true;
        }

        MAT_TYPE id;
        __device__ virtual MAT_TYPE get_id() { return id; }

        vec3 albedo;
};

class metal : public material {
    public:
        __device__ metal(const vec3& a, float f) : albedo(a), id(METAL) { if (f < 1) fuzz = f; else fuzz = 1; }
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0.0f);
        }

        __device__ virtual MAT_TYPE get_id() { return id; }

        MAT_TYPE id;
        vec3 albedo;
        float fuzz;
};

class dielectric : public material {
    public:
        __device__ dielectric(float ri) : ref_idx(ri), id(DIELECTRIC) {}
        __device__ virtual bool scatter(const ray & r_in, const hit_record & rec, vec3 & attenuate, ray & scattered, curandState *local_rand_state) const {
            vec3 out_normal;
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            float ni_over_nt;

            // (X) for debugging purposes - kill the blue channel
            attenuate = vec3(1.0, 1.0, 1.0);
            vec3 refracted;
            float reflect_prob;
            float cosine;
            if (dot(r_in.direction(), rec.normal) > 0) {
                out_normal = -rec.normal;
                ni_over_nt = ref_idx;
                cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
                cosine = sqrt(1.0f - ref_idx*ref_idx*(1-cosine*cosine));
            }
            else {
                out_normal = rec.normal;
                ni_over_nt = 1.0 / ref_idx;
                cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
            }

            if (refract(r_in.direction(), out_normal, ni_over_nt, refracted)) {
                reflect_prob = schlick(cosine, ref_idx);
            }
            else {
                scattered = ray(rec.p, reflected);
                reflect_prob = 1.0;
            }
            if (curand_uniform(local_rand_state) < reflect_prob) {
                scattered = ray(rec.p, reflected);
            }
            else {
                scattered = ray(rec.p, refracted);
            }
            return true;
        }

        __device__ virtual MAT_TYPE get_id() { return id; }

        MAT_TYPE id;
        float ref_idx;
};


#endif