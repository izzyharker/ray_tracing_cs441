#ifndef ROTATEH
#define ROTATEH

#include "hitable.h"
#include "ray.h"


/*
rotate a hitable object around the respective axis (x, y, z) - actually move the input rays,
not the objects
*/
class x_rotate : public hitable {
    public:
        __device__ x_rotate(hitable *p, float angle, bool r = true) : ptr(p) {
            float radians;
            if (!r) {
                radians = (M_PI / 180.f) * angle;
            }
            else radians = angle;
            sin_theta = sin(radians);
            cos_theta = cos(radians);
        };

        __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record & rec) const;
        __device__ virtual bool hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const { return false; };

        hitable *ptr;
        float sin_theta;
        float cos_theta;
};

__device__ bool x_rotate::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    vec3 origin = r.origin();
    vec3 direction = r.direction();
    origin[0] = cos_theta*r.origin()[0] - sin_theta*r.origin()[1];
    origin[1] = sin_theta*r.origin()[0] + cos_theta*r.origin()[1];
    direction[0] = cos_theta*r.direction()[0] - sin_theta*r.direction()[1];
    direction[1] = sin_theta*r.direction()[0] + cos_theta*r.direction()[1];
    ray rotated_r(origin, direction);
    if (ptr->hit(rotated_r, t0, t1, rec)) {
        vec3 p = rec.p;
        vec3 normal = rec.normal;
        p[0] = cos_theta*rec.p[0] + sin_theta*rec.p[1];
        p[1] = -sin_theta*rec.p[0] + cos_theta*rec.p[1];
        normal[0] = cos_theta*rec.normal[0] + sin_theta*rec.normal[1];
        normal[1] = -sin_theta*rec.normal[0] + cos_theta*rec.normal[1];
        rec.p = p;
        rec.normal = normal;
        return true;
    }
    return false;
}

class y_rotate : public hitable {
    public:
        __device__ y_rotate(hitable *p, float angle, bool r = true) : ptr(p) {
            float radians;
            if (!r) {
                radians = (M_PI / 180.f) * angle;
            }
            else radians = angle;
            sin_theta = sin(radians);
            cos_theta = cos(radians);
        };

        __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record & rec) const;
        __device__ virtual bool hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const { return false; };

        hitable *ptr;
        float sin_theta;
        float cos_theta;
};

__device__ bool y_rotate::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    vec3 origin = r.origin();
    vec3 direction = r.direction();
    origin[0] = cos_theta*r.origin()[0] - sin_theta*r.origin()[2];
    origin[2] = sin_theta*r.origin()[0] + cos_theta*r.origin()[2];
    direction[0] = cos_theta*r.direction()[0] - sin_theta*r.direction()[2];
    direction[2] = sin_theta*r.direction()[0] + cos_theta*r.direction()[2];
    ray rotated_r(origin, direction);
    if (ptr->hit(rotated_r, t0, t1, rec)) {
        vec3 p = rec.p;
        vec3 normal = rec.normal;
        p[0] = cos_theta*rec.p[0] + sin_theta*rec.p[2];
        p[2] = -sin_theta*rec.p[0] + cos_theta*rec.p[2];
        normal[0] = cos_theta*rec.normal[0] + sin_theta*rec.normal[2];
        normal[2] = -sin_theta*rec.normal[0] + cos_theta*rec.normal[2];
        rec.p = p;
        rec.normal = normal;
        return true;
    }
    return false;
}

class z_rotate : public hitable {
    public:
        __device__ z_rotate(hitable *p, float angle, bool r = true) : ptr(p) {
            float radians;
            if (!r) {
                radians = (M_PI / 180.f) * angle;
            }
            else radians = angle;

            sin_theta = sin(radians);
            cos_theta = cos(radians);
        };

        __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record & rec) const;
        __device__ virtual bool hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const { return false; };

        hitable *ptr;
        float sin_theta;
        float cos_theta;
};

__device__ bool z_rotate::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    vec3 origin = r.origin();
    vec3 direction = r.direction();
    origin[1] = cos_theta*r.origin()[1] - sin_theta*r.origin()[2];
    origin[2] = sin_theta*r.origin()[1] + cos_theta*r.origin()[2];
    direction[1] = cos_theta*r.direction()[1] - sin_theta*r.direction()[2];
    direction[2] = sin_theta*r.direction()[1] + cos_theta*r.direction()[2];
    ray rotated_r(origin, direction);
    if (ptr->hit(rotated_r, t0, t1, rec)) {
        vec3 p = rec.p;
        vec3 normal = rec.normal;
        p[1] = cos_theta*rec.p[1] + sin_theta*rec.p[2];
        p[2] = -sin_theta*rec.p[1] + cos_theta*rec.p[2];
        normal[1] = cos_theta*rec.normal[1] + sin_theta*rec.normal[2];
        normal[2] = -sin_theta*rec.normal[1] + cos_theta*rec.normal[2];
        rec.p = p;
        rec.normal = normal;
        return true;
    }
    return false;
}


/*
translate an object
*/
class translate : public hitable {
    public:
        __device__ translate(hitable *p, const vec3 off) : ptr(p), offset(off) {};
        __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record& rec) const;
        __device__ virtual bool hit_light(const ray& r, float t0, float t1, hit_record &rec) const { return false; }

        hitable *ptr;
        vec3 offset;
};

__device__ bool translate::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    ray moved_r(r.origin() - offset, r.direction());
    if (ptr->hit(moved_r, t0, t1, rec)) {
        rec.p += offset;
        return true;
    }
    else return false;
}

#endif