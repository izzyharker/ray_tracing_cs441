#ifndef RECTH
#define RECTH

#include "hitable.h"
#include "ray.h"

class xy_rect: public hitable  {
    public:
        __device__ xy_rect(float x0, float x1, float y0, float y1, float k, material *mat) : x0(x0), x1(x1), y0(y0), y1(y1), k(k), m(mat) {};
        __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record &rec) const;
        __device__ virtual bool hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const;


        material *m;
        float x0, x1, y0, y1, k;
};

__device__ bool xy_rect::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    float t = (k-r.origin().z()) / r.direction().z();
    if (t < t0 || t > t1) return false;

    float x = r.origin().x() + t*r.direction().x();
    float y = r.origin().y() + t*r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1) return false;
    rec.t = t;
    rec.mat = m;
    rec.p = r.point_at_parameter(t);
    rec.normal = vec3(0, 0, 1);
    return true;
}


__device__ bool xy_rect::hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const {
    return false;
}


class xz_rect: public hitable  {
    public:
        __device__ xz_rect(float x0, float x1, float z0, float z1, float k, material *mat) : x0(x0), x1(x1), z0(z0), z1(z1), k(k), m(mat) {};
        __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record &rec) const;
        __device__ virtual bool hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const;


        material *m;
        float x0, x1, z0, z1, k;
};

__device__ bool xz_rect::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    float t = (k-r.origin().y()) / r.direction().y();
    if (t < t0 || t > t1) return false;

    float x = r.origin().x() + t*r.direction().x();
    float z = r.origin().z() + t*r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1) return false;
    rec.t = t;
    rec.mat = m;
    rec.p = r.point_at_parameter(t);
    rec.normal = vec3(0, 1, 0);
    return true;
}


__device__ bool xz_rect::hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const {
    return false;
}

class yz_rect: public hitable  {
    public:
        __device__ yz_rect(float z0, float z1, float y0, float y1, float k, material *mat) : z0(z0), z1(z1), y0(y0), y1(y1), k(k), m(mat) {};
        __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record &rec) const;
        __device__ virtual bool hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const;


        material *m;
        float z0, z1, y0, y1, k;
};

__device__ bool yz_rect::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    float t = (k-r.origin().x()) / r.direction().x();
    if (t < t0 || t > t1) return false;

    float z = r.origin().z() + t*r.direction().z();
    float y = r.origin().y() + t*r.direction().y();
    if (z < z0 || z > z1 || y < y0 || y > y1) return false;
    rec.t = t;
    rec.mat = m;
    rec.p = r.point_at_parameter(t);
    rec.normal = vec3(1, 0, 0);
    return true;
}


__device__ bool yz_rect::hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const {
    return false;
}

class flip_normals : public hitable {
    public:
        __device__ flip_normals(hitable *p) : ptr(p) {};
        __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record &rec) const {
            if (ptr->hit(r, t0, t1, rec)) {
                rec.normal = -rec.normal;
                return true;
            }
            return false;
        }
        __device__ virtual bool hit_light(const ray& r, float t0, float t1, hit_record &rec) const;

        hitable *ptr;
};

__device__ bool flip_normals::hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const {
    return false;
}

class box : public hitable {
    public:
        __device__ box() {};
        __device__ box(const vec3& p0, const vec3& p1, material *ptr);
        __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record &rec) const;
        __device__ virtual bool hit_light(const ray& r, float t0, float t1, hit_record &rec) const;

        vec3 pmin, pmax;
        hitable *list_ptr;
};

__device__ box::box(const vec3& p0, const vec3& p1, material *ptr) {
    pmin = p0;
    pmax = p1;
    hitable **list = new hitable*[6];
    list[0] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr);
    list[1] = new flip_normals(new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));
    list[2] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr);
    list[3] = new flip_normals(new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));
    list[4] = new yz_rect(p0.z(), p1.z(), p0.y(), p1.y(), p1.x(), ptr);
    list[5] = new flip_normals(new yz_rect(p0.z(), p1.z(), p0.y(), p1.y(), p0.x(), ptr));
    list_ptr = new hitable_list(list, 6);
}

__device__ bool box::hit(const ray& r, float t0, float t1, hit_record &rec) const {
    return list_ptr->hit(r, t0, t1, rec);
};

__device__ bool box::hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const {
    return false;
}

#endif