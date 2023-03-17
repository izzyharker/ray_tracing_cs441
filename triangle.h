#ifndef TRIH
#define TRIH

#include "hitable.h"
#include "ray.h"
#include "material.h"
#include "rotate.h"

class material;

class xy_triangle : public hitable {
    public:
        __device__ xy_triangle(float x0, float x1, float x2, float y0, float y1, float y2, float k, material *mat) : x0(x0), x1(x1), x2(x2), y0(y0), y1(y1), y2(y2), k(k), mat(mat) {};
        __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record &rec) const;
        __device__ virtual bool hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const;

        float x0, x1, x2, y0, y1, y2, k;
        material *mat;
};

__device__ bool xy_triangle::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    float t = (k-r.origin().z()) / r.direction().z();
    if (t < t0 || t > t1) return false;

    float x = r.origin().x() + t*r.direction().x();
    float y = r.origin().y() + t*r.direction().y();

    // stack exchange post on barycentric coordinates
    float area = 0.5 *(-x0*x2 + y0*(-x1 + x2) + x0*(y1 - y2) + x1*y2);
    float s = s = 1/(2*area)*(y0*x2 - x0*y2 + (y2 - y0)*x + (x0 - x2)*y);
    float u = 1/(2*area)*(x0*y1 - y0*x1 + (y0 - y1)*x + (x1 - x0)*y);
    if (s < 0 || u < 0 || s + u > 1) return false;
    rec.t = t;
    rec.p = r.point_at_parameter(t);
    rec.mat = mat;
    rec.normal = vec3(0, 0, 1);
    return true;
}


__device__ bool xy_triangle::hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const {
    return false;
}

class xz_triangle : public hitable {
    public:
        __device__ xz_triangle(float x0, float x1, float x2, float z0, float z1, float z2, float k, material *mat) : x0(x0), x1(x1), x2(x2), z0(z0), z1(z1), z2(z2), k(k), mat(mat) {};
        __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record &rec) const;
        __device__ virtual bool hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const;

        float x0, x1, x2, z0, z1, z2, k;
        material *mat;
};

__device__ bool xz_triangle::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    float t = (k-r.origin().y()) / r.direction().y();
    if (t < t0 || t > t1) return false;

    float x = r.origin().x() + t*r.direction().x();
    float z = r.origin().z() + t*r.direction().z();

    // stack exchange post on barzcentric coordinates
    float area = 0.5 *(-x0*x2 + z0*(-x1 + x2) + x0*(z1 - z2) + x1*z2);
    float s = s = 1/(2*area)*(z0*x2 - x0*z2 + (z2 - z0)*x + (x0 - x2)*z);
    float u = 1/(2*area)*(x0*z1 - z0*x1 + (z0 - z1)*x + (x1 - x0)*z);
    if (s < 0 || u < 0 || s + u > 1) return false;
    rec.t = t;
    rec.p = r.point_at_parameter(t);
    rec.mat = mat;
    rec.normal = vec3(0, 0, 1);
    return true;
}


__device__ bool xz_triangle::hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const {
    return false;
}

class yz_triangle : public hitable {
    public:
        __device__ yz_triangle(float z0, float z1, float z2, float y0, float y1, float y2, float k, material *mat) : z0(z0), z1(z1), z2(z2), y0(y0), y1(y1), y2(y2), k(k), mat(mat) {};
        __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record &rec) const;
        __device__ virtual bool hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const;

        float z0, z1, z2, y0, y1, y2, k;
        material *mat;
};

__device__ bool yz_triangle::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    float t = (k-r.origin().x()) / r.direction().x();
    if (t < t0 || t > t1) return false;

    float z = r.origin().z() + t*r.direction().z();
    float y = r.origin().y() + t*r.direction().y();

    // stack exchange post on barycentric coordinates
    float area = 0.5 *(-z0*z2 + y0*(-z1 + z2) + z0*(y1 - y2) + z1*y2);
    float s = s = 1/(2*area)*(y0*z2 - z0*y2 + (y2 - y0)*z + (z0 - z2)*y);
    float u = 1/(2*area)*(z0*y1 - y0*z1 + (y0 - y1)*z + (z1 - z0)*y);
    if (s < 0 || u < 0 || s + u > 1) return false;
    rec.t = t;
    rec.p = r.point_at_parameter(t);
    rec.mat = mat;
    rec.normal = vec3(0, 0, 1);
    return true;
}


__device__ bool yz_triangle::hit_light(const ray& r, float tmin, float tmax, hit_record &rec) const {
    return false;
}

class pyramid : public hitable {
    public:
        __device__ pyramid(vec3 p0, vec3 p1, float height, material *m);
        __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record &rec) const;
        __device__ virtual bool hit_light(const ray& r, float t0, float t1, hit_record &rec) const { return false; };

        hitable_list * list;
        // ???
};

__device__ pyramid::pyramid(vec3 p0, vec3 p1, float height, material *m) {
    // finish this
    hitable ** sides = new hitable*[5];
    float x_avg = (p1.x() - p0.x()) / 2;
    float z_avg = (p1.z() - p0.z()) / 2;

    float x_h = sqrt(height*height + x_avg*x_avg);
    float z_h = sqrt(height*height + z_avg*z_avg);

    float xy_r = M_PI/2 - atan(height/x_avg);
    float yz_r = M_PI/2 - atan(height/z_avg);

    sides[0] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), m);
    sides[1] = new translate(new z_rotate(new translate(new xy_triangle(p0.x(), p1.x(), p0.x() + x_avg, p0.y(), p1.y(), p0.y() + x_h, p1.z(), m), -vec3(p0.x(), p0.y(), p1.z())), xy_r), vec3(p0.x(), p0.y(), p1.z()));
    sides[2] = new translate(new x_rotate(new translate(new yz_triangle(p1.z(), p0.z() + z_avg, p0.z(), p1.y(), (p0.y() + z_h), p0.y(), p1.x(), m), -vec3(p1.x(), p0.y(), p0.z())), -yz_r), vec3(p1.x(), p0.y(), p0.z()));
    sides[3] = new translate(new z_rotate(new translate(new xy_triangle(p0.x(), p1.x(), p0.x() + x_avg, p0.y(), p1.y(), p0.y() + z_h, p0.z(), m), -vec3(p0.x(), p0.y(), p0.z())), -xy_r), vec3(p0.x(), p0.y(), p0.z()));
    sides[4] = new translate(new x_rotate(new translate(new yz_triangle(p1.z(), p0.z() + z_avg, p0.z(), p1.y(), p0.y() + z_h, p0.y(), p0.x(), m), -vec3(p0.x(), p0.y(), p0.z())), yz_r), vec3(p0.x(), p0.y(), p0.z()));

    list = new hitable_list(sides, 5);
}

__device__ bool pyramid::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    return list->hit(r, t0, t1, rec);
}

#endif