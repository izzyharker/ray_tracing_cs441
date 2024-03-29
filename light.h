#ifndef LIGHT_H
#define LIGHT_H

#include "ray.h"
#include "hitable.h"
#include "material.h"

struct hit_record;
enum MAT_TYPE;

class spotlight {
    public:
        vec3 point;
        vec3 direction;
        float angle;
        float intensity;
        float n;

        // feather is the proportion of light at the edge of the spotlight - 
        // 1 = full light, sharp edge, 0 = no light
        // lerp based on angle
        float feather;

        __device__ spotlight() {};

        __device__ vec3 hit(hitable ** world, hit_record & obj_rec, const ray& r) {
            hit_record rec;
            ray d(obj_rec.p, point - obj_rec.p);
            // if we can see the light
            if ((*world)->hit_light(d, 0.001, FLT_MAX, rec)) {
                n = rec.light_index;

                // this was segfaulting for some reason...
                // if (rec.mat->get_id() == DIELECTRIC) {
                //     n = dot(unit_vector(d.B), unit_vector(rec.normal));
                // }

                float amount;
                float ang = acos(dot(unit_vector(d.B), unit_vector(point - direction)));
                if (ang <= angle) {

                    // ?????????
                    // float shading = 0.0;
                    // float ks = 2;
                    // vec3 refl = unit_vector(2*dot(unit_vector(d.B), unit_vector(obj_rec.normal))*obj_rec.normal - d.B);
                    // vec3 view = unit_vector(r.B);
                    // shading = ks*max(0.0f, pow(abs(dot(refl, view)), 50.5));

                    float t = ang/angle;
                    amount = t*feather + (1-t);
                    if (n <= 1.0f) {
                        if (n <= .33f && n >= 0.0f) { // blue -> red
                            return 1.2f*intensity*amount*vec3(1*(1-3*n), .1, 1*(3*n));
                        }
                        else if (n < 0.66f && n >= 0.0f) { // green -> blue
                            return 1.2f*intensity*amount*vec3(.1, abs(1*(3*n-1)), 1*(2-3*n));
                        }
                        else if (n <= 1.0f && n >= 0.0f) { // red -> green
                            return 1.2f*intensity*amount*vec3(abs(1*(3*n-2)), 1*(3-3*n), .1);
                        }
                    }
                    else if (n < 2.0f) return intensity*amount*1.0f*vec3(1, 1, 1);
                    return intensity*amount*vec3(1, 1, 1);
                }
            }
            if (feather <= .1) {
                return intensity*feather*(vec3(1, 1, 1));
            } else return intensity*0.1f*(vec3(1, 1, 1));
        }

        __device__ spotlight(vec3 &p, vec3 &dir, float a, float f, float i) {
            angle = (a*M_PI)/180;
            feather = f;
            point = p;
            direction = dir;
            intensity = i;
            n = 2.0f;
        }
};

/*
point_light is a spotlight with a view angle of 360 degrees
*/
class point_light : public spotlight {
    public:
        vec3 point;
        vec3 direction;
        float angle;
        float intensity;

        // feather is the proportion of light at the edge of the spotlight - 
        // 1 = full light, sharp edge, 0 = no light
        // lerp based on angle
        float feather;
        float n;

        __device__ vec3 hit(hitable ** world, hit_record & obj_rec, const ray& r) {
            hit_record rec;
            ray d(obj_rec.p, point - obj_rec.p);

            // if we can see the light
            if ((*world)->hit_light(d, 0.001, FLT_MAX, rec)) {
                float amount;
                float ang = acos(dot(unit_vector(d.B), unit_vector(point - direction)));
                float shading = 0.0;

                // ???? shading just. doesnt work like its suposed to
                // float ks = 2;
                // vec3 refl = unit_vector(2*dot(unit_vector(d.B), unit_vector(obj_rec.normal))*obj_rec.normal - d.B);
                // vec3 view = unit_vector(r.B);
                // shading = ks*max(0.0f, pow(abs(dot(refl, view)), 50.5));

                float t = ang/angle;
                amount = t*feather + (1-t);
                return (intensity*amount + shading)*vec3(1, 1, 1);
            }
            // else...
            return 0.5f*vec3(1, 1, 1);
        }

        __device__ point_light(vec3 &p, vec3 &dir, float f, float i) {
            feather = f;
            point = p;
            direction = dir;
            intensity = i;
            n = 1.0f;
        }
};

#endif