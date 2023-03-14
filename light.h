#ifndef LIGHT_H
#define LIGHT_H

#include "ray.h"
#include "hitable.h"
#include "material.h"

struct hit_record;

class spotlight {
    public:
        vec3 point;
        vec3 direction;
        float angle;
        float intensity;

        // feather is the proportion of light at the edge of the spotlight - 
        // 1 = full light, sharp edge, 0 = no light
        // lerp based on angle
        float feather;

        __device__ vec3 hit(hitable ** world, hit_record & obj_rec, const ray& r) {
            hit_record rec;
            ray d(obj_rec.p, point - obj_rec.p);
            float shading;
            float ks = 1.2;
            float glass = 10;
            float metal = 30;
            vec3 refl = unit_vector(2*dot(d.B, obj_rec.normal)*obj_rec.normal - d.B);
            vec3 view = unit_vector(r.B);
            switch (obj_rec.mat->get_id()) {
                case LAMBERTIAN:
                    shading = 0.0;
                    break;
                case METAL:
                    shading = ks*max(0.0f, pow(max(0.0, dot(refl, view)), metal)); 
                    break;
                case DIELECTRIC:
                    shading = ks*max(0.0f, pow(max(0.0, dot(refl, view)), glass));
                    break;
                default:
                    shading = 0.0;
                    break;
            };
            // if we can see the light
            if ((*world)->hit_light(d, 0.001, FLT_MAX)) {
                float amount;
                float ang = acos(dot(unit_vector(d.B), unit_vector(point - direction)));
                if (ang <= angle) {
                    float t = ang/angle;
                    amount = t*feather + (1-t);
                    return (intensity*amount + shading)*vec3(1, 1, 1);
                }
            }
            // else...
            return (intensity*feather)*vec3(1, 1, 1);
        }

        __device__ spotlight(vec3 &p, vec3 &dir, float a, float f, float i) {
            angle = (a*M_PI)/180;
            feather = f;
            point = p;
            direction = dir;
            intensity = i;
        }
};


#endif