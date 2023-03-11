#ifndef LIGHT_H
#define LIGHT_H

#include "ray.h"
#include "hitable.h"

struct hit_record;

class spotlight {
    public:
        vec3 point;
        vec3 direction;
        float angle;

        // feather is the proportion of light at the edge of the spotlight - 
        // 1 = full light, sharp edge, 0 = no light
        // lerp based on angle
        float feather;

        __device__ vec3 hit(vec3 p, hitable ** world) {
            hit_record rec;
            ray d(p, point - p);
            if (!(*world)->hit(d, .001, FLT_MAX, rec)) {
                float amount;
                float ang = acos(dot(unit_vector(d.B), unit_vector(point - direction)));
                if (ang <= angle) {
                    float t = ang/angle;
                    amount = t*feather + (1-t);
                    return amount*vec3(1, 1, 1);
                }
                amount = .1;
                return amount*vec3(1, 1, 1);
            } 
            return vec3(.1, .1, .1);
        }

        __device__ spotlight(vec3 &p, vec3 &dir, float a, float f) {
            angle = (a*M_PI)/180;
            feather = f;
            point = p;
            direction = dir;
        }
};


#endif