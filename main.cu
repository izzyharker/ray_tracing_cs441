#include <iostream>
#include <time.h>
#include <fstream>
#include <float.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "light.h"
#include "rect.h"
#include "triangle.h"
#include "rotate.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


// the actual ray tracing part of it - attenuating color
__device__ vec3 color(const ray& r, hitable **world, spotlight ** light, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            vec3 l = (*light)->hit(world, rec, cur_ray);
            if(rec.mat->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation*l;
                cur_ray = scattered;
            }
        }
        else {
            if (i == 0) return vec3(0, 0, 0);
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation*c;
        }
    }
    return cur_attenuation; // exceeded recursion
}

// set up random
__global__ void render_init(int nx, int ny, curandState *state) {
    // set up random values for pixels
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;
    int pixel_index = j*nx + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &state[pixel_index]);
}

// render - this one is the main kernel
__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable ** world, spotlight **light, curandState *state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;

    // antialiasing - send 4 random rays through each pixel on the screen and average their colors
    curandState local_rand_state = state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, light, &local_rand_state);
    }

    state[pixel_index] = local_rand_state;

    // calculate color
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);

    fb[pixel_index] = col;
}

__global__ void create_world(camera ** cam, spotlight **light, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // // set up vectors for camera
        vec3 lookfrom(600, 278, -600);
        vec3 lookat(278, 278, 278);
        vec3 vup(0, 1, 0);
        float vfov = 100;
        float aspect = float(nx)/float(ny);
        float aperture = 0.0;
        float focus = (lookfrom - lookat).length();

        *cam = new camera(lookfrom, lookat, vup, vfov, aspect, aperture, focus);

        *light = new spotlight(vec3(600, 555, 278), vec3(278, 400, 278), 45.0f, 0.5f, 2.5f);
        // *light = new spotlight(vec3(2, 2, 1), lookat, 45, .2, 1);
        // *light = new spotlight(lookfrom, lookat, 30.0f, 0.2f, 1.2f);
    }
}

__global__ void create_pyramid(hitable **d_list, hitable **d_world, int num, float degrees) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        material *m = new metal(vec3(.7, .5, .3), 0.0f);
        material *d = new dielectric(1.67);
        d_list[0] = new yz_rect(-400, 1000, -800, 1000, -200, new lambertian(vec3(.1, .1, .1)));
        d_list[1] = new xz_rect(-200, 1000, -400, 1000, -800, new lambertian(vec3(.1, .1, .1)));
        d_list[2] = new xy_rect(-200, 1000, -800, 1000, 1000, new lambertian(vec3(.1, .1, .1)));
        d_list[3] =  new translate(new y_rotate(new translate(new pyramid(vec3(0, 0, 0), vec3(555, 0, 555), 555, d), -vec3(555.0f/2, 0, 555.0f/2)), degrees, false), vec3(555.0f/2, 0, 555.0f/2));
        // d_list[3] = new translate(new y_rotate(new translate(new yz_triangle(0, 555, 555.0/2, 0, 0, 555, 555.0/2, d), -vec3(555.0/2, 0, 555.0/2)), 0, false), vec3(555.0/2, 0, 555.0/2));

        *d_world  = new hitable_list(d_list,num);
    }
}

__global__ void free_world(hitable **d_list, hitable ** d_world, int num) {
    for (int i = 0; i < num; i++) {
        delete *(d_list + i);
    }
    delete *d_world;
}

__global__ void free_cam(camera **cam, spotlight** light) {
    delete *cam;
    delete *light;
}

void write_image(std::string filename, vec3 *fb, int nx, int ny) {
    std::ofstream f;
    f.open(filename);
    f << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            float r = fb[pixel_index][0];
            float g = fb[pixel_index][1];
            float b = fb[pixel_index][2];
            int ir = max(0, min(255, int(255.99*r)));
            int ig = max(0, min(255, int(255.99*g)));
            int ib = max(0, min(255, int(255.99*b)));
            f << ir << " " << ig << " " << ib << "\n";
        }
    }
    f.close();
}

int main() {
    #ifdef HIGHDEF
    int nx = 1850;
    int ny = 1000;
    int ns = 200;
    #else
    int nx = 600;
    int ny = 300;
    int ns = 10;
    #endif
    int tx = 16;
    int ty = 32;

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);
 
    // allocate fb = buffer for image
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // make world
    hitable **d_list;
    int num_hitables = 4;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));

    camera **cam;
    checkCudaErrors(cudaMalloc((void **)&cam, sizeof(camera *)));

    spotlight **light;
    checkCudaErrors(cudaMalloc((void **)&light, sizeof(spotlight *)));

    create_world<<<1,1>>>(cam, light, nx, ny);


    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(nx/tx+1, ny/ty+1);
    dim3 threads(tx, ty);

    // init random values - for antialiasing
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);

    #ifdef IMAGE
    int i = 0;
    #endif
    // in for loop 
    #ifndef IMAGE
    for (int i = 0; i < 240; i++) {
    #endif
        float angle = float(i)*1.5f;

        create_pyramid<<<1, 1>>>(d_list, d_world, num_hitables, angle);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        render<<<blocks, threads>>>(fb, nx, ny, ns, cam, d_world, light, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        char filename[25];
        #ifndef IMAGE
        sprintf(filename, "video/frame%04d.ppm", i);
        #else
        sprintf(filename, "frame%04d.ppm", i);
        #endif
        write_image(filename, fb, nx, ny);

        // if (i % 20 == 0) std::cout << "wrote frame " << i << std::endl;

        free_world<<<1, 1>>>(d_list, d_world, num_hitables);
        checkCudaErrors(cudaDeviceSynchronize());
        // end for loop
    #ifndef IMAGE
    }
    #endif

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_cam<<<1,1>>>(cam, light);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(light));
    checkCudaErrors(cudaFree(fb));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();

    return 0;
}