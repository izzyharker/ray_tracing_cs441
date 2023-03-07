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

__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void render_init(int nx, int ny, curandState *state) {
    // set up random values for pixels
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;
    int pixel_index = j*nx + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &state[pixel_index]);
}

// render
__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable ** world, curandState *state) {
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
        ray r = (*cam)->get_ray(u, v);
        col += color(r, world, &local_rand_state);
    }

    state[pixel_index] = local_rand_state;

    // calculate color
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);

    fb[pixel_index] = col;
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera ** cam, vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect) {
    float r = cos(M_PI/4);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0,0,-1), 0.5,
                               new lambertian(vec3(0.1, 0.2, 0.5)));
        d_list[1] = new sphere(vec3(0,-100.5,-1), 100,
                               new lambertian(vec3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(vec3(1,0,-1), 0.5,
                               new metal(vec3(0.8, 0.6, 0.2), 0.2));
        d_list[3] = new sphere(vec3(-1,0,-1), .5,
                               new dielectric(1.5));
        d_list[4] = new sphere(vec3(-1,0,-1), -0.45,
                               new dielectric(1.5));
        *d_world  = new hitable_list(d_list,5);
        *cam = new camera(lookfrom, lookat, vup, vfov, aspect);
    }
}

__global__ void free_world(hitable **d_list, hitable ** d_world, camera ** cam) {
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
    delete *cam;
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
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            f << ir << " " << ig << " " << ib << "\n";
        }
    }
    f.close();
}

int main() {
    int nx = 1850;
    int ny = 1000;
    int ns = 200;
    int tx = 16;
    int ty = 32;

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);
 
    // allocate fb = buffer for image
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // make world
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 4*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));

    camera **cam;
    checkCudaErrors(cudaMalloc((void **)&cam, sizeof(camera *)));

    // set up vectors for camera
    vec3 lookfrom(-2, 2, 1);
    vec3 lookat(0, 0, -1);
    vec3 vup(0, 1, 0);
    float vfov = 25;
    float aspect = float(nx)/float(ny);

    create_world<<<1,1>>>(d_list,d_world, cam, lookfrom, lookat, vup, vfov, aspect);


    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(nx/tx+1, ny/ty+1);
    dim3 threads(tx, ty);

    // init random values - for antialiasing
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);

    render<<<blocks, threads>>>(fb, nx, ny, ns, cam, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    write_image("out.ppm", fb, nx, ny);


    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world, cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();

    return 0;
}
