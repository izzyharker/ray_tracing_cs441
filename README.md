I started this project with the goal of recreating the image they show in high school 
physics demonstrating how light refracts through a prism. To accomplish this, I wanted
to implement a ray tracing algorithm that was capable of tracking the color of the 
light as it refracted and split - more on the later.

Since I happen to have a Nvidia GPU, I decided to do this in CUDA, both because I was 
interested in learning about it and for efficiency's sake. Being able to leverage 
the parallelism of the GPU meant that rendering took much less time, which was very
helpful when I was testing new part of the code and it took several tries to create
the correct image.

I started the project by researching basic CUDA/ray tracing tutorials to find a starting
point, and stumbled upon this blog post on the Nvidia developers site:
https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/?tags=cuda&categories=

I spent the first 8 hours or so working through this tutorial and figuring out how all
the different pieces fit together, and how the implementations of the ray tracing equations
worked. This tutorial covers spheres, materials, and a camera.

The blog post referenced a series of tutorials by Peter Shirley (Ray Tracing in one Weekend,
Ray Tracing: the next week), and I found the second one very helpful in understanding 
a way to create hit methods for non-spheres. The book covers rectangles, so I spend around 3 
hours translating the implementation to CUDA and testing. 

The remaining 20ish hours were spent doing 2 things: light and triangles. 

To get triangles, I extrapolated from the rectangle code and did some googling on how to tell
if a point is in a triangle - it turns out that barycentric coordinates are good for this 
sort of thing. I probably spent about 5 hours making triangles and then around 5 more to make
a pyramid. (Math is hard)

Unfortunately for me, light is also hard. It is (relatively) easy to implement basic shadow rays
and a spotlight structure, but much harder to determine the behavior of light in a dielectric 
surface like glass. This took a lot of trial and error to figure out, and I don't know very 
much about light physics, so I was stumbling blindly most of the time.

Regardless, I think the final product (final_video.mp4) is still cool and demonstrates that
I was able to accomplish something like my project proposal.