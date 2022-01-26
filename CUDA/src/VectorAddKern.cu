#include <stdio.h>
#include <math.h>

typedef struct {
    float r;
    float i;
} comp_num;

__device__ void print_comp(comp_num* c);
__device__ comp_num add(comp_num c1, comp_num c2, comp_num* r);
__device__ comp_num mult(comp_num c1, comp_num c2, comp_num* r);
__device__ float absSq(comp_num c);

__device__ float scale_between(float num, float min, float max, float newMin, float newMax);


extern "C"
__global__ void mandelbrot(int n, int maxIt, int w, int h, float minX, float minY, float maxX, float maxY, float offsetX, float offsetY, float* it) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n)
    {
        int x = i % w;
        int y = (i - x) / w;
        comp_num c, p;
        c.r = 0;
        c.i = 0;
        p.r = scale_between(x + offsetX, 0, 800, minX, maxX);
        p.i = scale_between(y + offsetY, 0, 800, minY, maxY);

        int n = 0;
        while(n < maxIt && absSq(c) < 4)
        {
            comp_num new_c;
            mult(c, c, &new_c);
            c = new_c;
            add(c, p, &new_c);
            c = new_c;

            n++;
        }

        it[i] = n + 1 - logf(log2f(sqrtf(absSq(c))));
    }
}

__device__ void print_comp(comp_num* c)
{
    printf("%.2f + %.2fi\n", c->r, c->i);
}

__device__ comp_num add(comp_num c1, comp_num c2, comp_num* r)
{
    r->r = c1.r + c2.r;
    r->i = c1.i + c2.i;
}

__device__ comp_num mult(comp_num c1, comp_num c2, comp_num* r)
{
    r->r = c1.r * c2.r - c1.i * c2.i;
    r->i = c1.r * c2.i + c1.i * c2.r;
}

__device__ float absSq(comp_num c)
{
    return c.r * c.r + c.i * c.i;
}

__device__ float scale_between(float num, float min, float max, float newMin, float newMax)
{
    return ((newMax - newMin)*(num - min))/(max - min) + newMin;
}