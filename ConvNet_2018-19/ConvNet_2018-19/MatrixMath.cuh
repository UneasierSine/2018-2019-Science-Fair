#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

__global__ void add(double *a, double *b);
__global__ void sub(double *a, double *b);
__global__ void mul(double *a, double *b);
__global__ void div(double *a, double *b);
__global__ void pow(double *a, double *b);
__global__ void rad(double *a, double *b);

__global__ void sumTerms(double *a);
__global__ void dotProduct(double *a, double *b);