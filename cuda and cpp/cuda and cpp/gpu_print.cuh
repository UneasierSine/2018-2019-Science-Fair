#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

__global__ void hello(char *a, int *b);