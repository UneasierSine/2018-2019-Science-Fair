// cuda_with_cpp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "gpu_print.h"
#include <iostream>

int main()
{
	gpu_print gpuObject;
	gpuObject.printGpu();

	return 0;
}

