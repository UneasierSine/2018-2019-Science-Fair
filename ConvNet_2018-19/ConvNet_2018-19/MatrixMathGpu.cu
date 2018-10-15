#include "MatrixMath.h"
#include "MatrixMath.cuh"
#include <vector>

vector<double> addMatTermsGpu(vector<double> vec1, vector<double> vec2)
{
	device_vector<double> v1 = vec1;
	device_vector<double> v2 = vec2;
	device_vector<double> rV(1);
	thrust::transform(v1.begin(), v1.end(), v2.begin(), rV.begin(), thrust::plus<double>());
	
	vector<double> vec(1);
	for (double val : rV) vec.push_back(val);
	return vec;
}