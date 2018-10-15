#include <iostream>
#include "MatrixMath.h"
#include "MatrixMath.cuh"

using namespace thrust;

struct add
{
	add() {}
	__host__ __device__ double operator()(double a, double b)
	{
		return a + b;
	}
};

struct sub
{
	sub() {}
	__host__ __device__ double operator()(double a, double b)
	{
		return a - b;
	}
};

struct mul
{
	mul() {}
	__host__ __device__ double operator() (double a, double b)
	{
		return a * b;
	}
};

struct division
{
	division() {}
	__host__ __device__ double operator() (double a, double b)
	{
		return a / b;
	}
};

struct power
{
	power() {}
	__host__ __device__ double operator() (double a, double b)
	{
		return std::pow(a, b);
	}
};

struct rad
{
	rad() {}
	__host__ __device__ double operator() (double a, double b)
	{
		return pow(a, 1/b);
	}
};

vector<double> deviceToCppVec(device_vector<double> dV, vector<double> vec)
{
	vector<double> returnVec;
	for (double val : dV)
	{
		returnVec.push_back(val);
	}
	return returnVec;
}

vector<double> addMatTermsGpu(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVec(1);

	if (vec1.size() != vec2.size())
	{
		returnVec[0] = -1;
		return returnVec;
	}

	device_vector<double> d_rV(vec1.size());

	device_vector<double> d_v1 = vec1;
	device_vector<double> d_v2 = vec2;

	transform(d_v1.begin(), d_v1.end(), d_v2.begin, d_rV.begin(), add());

	deviceToCppVec(d_rV, returnVec);
	return returnVec;
}

vector<double> subMatTermsGpu(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVec(1);

	if (vec1.size() != vec2.size())
	{
		returnVec[0] = -1;
		return returnVec;
	}

	device_vector<double> d_rV(vec1.size());

	device_vector<double> d_v1 = vec1;
	device_vector<double> d_v2 = vec2;

	transform(d_v1.begin(), d_v1.end(), d_v2.begin, d_rV.begin(), sub());

	deviceToCppVec(d_rV, returnVec);
	return returnVec;
}

vector<double> mulMatTermsGpu(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVec(1);

	if (vec1.size() != vec2.size())
	{
		returnVec[0] = -1;
		return returnVec;
	}

	device_vector<double> d_rV(vec1.size());

	device_vector<double> d_v1 = vec1;
	device_vector<double> d_v2 = vec2;

	transform(d_v1.begin(), d_v1.end(), d_v2.begin, d_rV.begin(), mul());

	deviceToCppVec(d_rV, returnVec);
	return returnVec;
}

vector<double> divMatTermsGpu(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVec(1);

	if (vec1.size() != vec2.size())
	{
		returnVec[0] = -1;
		return returnVec;
	}

	device_vector<double> d_rV(vec1.size());

	device_vector<double> d_v1 = vec1;
	device_vector<double> d_v2 = vec2;

	transform(d_v1.begin(), d_v1.end(), d_v2.begin, d_rV.begin(), division());

	deviceToCppVec(d_rV, returnVec);
	return returnVec;
}

vector<double> powMatTermsGpu(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVec(1);

	if (vec1.size() != vec2.size())
	{
		returnVec[0] = -1;
		return returnVec;
	}

	device_vector<double> d_rV(vec1.size());

	device_vector<double> d_v1 = vec1;
	device_vector<double> d_v2 = vec2;

	transform(d_v1.begin(), d_v1.end(), d_v2.begin, d_rV.begin(), power());

	deviceToCppVec(d_rV, returnVec);
	return returnVec;
}

vector<double> radMatTermsGpu(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVec(1);

	if (vec1.size() != vec2.size())
	{
		returnVec[0] = -1;
		return returnVec;
	}

	device_vector<double> d_rV(vec1.size());

	device_vector<double> d_v1 = vec1;
	device_vector<double> d_v2 = vec2;

	transform(d_v1.begin(), d_v1.end(), d_v2.begin, d_rV.begin(), rad());

	deviceToCppVec(d_rV, returnVec);
	return returnVec;
}

double sumTermsGpu(vector<double> vec)
{
	device_vector<double> d_v = vec;
	return reduce(d_v.begin(), d_v.end());
}

double dotProductGpu(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVec(1);

	if (vec1.size() != vec2.size())
	{
		return -1;
	}

	device_vector<double> d_rV(vec1.size());

	device_vector<double> d_v1 = vec1;
	device_vector<double> d_v2 = vec2;

	transform(d_v1.begin(), d_v1.end(), d_v2.begin, d_rV.begin(), mul());
	return reduce(d_rV.begin(), d_rV.end());
}