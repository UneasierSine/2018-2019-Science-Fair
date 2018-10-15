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

struct div
{
	div() {}
	__host__ __device__ double operator() (double a, double b)
	{
		return a / b;
	}
};

struct pow
{
	pow() {}
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

vector<double> addMatTermsGpu(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVec(1);

	if (vec1.size() != vec2.size())
	{
		returnVec[0] = -1;
		return returnVec;
	}

	host_vector<double> h_v1 = vec1;
	host_vector<double> h_v2 = vec2;

	device_vector<double> d_rV(vec1.size());

	device_vector<double> d_v1 = h_v1;
	device_vector<double> d_v2 = h_v1;

	transform(d_v1.begin(), d_v1.end(), d_v2.begin, d_rV.begin(), add());

	for (int i = 0; i < d_rV.size(); i++)
	{
		returnVec[i] = d_rV[i];
	}

	return returnVec;
}

vector<double> subMatTermsGpu(vector<double> vec1, vector<double> vec2)
{

}

vector<double> mulMatTermsGpu(vector<double> vec1, vector<double> vec2)
{

}

vector<double> divMatTermsGpu(vector<double> vec1, vector<double> vec2)
{

}

vector<double> powMatTermsGpu(vector<double> vec1, vector<double> vec2)
{

}

vector<double> radMatTermsGpu(vector<double> vec1, vector<double> vec2)
{

}

double sumTermsGpu(vector<double> vector)
{

}

double dotProductGpu(vector<double> vec1, vector<double> vec2)
{

}