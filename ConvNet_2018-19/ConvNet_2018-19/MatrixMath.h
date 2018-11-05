#pragma once
#include <vector>
#include <random>
#include <stdlib.h>
#include "stdafx.h"
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <functional>

using namespace std;

//Basic operations on matrix indices
vector<double> addMat(vector<double> vec1, vector<double> vec2);
vector<double> subMat(vector<double> vec1, vector<double> vec2);
vector<double> mulMat(vector<double> vec1, vector<double> vec2);
vector<double> divMat(vector<double> vec1, vector<double> vec2);
vector<double> powMat(vector<double> vec1, vector<double> vec2);
vector<double> radMat(vector<double> vec1, vector<double> vec2);

//Matrix-specific operations
double sumTerms(vector<double> vector);
double dotProduct(vector<double> vec1, vector<double> vec2);

struct gaussianRandom
{
	double operator()(double x);
};

////GPU methods basic matrix index operations
//vector<double> addMatTermsGpu(vector<double> vec1, vector<double> vec2);
//vector<double> subMatTermsGpu(vector<double> vec1, vector<double> vec2);
//vector<double> mulMatTermsGpu(vector<double> vec1, vector<double> vec2);
//vector<double> divMatTermsGpu(vector<double> vec1, vector<double> vec2);
//vector<double> powMatTermsGpu(vector<double> vec1, vector<double> vec2);
//vector<double> radMatTermsGpu(vector<double> vec1, vector<double> vec2);
//
////GPU methods matrix-specific
//double sumTermsGpu(vector<double> vector);
//double dotProductGpu(vector<double> vec1, vector<double> vec2);