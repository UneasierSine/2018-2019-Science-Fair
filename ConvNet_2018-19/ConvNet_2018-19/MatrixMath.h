#pragma once
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
vector<double> powMat(vector<double> vec, double x);
vector<double> radMat(vector<double> vec1, vector<double> vec2);

//Matrix-specific operations
double sumTerms(vector<double> vector);
double dotProduct(vector<double> vec1, vector<double> vec2);

struct gaussianRandom
{
	double operator()(double x);
};