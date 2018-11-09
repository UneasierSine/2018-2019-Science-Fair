#pragma once
#include <math.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <functional>
#include "MatrixMath.h"

using namespace std;

vector<double> squaredError(vector<double> actual, vector<double> predicted);
vector<double> squaredLogError(vector<double> actual, vector<double> predicted);
vector<double> l1Loss(vector<double> actual, vector<double> predicted);

vector<double> squaredErrorDeriv(vector<double> actual, vector<double> predicted);
vector<double> squaredLogErrorDeriv(vector<double> actual, vector<double> predicted);
vector<double> l1LossDeriv(vector<double> actual, vector<double> predicted);

struct errorFunction
{
	vector<double> (*function)(vector<double>, vector<double>);
	errorFunction(std::string typeFunction);
	vector<double> operator()(vector<double> actual, vector<double> predicted);
};

struct errorFunctionDeriv
{
	vector<double> (*function)(vector<double>, vector<double>);
	errorFunctionDeriv(std::string typeFunction);
	vector<double> operator()(vector<double> actual, vector<double> predicted);
};