#pragma once
#include <math.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <functional>
#include "MatrixMath.h"

using namespace std;

double squaredError(double actual, double predicted);
double squaredLogError(double actual, double predicted);
double l1Loss(double actual, double predicted);

double squaredErrorDeriv(double actual, double predicted);
double squaredLogErrorDeriv(double actual, double predicted);
double l1LossDeriv(double actual, double predicted);

struct errorFunction
{
	double (*function)(double, double);
	errorFunction(std::string typeFunction);
	double operator()(double actual, double predicted);
};

struct errorFunctionDeriv
{
	double (*function)(double, double);
	errorFunctionDeriv(std::string typeFunction);
	double operator()(double actual, double predicted);
};