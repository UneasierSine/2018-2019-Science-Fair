#pragma once
#include <math.h>
#include <string>

double alpha;

//Base activation functions
double identity(double x);
double binaryStep(double x);
double logistic(double x);
double tanhFunc(double x);
double arctan(double x);
double relu(double x);
double prelu(double a, double x);
double prelu(double x);
double elu(double a, double x);
double elu(double x);
double softPlus(double x);

//Activation function derivatives
double identityDeriv(double x);
double binaryStepDeriv(double x);
double logisticDeriv(double x);
double tanhDeriv(double x);
double arctanDeriv(double x);
double reluDeriv(double x);
double preluDeriv(double a, double x);
double preluDeriv(double x);
double eluDeriv(double a, double x);
double eluDeriv(double x);
double softPlusDeriv(double x);

//Functors
struct activationFunction
{
	double(*function)(double);
	activationFunction(std::string typeFunction);
	double operator()(double x);
};

struct activationFunctionDeriv
{
	double(*function)(double);
	activationFunctionDeriv(std::string typeFunction);
	double operator()(double x);
};