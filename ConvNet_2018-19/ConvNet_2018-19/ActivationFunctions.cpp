#include "stdafx.h"
#include "ActivationFunctions.h"

double identity(double x)
{
	return x;
}

double binaryStep(double x)
{
	if (x < 0) return 0;
	else return 1;
}

double logistic(double x)
{
	return 1 / (1 + exp(-1.0 * x));
}

double tanh(double x)
{
	return tanh(x);
}

double arctan(double x)
{
	return atan(x);
}

double relu(double x)
{
	if (x < 0) return 0;
	else return x;
}

double prelu(double a, double x)
{
	if (x < 0) return a * x;
	else return x;
}

double elu(double a, double x)
{
	if (x < 0) return a * (exp(x) - 1.0);
	else return x;
}

double softPlus(double x)
{
	return log(1 + exp(x));
}

double identityDeriv(double x)
{
	return 1;
}

double binaryStepDeriv(double x)
{
	//Ignore the undefined derivative at x=0
	return 0;
}

double logisticDeriv(double x)
{
	return 1.0 / (exp(x) * pow(1 + exp(-1 * x), 2));
}

double tanhDeriv(double x)
{
	return pow(1 / cosh(x), 2);
}

double arctanDeriv(double x)
{
	return 1 / (x*x + 1);
}

double reluDeriv(double x)
{
	if (x < 0) return 0;
	else return 1;
}

double preluDeriv(double a, double x)
{
	if (x < 0)return a;
	else return 1;
}

double eluDeriv(double a, double x)
{
	if (x < 0) return a * exp(x);
	else return 1;
}

double softPlusDeriv(double x)
{
	return exp(x) / (1 + exp(x));
}