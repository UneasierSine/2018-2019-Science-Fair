#include "stdafx.h"
#include "ActivationFunctions.h"

double alpha = 0.1;

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

double tanhFunc(double x)
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

double prelu(double x)
{
	if (x < 0) return alpha * x;
	else return x;
}

double elu(double a, double x)
{
	if (x < 0) return a * (exp(x) - 1.0);
	else return x;
}

double elu(double x)
{
	if (x < 0) return alpha * (exp(x) - 1.0);
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

double preluDeriv(double x)
{
	if (x < 0)return alpha;
	else return 1;
}

double eluDeriv(double a, double x)
{
	if (x < 0) return a * exp(x);
	else return 1;
}

double eluDeriv(double x)
{
	if (x < 0) return alpha * exp(x);
	else return 1;
}

double softPlusDeriv(double x)
{
	return exp(x) / (1 + exp(x));
}


//Functors 

activationFunction::activationFunction(std::string typeFunction)
{
	if (typeFunction.find("identity") >= 0)
	{
		function = &identity;
	}
	else if (typeFunction.find("binaryStep") >= 0)
	{
		function = &binaryStep;
	}
	else if (typeFunction.find("logistic") >= 0)
	{
		function = &logistic;
	}
	else if (typeFunction.find("tanh") >= 0)
	{
		function = &tanhFunc;
	}
	else if (typeFunction.find("arctan") >= 0)
	{
		function = &arctan;
	}
	else if (typeFunction.find("relu") >= 0)
	{
		function = &relu;
	}
	else if (typeFunction.find("prelu") >= 0)
	{
		function = &prelu;
	}
	else if (typeFunction.find("elu") >= 0)
	{
		function = &elu;
	}
	else if (typeFunction.find("softPlus") >= 0)
	{
		function = &softPlus;
	}
}

double activationFunction::operator()(double x)
{
	return function(x);
}

activationFunctionDeriv::activationFunctionDeriv(std::string typeFunction)
{
	if (typeFunction.find("identity") >= 0)
	{
		function = &identityDeriv;
	}
	else if (typeFunction.find("binaryStep") >= 0)
	{
		function = &binaryStepDeriv;
	}
	else if (typeFunction.find("logistic") >= 0)
	{
		function = &logisticDeriv;
	}
	else if (typeFunction.find("tanh") >= 0)
	{
		function = &tanhDeriv;
	}
	else if (typeFunction.find("arctan") >= 0)
	{
		function = &arctanDeriv;
	}
	else if (typeFunction.find("relu") >= 0)
	{
		function = &reluDeriv;
	}
	else if (typeFunction.find("prelu") >= 0)
	{
		function = &preluDeriv;
	}
	else if (typeFunction.find("elu") >= 0)
	{
		function = &eluDeriv;
	}
	else if (typeFunction.find("softPlus") >= 0)
	{
		function = &softPlusDeriv;
	}
}

double activationFunctionDeriv::operator()(double x)
{
	return function(x);
}