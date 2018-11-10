#include "stdafx.h"
#include "error.h"

double squaredError(double actual, double predicted)
{
	return pow(actual - predicted,2);
}

double squaredLogError(double actual, double predicted)
{
	return pow(std::log(actual + 1.0) - std::log(predicted + 1.0),2);
}

double l1Loss(double actual, double predicted)
{
	return actual - predicted;
}

double squaredErrorDeriv(double actual, double predicted)
{
	return -2.0 * (actual - predicted);
}

double squaredLogErrorDeriv(double actual, double predicted)
{
	return  -2.0*(log(actual + 1) - log(predicted + 1)) / (predicted + 1);
}

double l1LossDeriv(double actual, double predicted)
{
	return -1;
}

errorFunction::errorFunction(std::string typeFunction)
{
	if (typeFunction.find("squared error") >= 0)
	{
		function = &squaredError;
	}
	else if (typeFunction.find("squared log error") >= 0)
	{
		function = &squaredLogError;
	}
	else if (typeFunction.find("l1") >= 0)
	{
		function = &l1Loss;
	}
}

double errorFunction::operator()(double actual, double predicted)
{
	return function(actual, predicted);
}

errorFunctionDeriv::errorFunctionDeriv(std::string typeFunction)
{
	if (typeFunction.find("squared error") >= 0)
	{
		function = &squaredErrorDeriv;
	}
	else if (typeFunction.find("squared log error") >= 0)
	{
		function = &squaredLogErrorDeriv;
	}
	else if (typeFunction.find("l1") >= 0)
	{
		function = &l1LossDeriv;
	}
}

double errorFunctionDeriv::operator()(double actual, double predicted)
{
	return function(actual, predicted);
}