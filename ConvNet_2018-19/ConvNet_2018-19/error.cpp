#include "stdafx.h"
#include "error.h"

vector<double> squaredError(vector<double> actual, vector<double> predicted)
{
	return powMat(subMat(actual, predicted),2);
}

vector<double> squaredLogError(vector<double> actual, vector<double> predicted)
{
	vector<double> returnVector;
	for (int i = 0; i < actual.size(); i++)
	{
		returnVector.push_back(pow(std::log(actual[i] + 1.0) - std::log(predicted[i] + 1),2));
	}
	return returnVector;
}

vector<double> l1Loss(vector<double> actual, vector<double> predicted)
{
	return subMat(actual, predicted);
}

vector<double> squaredErrorDeriv(vector<double> actual, vector<double> predicted)
{
	vector<double> returnVector;
	for (int i = 0; i < actual.size(); i++)
	{
		returnVector.push_back(-2.0 * (actual[i] - predicted[i]));
	}
	return returnVector;
}

vector<double> squaredLogErrorDeriv(vector<double> actual, vector<double> predicted)
{
	vector<double> returnVector;
	for (int i = 0; i < actual.size(); i++)
	{
		returnVector.push_back( -2.0*(log(actual[i] + 1) - log(predicted[i] + 1)) / (predicted[i] + 1) );
	}
	return returnVector;
}

vector<double> l1LossDeriv(vector<double> actual, vector<double> predicted)
{
	vector<double> returnVector(actual.size(), -1);
	return returnVector;
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

vector<double> errorFunction::operator()(vector<double> actual, vector<double> predicted)
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

vector<double> errorFunction::operator()(vector<double> actual, vector<double> predicted)
{
	return function(actual, predicted);
}