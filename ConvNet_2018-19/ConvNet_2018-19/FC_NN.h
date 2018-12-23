#pragma once
#include "ActivationFunctions.h"
#include "MatrixMath.h"
#include "error.h"

#include <vector>
#include <math.h>
#include <string>
#include <iterator>
#include <iostream>
#include <random>

class FC_NN
{
private:
	int num_hiddenLayers;
	int num_inputs;
	int num_outputs;
	int num_neurons;

	std::string activationStr;

	gaussianRandom dist;

	vector<int> NNlayers = { 0 };
	vector<vector<double>> inputs;
	vector<vector<double>> outputs;
	vector<vector<vector<double>>> weights;

	activationFunction activation = activationFunction("identity");
	activationFunctionDeriv activationDeriv = activationFunctionDeriv("identity");

	errorFunction errorFunc = errorFunction("squared error");
	errorFunctionDeriv errorFuncDeriv = errorFunctionDeriv("squared error");

public:
	FC_NN(std::string activationFunc, std::string errorFunctionName, vector<int> layers);

	vector<double> feedforwardTemplate(vector<double> inputsVec);
	vector<double> feedforwardTemplate(vector<double> inputsVec, vector<vector<int>> droppedOut);

	vector<double> feedforwardPreserve(vector<double> inputsVec);
	vector<double> feedforwardPreserve(vector<double> inputsVec, vector<vector<int>> droppedOut);

	vector<double> feedforwardError(vector<double> inputsVec, vector<double> labels);
	vector<double> feedforwardAccuracy(vector<double> inputsVec, vector<double> labels);
	vector<double> feedforwardError(vector<double> inputsVec, vector<vector<int>> droppedOut, vector<double> labels);

	vector<vector<vector<double>>> backprop(vector<double> actual, vector<double> predicted);
	vector<vector<vector<double>>> backprop(vector<double> actual, vector<double> predicted, vector<vector<int>> droppedOut);

	void sgd(vector<vector<vector<double>>> weightGrads);
};

vector<vector<int>> dropoutNeurons(vector<int> layerData, double dropoutRate);