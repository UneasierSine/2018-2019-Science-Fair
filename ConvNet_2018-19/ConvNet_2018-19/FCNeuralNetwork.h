#pragma once
#include "ActivationFunctions.h"
#include "MatrixMath.h"
#include <vector>
#include <math.h>
#include <string.h>

class FC_NN
{
private:
	int num_hiddenLayers;
	int num_inputs;
	int num_outputs;

	gaussianRandom dist;
	vector<vector<double>> inputs;
	vector<vector<double>> outputs;
	vector<vector<vector<double>>> weights;

public:
	activationFunction activation;
	activationFunctionDeriv activationDeriv;

	FC_NN(std::string activationFunc, vector<int> layers);
};