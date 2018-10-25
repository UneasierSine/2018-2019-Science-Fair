#pragma once
#include "ActivationFunctions.h"
#include "MatrixMath.h"
#include <vector>
#include <string.h>

class FC_NN
{
private:
	int num_hiddenLayers;
	int num_inputs;
	int num_outputs;

	vector<vector<double>> neurons;


public:
	activationFunction activation;
	activationFunctionDeriv activationDeriv;

	FC_NN(std::string activationFunc, vector<int> layers);
};