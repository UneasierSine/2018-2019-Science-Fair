#include "FCNeuralNetwork.h"

FC_NN::FC_NN(std::string activationFunc, vector<int> layers)
{
	num_inputs = layers[0];
	num_outputs = layers.back();

	FC_NN::activation = activationFunction(activationFunc);
	FC_NN::activationDeriv = activationFunctionDeriv(activationFunc);

	for (int val : layers)
	{
		vector<double> yeet(val);
	}
}