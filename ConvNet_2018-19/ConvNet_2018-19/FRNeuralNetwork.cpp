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
		for (int i = 0; i < layers.size(); i++)
		{
			yeet.push_back(0);
		}

		FC_NN::inputs.push_back(yeet);
		FC_NN::outputs.push_back(yeet);
	}

	for (int i = 0; i < layers.size() - 1; i++)
	{
		FC_NN::weights.push_back(vector<vector<double>> (inputs[i + 1].size()));

		for (vector<double> vec : weights[i])
		{
			vec.push_back(vector<double>(outputs[i].size()));
			for (int y = 0; y < inputs[i].size(); y++)
			{
				vec.push_back(FC_NN::dist(1));
			}
		}
	}
}