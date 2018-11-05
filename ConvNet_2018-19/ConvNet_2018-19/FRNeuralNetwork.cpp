#include "FCNeuralNetwork.h"
#include "stdafx.h"

FC_NN::FC_NN(string activationFunc, vector<int> layers)
{
	num_inputs = layers[0];
	num_outputs = layers.back();

	activation = activationFunc;
	FC_NN::activation = activationFunction(activationFunc);
	FC_NN::activationDeriv = activationFunctionDeriv(activationFunc);

	FC_NN::NNlayers = layers;

	for (int val : layers)
	{
		vector<double> yeet(val,0.0);

		FC_NN::inputs.push_back(yeet);
		FC_NN::outputs.push_back(yeet);
	}

	FC_NN::bias = vector<double>(layers.size()-1,FC_NN::dist(0));

	for (int i = 0; i < layers.size() - 1; i++)
	{
		FC_NN::weights.push_back(vector<vector<double>> (inputs[i + 1].size()));

		for (vector<double> vec : weights[i])
		{
			vec = vector<double> (outputs[i].size(),FC_NN::dist(0));
		}
	}

	int numNeurons = 0;
	for (vector<vector<double>> vecs : weights)
	{
		for (vector<double> layer : vecs)
		{
			numNeurons += layer.size();
		}
	}

	num_neurons = numNeurons;
}

vector<double> FC_NN::feedforwardTemplate(vector<double> inputsVec)
{
	FC_NN::inputs[0] = inputsVec;
	
	for (int i = 0; i < inputs.size()-1; i++)
	{
		std::transform(inputs[i].begin(), inputs[i].end(), outputs[i].begin(), activation);
		for (int n = 0; n < inputs[i+1].size(); n++)
		{
			inputs[i + 1][n] = dotProduct(weights[i][n], outputs[i]) + bias[i];
		}
	}

	std::transform(inputs[inputs.size() - 1].begin(), inputs[inputs.size() - 1].end(), outputs[outputs.size() - 1].begin(), activation);
	return outputs[outputs.size() - 1];
}

vector<double> FC_NN::feedforwardTemplate(vector<double> inputsVec, vector<vector<int>> droppedOut)
{
	inputs[0] = inputsVec;

	for (int i = 0; i < inputs.size() - 1; i++)
	{
		std::transform(inputs[i].begin(), inputs[i].end(), outputs[i].begin(), activation);
		for (int n = 0; n < inputs[i + 1].size(); n++)
		{
			inputs[i + 1][n] = dotProduct(weights[i][n], outputs[i]) + bias[i];

			double droppedOutSum = 0;
			for (int dONeuron : droppedOut[i])
			{
				droppedOutSum = droppedOutSum + outputs[i][dONeuron] * weights[i][n][droppedOutSum];
			}

			inputs[i + 1][n] = inputs[i + 1][n] - droppedOutSum;
		}
	}

	std::transform(inputs[inputs.size() - 1].begin(), inputs[inputs.size() - 1].end(), outputs[outputs.size() - 1].begin(), activation);
	return outputs[outputs.size() - 1];
}

vector<double> FC_NN::feedforwardPreserve(vector<double> inputsVec)
{
	vector<double> biasClone = FC_NN::bias;
	vector<vector<double>> inputsClone = FC_NN::inputs;
	vector<vector<double>> outputsClone = FC_NN::outputs;
	vector<vector<vector<double>>> weightsClone = FC_NN::weights;

	inputsClone[0] = inputsVec;

	for (int i = 0; i < inputsClone.size() - 1; i++)
	{
		std::transform(inputsClone[i].begin(), inputsClone[i].end(), outputsClone[i].begin(), activation);
		for (int n = 0; n < inputsClone[i + 1].size(); n++)
		{
			inputsClone[i + 1][n] = dotProduct(weightsClone[i][n], outputsClone[i]) + biasClone[i];
		}
	}

	std::transform(inputsClone[inputsClone.size() - 1].begin(), inputsClone[inputs.size() - 1].end(), outputsClone[outputsClone.size() - 1].begin(), activation);
	return outputsClone[outputsClone.size() - 1];
}

vector<double> FC_NN::feedforwardPreserve(vector<double> inputsVec, vector<vector<int>> droppedOut)
{
	vector<double> biasClone = FC_NN::bias;
	vector<vector<double>> inputsClone = FC_NN::inputs;
	vector<vector<double>> outputsClone = FC_NN::outputs;
	vector<vector<vector<double>>> weightsClone = FC_NN::weights;

	inputsClone[0] = inputsVec;

	for (int i = 0; i < inputsClone.size() - 1; i++)
	{
		std::transform(inputsClone[i].begin(), inputsClone[i].end(), outputsClone[i].begin(), activation);
		for (int n = 0; n < inputsClone[i + 1].size(); n++)
		{
			inputsClone[i + 1][n] = dotProduct(weightsClone[i][n], outputsClone[i]) + biasClone[i];
			
			double droppedOutSum = 0;
			for (int dONeuron : droppedOut[i])
			{
				droppedOutSum = droppedOutSum + outputsClone[i][dONeuron] * weightsClone[i][n][droppedOutSum];
			}

			inputsClone[i + 1][n] = inputsClone[i + 1][n] - droppedOutSum;
		}
	}

	std::transform(inputsClone[inputsClone.size() - 1].begin(), inputsClone[inputs.size() - 1].end(), outputsClone[outputsClone.size() - 1].begin(), activation);
	return outputsClone[outputsClone.size() - 1];
}

vector<vector<int>> dropoutNeurons(vector<int> layerData, double dropoutRate)
{
	vector<int> nums;
	std::default_random_engine rng = std::default_random_engine{};

	int num = 0;
	for (int val : layerData)
	{
		for (int i = 0; i < val; i++)
		{
			nums.push_back(num);
			num++;
		}
	}
	
	std::shuffle(std::begin(nums), std::end(nums), rng);
	nums.resize(dropoutRate * nums.size());
	std::sort(nums.begin(), nums.end());

	vector<vector<int>> droppedOutNeurons;
	
	for (int dO : nums)
	{
		int subtracter = 0;
		int x = 0;
		for (int val : layerData)
		{
			dO -= subtracter;
			if (dO < val)
			{
				droppedOutNeurons[x].push_back(dO);
				break;
			}
			else
			{
				x++;
				subtracter += val;
			}
		}
	}

	return droppedOutNeurons;
}