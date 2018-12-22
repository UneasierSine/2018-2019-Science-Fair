#include "stdafx.h"
#include "FC_NN.h"

FC_NN::FC_NN(std::string activationFunc, std::string errorFunctionName, vector<int> layers)
{
	num_inputs = layers[0];
	num_outputs = layers.back();

	activationStr = activationFunc;
	FC_NN::activation = activationFunction(activationFunc);
	FC_NN::activationDeriv = activationFunctionDeriv(activationFunc);

	FC_NN::errorFunc = errorFunction(errorFunctionName);
	FC_NN::errorFuncDeriv = errorFunctionDeriv(errorFunctionName);

	FC_NN::NNlayers = layers;

	for (int val : layers)
	{
		vector<double> yeet(val,0.0);

		FC_NN::inputs.push_back(yeet);
		FC_NN::outputs.push_back(yeet);
	}

	for (int i = 0; i < layers.size() - 1; i++)
	{
		FC_NN::weights.push_back(vector<vector<double>> (inputs[i + 1].size(), vector<double> (outputs[i].size(), rand()%2+1)));
	}

	int numNeurons = 0;
	for (vector<vector<double>> vecs : weights)
	{
		for (vector<double> layer : vecs)
		{
			numNeurons += (int)layer.size();
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
			inputs[i + 1][n] = dotProduct(weights[i][n], outputs[i]);
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
			inputs[i + 1][n] = dotProduct(weights[i][n], outputs[i]);

			double droppedOutSum = 0;
			for (int dONeuron : droppedOut[i])
			{
				droppedOutSum = droppedOutSum + outputs[i][dONeuron] * weights[i][n][dONeuron];
			}

			inputs[i + 1][n] = inputs[i + 1][n] - droppedOutSum;
		}
	}

	std::transform(inputs[inputs.size() - 1].begin(), inputs[inputs.size() - 1].end(), outputs[outputs.size() - 1].begin(), activation);
	return outputs[outputs.size() - 1];
}

vector<double> FC_NN::feedforwardPreserve(vector<double> inputsVec)
{
	vector<vector<double>> inputsClone = FC_NN::inputs;
	vector<vector<double>> outputsClone = FC_NN::outputs;
	vector<vector<vector<double>>> weightsClone = FC_NN::weights;

	inputsClone[0] = inputsVec;

	for (int i = 0; i < inputsClone.size() - 1; i++)
	{
		std::transform(inputsClone[i].begin(), inputsClone[i].end(), outputsClone[i].begin(), activation);
		for (int n = 0; n < inputsClone[i + 1].size(); n++)
		{
			inputsClone[i + 1][n] = dotProduct(weightsClone[i][n], outputsClone[i]);
		}
	}

	std::transform(inputsClone[inputsClone.size() - 1].begin(), inputsClone[inputs.size() - 1].end(), outputsClone[outputsClone.size() - 1].begin(), activation);
	return outputsClone[outputsClone.size() - 1];
}

vector<double> FC_NN::feedforwardPreserve(vector<double> inputsVec, vector<vector<int>> droppedOut)
{
	vector<vector<double>> inputsClone = FC_NN::inputs;
	vector<vector<double>> outputsClone = FC_NN::outputs;
	vector<vector<vector<double>>> weightsClone = FC_NN::weights;

	inputsClone[0] = inputsVec;

	for (int i = 0; i < inputsClone.size() - 1; i++)
	{
		std::transform(inputsClone[i].begin(), inputsClone[i].end(), outputsClone[i].begin(), activation);
		for (int n = 0; n < inputsClone[i + 1].size(); n++)
		{
			inputsClone[i + 1][n] = dotProduct(weightsClone[i][n], outputsClone[i]);
			
			double droppedOutSum = 0;
			for (int dONeuron : droppedOut[i])
			{
				droppedOutSum = droppedOutSum + outputsClone[i][dONeuron] * weightsClone[i][n][dONeuron];
			}

			inputsClone[i + 1][n] = inputsClone[i + 1][n] - droppedOutSum;
		}
	}

	std::transform(inputsClone[inputsClone.size() - 1].begin(), inputsClone[inputs.size() - 1].end(), outputsClone[outputsClone.size() - 1].begin(), activation);
	return outputsClone[outputsClone.size() - 1];
}

vector<double> FC_NN::feedforwardError(vector<double> inputsVec, vector<double> labels)
{
	vector<double> results = feedforwardTemplate(inputsVec);
	vector<double> errors = vector<double>(results.size());
	transform(labels.begin(), labels.end(), results.begin(), errors.begin(), errorFunc);
	return errors;
}

vector<double> FC_NN::feedforwardError(vector<double> inputsVec, vector<vector<int>> droppedOut, vector<double> labels)
{
	vector<double> results = feedforwardPreserve(inputsVec, droppedOut);
	vector<double> errors = vector<double>(results.size());
	transform(labels.begin(), labels.end(), results.begin(), errors.begin(), errorFunc);
	return errors;
}

vector<vector<vector<double>>> FC_NN::backprop(vector<double> actual, vector<double> predicted)
{
	vector<vector<double>> grads;
	vector<double> biasGrads;

	for (vector<double> column : inputs)
	{
		grads.push_back(vector<double>(column.size(), 0.0));
	}

	vector<vector<vector<double>>> weightGradients;
	for (int x = 0; x < weights.size(); x++)
	{
		weightGradients.push_back(vector<vector<double>>());
		for (int y = 0; y < weights[x].size(); y++)
		{
			weightGradients[x].push_back(vector<double>(weights[x][y].size(),1));
		}
	}
	
	vector<double> error(actual.size());
	transform(actual.begin(), actual.end(), predicted.begin(), error.begin(), errorFuncDeriv);
	//set the errors to the last column of the network, index -> sizeOfVector - 1
	grads[grads.size() - 1] = error;

	int layerNum = (int)grads.size() - 1;
	while (layerNum > 1)
	{
		for_each(grads[layerNum].begin(), grads[layerNum].end(), activationDeriv);
		for (int x = 0; x < grads[layerNum].size(); x++)
		{
			for (int y = 0; y < grads[layerNum - 1].size(); y++)
			{
				if (layerNum > 0)
				{
					weightGradients[layerNum-1][x][y] = (outputs[layerNum - 1][y] * grads[layerNum][x]);
					grads[layerNum - 1][y] += weights[layerNum-1][x][y] * grads[layerNum][x];
				}
			}
		}
		layerNum--;
	}
	return weightGradients;
}

vector<vector<vector<double>>> FC_NN::backprop(vector<double> actual, vector<double> predicted, vector<vector<int>> droppedOut)
{
	vector<vector<double>> grads;
	vector<double> errors = vector<double>(actual.size());

	for (vector<double> column : inputs)
	{
		grads.push_back(vector<double>(column.size(), 0.0));
	}

	vector<vector<vector<double>>> weightGradients;
	for (int x = 0; x < weights.size(); x++)
	{
		for (int y = 0; y < weights[x].size(); y++)
		{
			for (int z = 0; z < weights[x][y].size(); z++)
			{
				weightGradients[x][y][z] = 0.0;
			}
		}
	}

	transform(actual.begin(), actual.end(), predicted.begin(), errors.begin(), errorFuncDeriv);
	grads[grads.size() - 1] = errors;

	int layerNum = (int)grads.size() - 1;
	while (layerNum > 0)
	{
		for_each(grads[layerNum].begin(), grads[layerNum].end(), activationDeriv);
		for (int x = 0; x < grads[layerNum].size(); x++)
		{
			for (int y = 0; y < grads[layerNum - 1].size(); y++)
			{
				if (find(droppedOut[layerNum].begin(), droppedOut[layerNum].end(), y) != droppedOut[layerNum].end())
				{
					weightGradients[layerNum][x][y] = 0;
					grads[layerNum - 1][y] += 0;
				}
				else
				{
					weightGradients[layerNum][x][y] = (grads[layerNum][y] * grads[layerNum + 1][x]);
					grads[layerNum - 1][y] += weights[layerNum][x][y] * grads[layerNum + 1][x];
				}
			}
		}
		layerNum--;
	}
	return weightGradients;
}

void FC_NN::sgd(vector<vector<vector<double>>> weightGrads)
{
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			for (int k = 0; k < weights[i][j].size(); k++)
			{
				weights[i][j][k] += -0.25 * (weightGrads[i][j][k]);
			}
		}
	}
}

vector<vector<int>> dropoutNeurons(vector<int> layerData, double dropoutRate)
{
	vector<int> nums;
	
	int num = 0;
	for (int val : layerData)
	{
		for (int i = 0; i < val; i++)
		{
			nums.push_back(num);
			num++;
		}
	}
	
	random_shuffle(nums.begin(), nums.end());
	nums.resize((int)(dropoutRate * nums.size()));
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