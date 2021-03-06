// ConvNet_2018-19.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ConvNet_2018-2019.h"

int main()
{

#pragma region File_Inits

	//Initialize cost function output file
	std::ofstream accuracies;
	accuracies.open("Accuracies.csv");
	if (accuracies.fail())
	{
		cout << "There was an error opening the accuracies file." << endl;
		return 1;
	}

#pragma endregion

#pragma region Constructor
	/**cout << "Activation function type?" << endl;
	std::string actFuncType;
	cin >> actFuncType;
	cout << actFuncType << " is the activation function." << endl;

	cout << "Error function type?" << endl;
	std::string erFuncType;
	cin >> erFuncType;
	cout << erFuncType << " is the error function." << endl;
	**/

	vector<int> nnStructure = {3,1};
	
	FC_NN neuralNet = FC_NN::FC_NN("tanh", "squared error", nnStructure);
#pragma endregion

#pragma region Feed_Data
	vector<vector<double>> inputs;
	vector<vector<double>> outputs_actual;

	string line;

	for (int n = 0; n < 50; n++)
	{
		vector<double> in;
		in.push_back(rand() % 10 + 1);
		in.push_back(rand() % 10 + 1);
		in.push_back(rand() % 10 + 1);
		
		double out = in[0] + in[1] / in[2];
		inputs.push_back(in);
		vector<double> outer = { out };
		outputs_actual.push_back(outer);

		cout << in[0] << " " << in[1] << " " << in[2] << " " << out << endl;
	}

	for (int x = 0; x < inputs.size(); x++)
	{
		vector<double> errors = neuralNet.feedforwardError(inputs[x], outputs_actual[x]);
		for (double err : errors)
		{
			accuracies << err << ",";
			cout << err << ",";
		}
		vector<double> accuracy = neuralNet.feedforwardAccuracy(inputs[x], outputs_actual[x]);
		for (double acc : accuracy)
		{
			accuracies << acc << ",";
		}
		cout << endl;
		accuracies << endl;
	}
#pragma endregion

#pragma region Training
	for (int epoch = 0; epoch < 10; epoch++)
	{
		double sum = 0;
		vector<vector<vector<vector<double>>>> gradients;
		for (int x = 0; x < inputs.size(); x++)
		{
			vector<double> vals = neuralNet.feedforwardTemplate(inputs[x]);
			vector<double> errors = neuralNet.feedforwardError(inputs[x], outputs_actual[x]);
			for (double err : errors)
			{
				cout << "epoch: " << epoch << " training sample: " << x << " err: " << err << " expected: " << outputs_actual[x][0] << " predicted: " << vals[0] << endl;
				accuracies << err << endl;
			}
			vector<double> accuracy = neuralNet.feedforwardAccuracy(inputs[x], outputs_actual[x]);
			double avgSum = 0;
			for (double acc : accuracy)
			{
				avgSum += acc;
				accuracies << acc << ",";
			}
			avgSum = avgSum / accuracy.size();
			cout << "avgSum: " << avgSum << endl;
			sum += avgSum;

			vector<vector<vector<double>>> gradsToPass = neuralNet.backprop(vals, outputs_actual[x]);
			gradients.push_back(gradsToPass);
		}

		vector<vector<vector<double>>> averageGrad;
		for (int counter1 = 0; counter1 < gradients[0].size(); counter1++)
		{
			averageGrad.push_back(vector<vector<double>>());
			for (int counter2 = 0; counter2 < gradients[0][counter1].size(); counter2++)
			{
				averageGrad[counter1].push_back(vector<double>());
				for (int counter3 = 0; counter3 < gradients[0][counter1][counter2].size(); counter3++)
				{
					double sum = 0;
					for (vector<vector<vector<double>>> singleSample : gradients)
					{
						sum += singleSample[counter1][counter2][counter3];
					}
					sum /= gradients.size();
					averageGrad[counter1][counter2].push_back(sum);
				}
			}
		}
		sum = sum / inputs.size();
		cout << sum;
		if (abs(sum) <= 1)
		{
			break;
		}
		else
		{
			neuralNet.sgd(averageGrad);
		}
	}
#pragma endregion

#pragma region Final
	for (int x = 0; x < inputs.size(); x++)
	{
		vector<double> errors = neuralNet.feedforwardError(inputs[x], outputs_actual[x]);
		for (double err : errors)
		{
			accuracies << err << ",";
			cout << err << ",";
		}
		cout << endl;
		accuracies << endl;
		vector<double> accuracy = neuralNet.feedforwardAccuracy(inputs[x], outputs_actual[x]);
		for (double acc : accuracy)
		{
			accuracies << acc << ",";
		}
	}

	accuracies.close();
#pragma endregion

    return 0;
}

