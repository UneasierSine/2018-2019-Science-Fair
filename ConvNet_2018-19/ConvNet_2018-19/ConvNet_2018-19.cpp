// ConvNet_2018-19.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ConvNet_2018-2019.h"

int main()
{

#pragma region File_Inits

	//Initialize input file for the input neuron values
	std::ifstream inputFile;
	inputFile.open("Input.csv");
	if (inputFile.fail())
	{
		cout << "There was an error when opening the input file." << endl;
		return 1;
	}

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
	std::string actFuncType;
	cin >> actFuncType;
	cout << actFuncType << " is the activation function." << endl;

	std::string erFuncType;
	cin >> erFuncType;
	cout << erFuncType << " is the error function." << endl;

	vector<int> nnStructure = {3,6,2,1,8,4,1};

	FC_NN neuralNet = FC_NN::FC_NN(actFuncType, erFuncType, nnStructure);
#pragma endregion

#pragma region Feed_Data
	vector<vector<double>> inputs;
	vector<double> outputs_actual;

	string::size_type sz;
	string line;
	while (getline(inputFile, line))
	{
		vector<double> in;
		in.push_back(stod(line.substr(0, line.find_first_of(',')), &sz));
		line.erase(0, line.find_first_of(',')+1);
		in.push_back(stod(line.substr(0, line.find_first_of(',')), &sz));
		line.erase(0, line.find_first_of(',') + 1);
		in.push_back(stod(line.substr(0, line.find_first_of(',')), &sz));
		line.erase(0, line.find_first_of(',') + 1);
		double out = stod(line.substr(0, line.find_first_of(',')), &sz);

		inputs.push_back(in);
		outputs_actual.push_back(out);
	}

	for (int x = 0; x < inputs.size(); x++)
	{
		vector<double> errors = neuralNet.feedforwardError(inputs[x], vector<double>(1, outputs_actual[x]));
		for (double err : errors)
		{
			accuracies << err << ",";
		}
		accuracies << endl;
	}
#pragma endregion

#pragma region Training
	for (int epoch = 0; epoch < 20; epoch++)
	{
		for (int x = 0; x < inputs.size(); x++)
		{
			vector<double> errors = neuralNet.feedforwardError(inputs[x], vector<double>(1, outputs_actual[x]));
			for (double err : errors)
			{
				accuracies << err << ",";
			}

			accuracies << endl;
			
			vector<double> vals = neuralNet.feedforwardPreserve(inputs[x]);
			
			vector<vector<vector<double>>> gradsToPass = neuralNet.backprop(vector<double>(1, outputs_actual[x]), inputs[x]);
			neuralNet.sgd(gradsToPass);
		}
	}
#pragma endregion

#pragma region Final
	for (int x = 0; x < inputs.size(); x++)
	{
		vector<double> errors = neuralNet.feedforwardError(inputs[x], vector<double>(1, outputs_actual[x]));
		for (double err : errors)
		{
			accuracies << err << ",";
		}
		accuracies << endl;
	}

	accuracies.close();
	inputFile.close();
#pragma endregion

    return 0;
}

