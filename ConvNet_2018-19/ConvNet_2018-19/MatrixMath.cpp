#include <vector>
#include "MatrixMath.h"
#include <math.h>

using namespace std;

vector<double> addMat(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVector = vector<double>();
	if (vec1.size() != vec2.size())
	{
		returnVector[0] = -1;
		return returnVector;
	}
	else
	{
		for (int i = 0; i < vec1.size(); i++)
		{
			returnVector[i] = vec1[i] + vec2[i];
		}
		return returnVector;
	}
}

vector<double> subMat(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVector = vector<double>();
	if (vec1.size() != vec2.size())
	{
		returnVector[0] = -1;
		return returnVector;
	}
	else
	{
		for (int i = 0; i < vec1.size(); i++)
		{
			returnVector[i] = vec1[i] - vec2[i];
		}
		return returnVector;
	}
}

vector<double> mulMat(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVector = vector<double>();
	if (vec1.size() != vec2.size())
	{
		returnVector[0] = -1;
		return returnVector;
	}
	else
	{
		for (int i = 0; i < vec1.size(); i++)
		{
			returnVector[i] = vec1[i] * vec2[i];
		}
		return returnVector;
	}
}

vector<double> divMat(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVector = vector<double>();
	if (vec1.size() != vec2.size())
	{
		returnVector[0] = -1;
		return returnVector;
	}
	else
	{
		for (int i = 0; i < vec1.size(); i++)
		{
			returnVector[i] = vec1[i] / vec2[i];
		}
		return returnVector;
	}
}

vector<double> powMat(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVector = vector<double>();
	if (vec1.size() != vec2.size())
	{
		returnVector[0] = -1;
		return returnVector;
	}
	else
	{
		for (int i = 0; i < vec1.size(); i++)
		{
			returnVector[i] = pow(vec1[i], vec2[i]);
		}
		return returnVector;
	}
}

vector<double> radMat(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVector = vector<double>();
	if (vec1.size() != vec2.size())
	{
		returnVector[0] = -1;
		return returnVector;
	}
	else
	{
		for (int i = 0; i < vec1.size(); i++)
		{
			returnVector[i] = pow(vec1[i], 1/vec2[i]);
		}
		return returnVector;
	}
}

double sumTerms(vector<double> vector)
{
	double sum = 0.0;
	for (double val : vector)
	{
		sum += val;
	}
	return sum;
}

double dotProduct(vector<double> vec1, vector<double> vec2)
{
	vector<double> returnVector = vector<double>();
	if (vec1.size() != vec2.size())
	{
		return -1;
	}
	else
	{
		for (int i = 0; i < vec1.size(); i++)
		{
			returnVector[i] = vec1[i] - vec2[i];
		}
	}

	return sumTerms(returnVector);
}