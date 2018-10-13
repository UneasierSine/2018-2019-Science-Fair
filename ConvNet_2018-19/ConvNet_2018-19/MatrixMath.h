#pragma once

#include <vector>

using namespace std;

//Basic operations on matrix indices
vector<double> addMatTerms(vector<double> vec1, vector<double> vec2);
vector<double> subMatTerms(vector<double> vec1, vector<double> vec2);
vector<double> mulMatTerms(vector<double> vec1, vector<double> vec2);
vector<double> divMatTerms(vector<double> vec1, vector<double> vec2);
vector<double> powMatTerms(vector<double> vec1, vector<double> vec2);
vector<double> radMatTerms(vector<double> vec1, vector<double> vec2);

//Matrix-specific operations
double sumTerms(vector<double> vector);
double dotProduct(vector<double> vec1, vector<double> vec2);