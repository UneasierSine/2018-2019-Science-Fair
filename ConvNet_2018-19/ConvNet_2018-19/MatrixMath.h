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

////GPU methods basic matrix index operations
//vector<double> addMatTermsGpu(vector<double> vec1, vector<double> vec2);
//vector<double> subMatTermsGpu(vector<double> vec1, vector<double> vec2);
//vector<double> mulMatTermsGpu(vector<double> vec1, vector<double> vec2);
//vector<double> divMatTermsGpu(vector<double> vec1, vector<double> vec2);
//vector<double> powMatTermsGpu(vector<double> vec1, vector<double> vec2);
//vector<double> radMatTermsGpu(vector<double> vec1, vector<double> vec2);
//
////GPU methods matrix-specific
//double sumTermsGpu(vector<double> vector);
//double dotProductGpu(vector<double> vec1, vector<double> vec2);