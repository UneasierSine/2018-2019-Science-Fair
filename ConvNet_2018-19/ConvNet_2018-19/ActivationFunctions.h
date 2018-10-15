#pragma once
#include <math.h>

//Base activation functions
double identity(double x);
double binaryStep(double x);
double logistic(double x);
double tanh(double x);
double arctan(double x);
double relu(double x);
double prelu(double a, double x);
double elu(double a, double x);
double softPlus(double x);

//Activation function derivatives
double identityDeriv(double x);
double binaryStepDeriv(double x);
double logisticDeriv(double x);
double tanhDeriv(double x);
double arctanDeriv(double x);
double reluDeriv(double x);
double preluDeriv(double a, double x);
double eluDeriv(double a, double x);
double softPlusDeriv(double x);