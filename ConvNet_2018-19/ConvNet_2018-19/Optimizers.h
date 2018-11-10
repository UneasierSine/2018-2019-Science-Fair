#pragma once
#include <string>
#include <math.h>
#include <vector>

double stochastic(double mu, double gradient, double currentVal);

double momentum(double mu, double gradient, double currentVal, double lastUpdate, double momentum);

double nesterovGradPlace(double momentum, double lastUpdate, double currentVal);

double nesterov(double mu, double gradient, double currentVal, double lastUpdate, double momentum);

double adagrad(double mu, double gradient, double currentVal, double epsilon, double sumGrads);

double adadelta(double mu, double gradient, double currentVal, double epsilon, double sumGrads, double momentum);

double adam(double mu, double gradient, double currentVal, double epsilon, double b1, double b2, double dAvg, double dAvgSq);