#include <iostream>
#include "stdafx.h"
#include "Optimizers.h"

double stochastic(double mu, double gradient, double currentVal)
{
	return currentVal - gradient * mu;
}

double momentum(double mu, double gradient, double currentVal, double lastUpdate, double momentum)
{
	return currentVal - (momentum * lastUpdate + mu * gradient);
}

double nesterovGradPlace(double momentum, double lastUpdate, double currentVal)
{
	return currentVal - momentum * lastUpdate;
}

double nesterov(double mu, double gradient, double currentVal, double lastUpdate, double momentum)
{
	return currentVal - (momentum * lastUpdate + mu * gradient);
}

double adagrad(double mu, double gradient, double currentVal, double epsilon, double sumGrads)
{
	return currentVal - (mu * gradient / sqrt(sumGrads + epsilon));
}

double adadelta(double mu, double gradient, double currentVal, double epsilon, double sumGrads, double momentum)
{
	double rms = momentum * sumGrads + (1.0 - momentum) * gradient * gradient;
	return currentVal - (mu * gradient / sqrt(rms + epsilon));
}

double adam(double mu, double gradient, double currentVal, double epsilon, double b1, double b2, double dAvg, double dAvgSq)
{
	double mVal = b1 * dAvg + (1.0 - b1) * gradient;
	double vVal = b2 * dAvgSq + (1.0 - b2) * gradient * gradient;
	double mHat = mVal / (1.0 - b1);
	double vHat = vVal / (1.0 - b2);
	return currentVal - (mu * mHat / (sqrt(vHat) + epsilon));
}