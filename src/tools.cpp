#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	if (estimations.size() == 0) {
		cout << "Empty estimation vector" << endl;
		return rmse;
	}

	if (estimations.size() != ground_truth.size()) {
		cout << "Error: expected the dimensions of the estimation and ground truth vectors to be the same" << endl;
		return rmse;
	}

	VectorXd squaredResiduals(4);
	squaredResiduals << 0, 0, 0, 0;

	//accumulate squared residuals
	for (int i = 0; i < estimations.size(); ++i) {
		VectorXd currentSquaredResiduals = (estimations[i] - ground_truth[i]).array().pow(2);
		squaredResiduals += currentSquaredResiduals;
	}

	VectorXd mean = squaredResiduals / estimations.size();
	rmse = mean.array().sqrt();

	return rmse;
}