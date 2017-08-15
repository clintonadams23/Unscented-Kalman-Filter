#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
	is_initialized_ = false;
  // if this is false, laser measurements will be ignored 
  use_laser_ = true;

  // if this is false, radar measurements will be ignored 
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
	P_ << .5,  0,  0,  0,  0,
				 0, .5,  0,  0,  0,
				 0,  0, .5,  0,  0,
				 0,  0,  0, .5,  0,
				 0,  0,  0,  0, .5;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = .5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

	n_x_ = x_.size();

	n_aug_ = n_x_ + 2;

	n_sig_ = 2 * n_aug_ + 1;

	lambda_ = 3 - n_aug_;

	weights_ = VectorXd(n_sig_);
	weights_(0) = lambda_ / (lambda_ + n_aug_);

	for (int i = 1; i < n_sig_; i++) {
		weights_(i) = 0.5 / (n_aug_ + lambda_);
	}
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) 
		ProcessLaser(meas_package);

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) 
		ProcessRadar(meas_package);
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
	MatrixXd Xsig_aug = GenerateSigmaPoints();
	Xsig_pred_ = PredictSigmaPoints(Xsig_aug, delta_t);
	x_ = PredictMean();
	P_ = PredictCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	// We can just use a standard Kalman Filter for Lidar.

	VectorXd z = meas_package.raw_measurements_;
	MatrixXd H = MatrixXd(2, 5);
	H << 1, 0, 0, 0, 0,
			 0, 1, 0, 0, 0;

	MatrixXd R = MatrixXd(2, 2);
	R << pow(std_laspx_, 2),									0,
												0, pow(std_laspy_, 2);

	VectorXd z_pred = H * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H.transpose();
	MatrixXd S = H * P_ * Ht + R;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H) * P_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	VectorXd z = meas_package.raw_measurements_;
	const int n_z = 3;
	MatrixXd Zsig = MatrixXd(n_z, n_sig_);

	for (int i = 0; i < n_sig_; i++) {
		Zsig.col(i) = TransformSigmaPointSpace(Xsig_pred_.col(i));
	}

	VectorXd z_pred = PredictRadarMean(Zsig, n_z);
	MatrixXd S = PredictRadarCovariance(z_pred, Zsig, n_z);

	// Cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	Tc.fill(0);
	for (int i = 0; i < n_sig_; i++) {
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		x_diff(3) = NormalizeAngle(x_diff(3));

		VectorXd z_diff = Zsig.col(i) - z_pred;
		z_diff(1) = NormalizeAngle(z_diff(1));

		Tc += weights_(i) * x_diff * z_diff.transpose();
	}

	MatrixXd K = Tc * S.inverse();

	x_ += K * (z - z_pred);
	P_ -= K * S * K.transpose();
}

/**
* Converts a radar measurement into cartesian space. Intended to initialize the x state given a radar measurement
*/
VectorXd UKF::ConvertMeasurementToCartesian(const VectorXd &measurements, bool is_inital) {
	float rho = measurements(0);
	float phi = measurements(1);
	float rho_dot = measurements(2);

	float px = rho*cos(phi);
	float py = rho*sin(phi);
	float vx = is_inital ? 0 : rho_dot*cos(phi);
	float vy = is_inital ? 0 : rho_dot*sin(phi);
	float v = sqrt(pow(vx, 2) + pow(vy, 2));
	float psi = 0;
	float psi_dot = 0;

	VectorXd cartesian_measurements = VectorXd(5);
	cartesian_measurements << px, py, v, psi, psi_dot;

	return cartesian_measurements;
}

/**
* Creates an augmented covariance matrix and uses it to generate an augmented sigma point Matrix. 
*/
MatrixXd UKF::GenerateSigmaPoints() {
	VectorXd x_aug = VectorXd(n_aug_);
	x_aug.head(n_x_) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;

	const int n_q = 2;
	MatrixXd Q = MatrixXd(n_q, n_q);
	Q <<
		pow(std_a_, 2),									 0,
								 0, pow(std_yawdd_, 2);

	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug.bottomRightCorner(n_q, n_q) = Q;
	P_aug.bottomLeftCorner(n_aug_ - n_x_, n_x_) = MatrixXd::Zero(n_aug_ - n_x_, n_x_);
	P_aug.topRightCorner(n_x_, n_aug_ - n_x_) = MatrixXd::Zero(n_x_, n_aug_ - n_x_);

	//create square root matrix
	MatrixXd A = P_aug.llt().matrixL();

	MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);

	Xsig_aug.col(0) = x_aug;
	for (int i = 1; i <= n_aug_; i++)
	{
		Xsig_aug.col(i) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i - 1);
		Xsig_aug.col(i + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i - 1);
	}

	return Xsig_aug;
}

/**
* Ensures that an angle is within plus/minus two pi. Helpful for angle subtraction operations.
* @param {float} phi
*/
float UKF::NormalizeAngle(float phi) {
	float pi = M_PI;

	if (phi > pi)
		phi = fmod(phi - pi, 2 * pi) - pi;
	if (phi < -pi)
		phi = fmod(phi + pi, 2 * pi) + pi;

	return phi;
}

/**
* Processing chain for laser measurements. Initializes the state if it is the first measurement, performs a prediction, and then initiates a kalman filter update step
* @param {MeasurementPackage} meas_package
*/
void UKF::ProcessLaser(MeasurementPackage meas_package) {
	if (!is_initialized_) {
		x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
		time_us_ = meas_package.timestamp_;
		is_initialized_ = true;
		return;
	}

	double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
	time_us_ = meas_package.timestamp_;

	Prediction(delta_t);
	UpdateLidar(meas_package);
}

/**
* Takes a generated sigma point from an augmented sigma point matrix and returns a sigma point prediction
* @param {VectorXd} x
* @param {double} delta_t
*/
VectorXd UKF::ProcessModel(VectorXd &x, double delta_t) {
	double pos_x = x(0);
	double pos_y = x(1);
	double v = x(2);
	double psi = NormalizeAngle(x(3));
	double psi_dot = x(4);
	double nu_a = x(5);
	double nu_psi_dot_dot = x(6);

	VectorXd process_noise = VectorXd(5);
	process_noise << 
		   0.5 * pow(delta_t, 2)*cos(psi)*nu_a,
		   0.5 * pow(delta_t, 2)*sin(psi)*nu_a,
												    delta_t * nu_a,
		0.5 * pow(delta_t, 2) * nu_psi_dot_dot,
		              delta_t * nu_psi_dot_dot;

	VectorXd delta_x = VectorXd(5);

	if (fabs(psi_dot) > 0.001) {
		delta_x << 
			 (v / psi_dot) * (sin(psi + psi_dot*delta_t) - sin(psi)),
			(v / psi_dot) * (-cos(psi + psi_dot*delta_t) + cos(psi)),
																														 0,
																						 psi_dot * delta_t,
																						                 0;
	}
	else {
		delta_x << 
			v * cos(psi) * delta_t,
			v * sin(psi) * delta_t,
													 0,
													 0,
													 0;
	}

	VectorXd x_kplus1 = x.head(5) + delta_x + process_noise;
	return x_kplus1;
}

/**
* Processing chain for radar measurements. Initializes the state if it is the first measurement, performs a prediction, and then initiates an unscented kalman filter update step
* @param {MeasurementPackage} meas_package
*/
void UKF::ProcessRadar(MeasurementPackage meas_package) {
	if (!is_initialized_) {
		x_ << ConvertMeasurementToCartesian(meas_package.raw_measurements_, true);
		time_us_ = meas_package.timestamp_;
		is_initialized_ = true;
		return;
	}

	double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
	time_us_ = meas_package.timestamp_;

	Prediction(delta_t);
	UpdateRadar(meas_package);
}

MatrixXd UKF::PredictCovariance() {
	MatrixXd P = MatrixXd(n_x_, n_x_);
	P.fill(0.0);

	for (int i = 0; i < n_sig_; i++) {
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		x_diff(3) = NormalizeAngle(x_diff(3));
		P += weights_(i) * x_diff*x_diff.transpose();
	}

	return P;
}

VectorXd UKF::PredictMean() {
	VectorXd x = VectorXd(n_x_);
	x.fill(0.0);

	for (int i = 0; i < n_sig_; i++) {
		x += weights_(i) * Xsig_pred_.col(i);
	}

	return x;
}

MatrixXd UKF::PredictRadarCovariance(const VectorXd &z_pred, const MatrixXd &Zsig, const int n_z) {
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);

	MatrixXd R = MatrixXd(n_z, n_z);
	R <<
		pow(std_radr_, 2),                   0,                  0,
							      0, pow(std_radphi_, 2),                  0,
		                0,                   0, pow(std_radrd_, 2);

	for (int i = 0; i < n_sig_; i++) {
		VectorXd z_diff = Zsig.col(i) - z_pred;
		z_diff(1) = NormalizeAngle(z_diff(1));
		S += weights_(i) * z_diff*z_diff.transpose();
	}

	S += R;
	return S;
}

MatrixXd UKF::PredictRadarMean(const MatrixXd &Zsig, const int n_z) {
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);

	for (int i = 0; i < n_sig_; i++) {
		z_pred += weights_(i) * Zsig.col(i);
	}

	return z_pred;
}

MatrixXd UKF::PredictSigmaPoints(const MatrixXd &Xsig_aug, double delta_t) {
	MatrixXd Xsig_pred = MatrixXd(n_x_, n_sig_);
	for (int i = 0; i < 2 * n_aug_ + 1; i++){
		VectorXd xsig = Xsig_aug.col(i);
		Xsig_pred.col(i) = ProcessModel(xsig, delta_t);
	}
	return Xsig_pred;
}

/**
* Performs the transformation of a sigma point from cartesian space into radar measurement space
* @param {VectorXd} x
*/
VectorXd UKF::TransformSigmaPointSpace(const VectorXd &x) {
	double p_x = x[0];
	double p_y = x[1];
	double v = x[2];
	double psi = NormalizeAngle(x[3]);
	double psi_dot = x[4];

	double rho = sqrt(pow(p_x, 2) + pow(p_y, 2));

	if (fabs(rho) < 0.001)
		rho = 0.001;

	double phi = atan2(p_y, p_x);
	double rho_dot = (p_x*cos(psi)*v + p_y*sin(psi)*v) / rho;
	
	VectorXd z = VectorXd(3);
	z << rho, phi, rho_dot;
	return z;
}


