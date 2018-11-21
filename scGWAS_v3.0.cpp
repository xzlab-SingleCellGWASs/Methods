#include <iostream>
#include <fstream>
#include <RcppEigen.h>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>        // For boost::mt19937
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/math/distributions/normal.hpp>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(BH)]]

#include <R.h>
#include <Rmath.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h> 
#include <cstring>
#include <ctime>
#include <Rcpp.h>

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

using namespace std;
using namespace Eigen;
using namespace Rcpp;

typedef SparseMatrix<double> SpMat;


MatrixXd col_multiple(VectorXd p, MatrixXd A) {

	int col_num = A.cols();
	int row_num = A.rows();
	MatrixXd A_p(row_num, col_num);
	for (int i = 0; i < col_num; i++) {

		A_p.col(i) = p.array() * A.col(i).array();
	}
	return A_p;
}


//**********************************************************************//
//                   Log Likelihood Function                            //
//**********************************************************************//
double calcLogLikelihood_mcmc(int num_sample, double SSR, double a_beta, double b_beta, double sigma2_beta,
							  double sigma2_e, VectorXd beta, VectorXd gamma, VectorXd pi)
{
	
	double log_density = - 0.5 * num_sample * log(sigma2_e) - 0.5 * SSR / sigma2_e ;
	//cout << "log_density1: " << log_density << endl;
	log_density -= 0.5 * ((gamma.array() * (log(sigma2_beta) + log(sigma2_e) + beta.array() * beta.array() / (sigma2_beta*sigma2_e))).matrix().sum());
	//cout << "log_density2: " << log_density << endl;
	log_density += ((1.0 - gamma.array()) * log(1.0 - pi.array()) + gamma.array() * log(pi.array())).matrix().sum();
	//cout << "log_density3: " << log_density << endl;
	log_density -= (a_beta + 1.0) * log(sigma2_beta) + b_beta / sigma2_beta;
	//cout << "log_density4: " << log_density << endl;
		
	return log_density;
}

double calcLogLikelihood_em(MatrixXd A, VectorXd u, VectorXd pi)
{

	double log_density = pi.dot(A*u);
	log_density -= log(1.0 + exp((A * u).array())).matrix().sum();

	return log_density;
}

//*************************************************//
//                    MCMC Step                    //
//*************************************************//
void Estep(VectorXd z, SpMat R, VectorXd &beta, VectorXd &POST_gamma, VectorXd pi, double &sigma2_beta, double sigma2_e, double &logLikelihood_hat,
	      int num_sample, int mcmc_iter, int loop) {

	  ///////////////////////////
	 // INITIALIZE PARAMETERS //
	///////////////////////////
	// *burn-in
	int w_step = ceil(0.8*mcmc_iter);
	int s_step = mcmc_iter - w_step;

	// *SNP number 
	int num_snp = R.rows();

	// *parameters of sigma2_beta
	double a_beta = 0.01, b_beta = 0.01;

	  //////////////////
	 // STORE RESULT //
	//////////////////	
	// *beta
	VectorXd mean_beta(num_snp);
	VectorXd var_beta(num_snp);
	VectorXd POST_beta(num_snp);
	POST_beta.fill(0);

	// *postrior distribution
	double POST_nonG = 0.0;
	double POST_sigma2_beta = 0.0;
	double SSR = 0.0;

	// *cumulation of posterior distribution
	VectorXd HIST_logLikelihood((s_step + w_step));
	VectorXd HIST_sigma2_e((s_step + w_step));
	VectorXd HIST_sigma2_beta((s_step + w_step));
	VectorXd HIST_nonzerobeta((s_step + w_step));

	// *probability of gamma (related to pi)
	MatrixXd log_pi(num_snp, 2);
	VectorXd pi_b1(num_snp);
	

	  /////////////////////
	 // FIRST ITERATION //
	/////////////////////
	// *initialize gamma and beta
	VectorXd beta_s = beta;
	VectorXd gamma_s(num_snp);
	boost::mt19937 rng;
	boost::random::uniform_real_distribution<> unif_dist(0, 1);
	boost::random::variate_generator<boost::mt19937&, boost::random::uniform_real_distribution<> > randu(rng, unif_dist);
	for (size_t i = 0; i < num_snp; i++) {

		if (pi(i) > randu()) { gamma_s(i) = 1; } // *pi is the probability of gamma = 1
		else { gamma_s(i) = 0; }
	}
	VectorXd Ebeta = (beta_s.array() * gamma_s.array()).matrix();

	// *update sampling sigma2_beta
	VectorXd Ebeta2 = (Ebeta.array() * Ebeta.array() * gamma_s.array()).matrix();
	double a_beta_tilde = gamma_s.sum() / 2.0 + a_beta;
	double b_beta_tilde = Ebeta2.sum() / (2.0 * sigma2_e) + b_beta;

	
	  ///////////////
	 // MAIN MCMC //
	///////////////
	double tstart1 = clock();
	for (size_t iter = 0; iter < mcmc_iter; iter++) {
		// cout << iter << endl;
		
		// *sampling sigma2_beta from inverse gamma
		boost::random::gamma_distribution<> sigma2_beta_gamma(a_beta_tilde, 1.0 / b_beta_tilde);
		boost::variate_generator<boost::mt19937&, boost::random::gamma_distribution<> > rand_sigma2_beta_gamma(rng, sigma2_beta_gamma);
		sigma2_beta = 1.0 / rand_sigma2_beta_gamma();
		
		// *update the beta in turn, beta from spike and slab
		Ebeta = (beta_s.array() * gamma_s.array()).matrix(); // pair-wise multiple
		VectorXd R_beta = R * Ebeta;
		double z_Rbeta = 0.0, R_sigma = 0.0;
		for (int j = 0; j < num_snp; j++) { 	// *for each snp

			R_beta -= R.col(j) * Ebeta(j);

			// *compute the mean and variance to sample beta from normal distribution
			z_Rbeta = z(j) / sqrt(num_sample) - R_beta(j); // *here, Xbeta has removed ith row
			R_sigma = 1.0 + 1.0 / (sigma2_beta * num_sample);
			mean_beta(j) = z_Rbeta / R_sigma;
			var_beta(j) = sigma2_e / (num_sample * R_sigma);

			// *sampling beta with mean and variance
			boost::random::normal_distribution<> beta_s_normal(mean_beta(j), sqrt(var_beta(j)));
			boost::variate_generator<boost::mt19937&, boost::random::normal_distribution<> > rand_beta_s_normal(rng, beta_s_normal);
			beta_s(j) = rand_beta_s_normal();

			// *for case, gamma = 1
			log_pi(j, 1) = mean_beta(j) * mean_beta(j) / (2.0 * var_beta(j)) + log(sqrt(var_beta(j))) - log(sqrt(sigma2_beta)) - log(sqrt(sigma2_e)) + log(pi(j));

			// *for case, gamma = 0
			log_pi(j, 0) = log(1.0 - pi(j));

			// *new pi
			// log_pi.row(j) = (log_pi.row(j).array() - log_pi.row(j).maxCoeff()).matrix();
			pi_b1(j) = exp(log_pi(j, 1)) / (exp(log_pi(j, 0)) + exp(log_pi(j, 1)));
			if (pi_b1(j) > 0.9999) { pi_b1(j) = 0.9999; }
			if (pi_b1(j) < 0.0001) { pi_b1(j) = 0.0001; }
			
			// cout << "pi_b1: " << pi_b1.size() << endl;
			// for (int i = 0; i < 50; i++) { cout << "pi_b1: "<< pi_b1(i) << endl; }

			// *Bernoulli distribution (MH)
			boost::random::uniform_real_distribution<> unif_dist(0, 1);
			boost::random::variate_generator<boost::mt19937&, boost::random::uniform_real_distribution<> > randu(rng, unif_dist);
			if (pi_b1(j) > randu()) { gamma_s(j) = 1; }
			else { gamma_s(j) = 0; }// *end if

			Ebeta(j) = beta_s(j) * gamma_s(j);

			// add the value back
			R_beta += R.col(j) * Ebeta(j);
		}// *end for each snp

		// *update a_beta, b_beta
		Ebeta2 = (beta_s.array() * beta_s.array() * gamma_s.array()).matrix();
		a_beta_tilde = gamma_s.sum() / 2.0 + a_beta;
		b_beta_tilde = Ebeta2.sum() / (2.0 * sigma2_e) + b_beta;
		
		if (iter >= w_step) {

			POST_beta += Ebeta;
			POST_gamma += gamma_s;
			POST_nonG += gamma_s.sum();
			POST_sigma2_beta += sigma2_beta;
		}

		SSR = num_sample * (1.0 - 2.0 * Ebeta.dot(z) / sqrt(num_sample) + Ebeta.dot(R * Ebeta));
		HIST_logLikelihood(iter) = calcLogLikelihood_mcmc(num_sample, SSR, a_beta, b_beta, sigma2_beta, sigma2_e, beta_s, gamma_s, pi);
		HIST_sigma2_beta(iter) = sigma2_beta;
		HIST_nonzerobeta(iter) = (Ebeta.array() != 0.0).cast<int>().sum();
	}// *end MCMC iteration
	double time_mcmc = (clock() - tstart1) / (double(CLOCKS_PER_SEC));
	// cout << "MCMC time: " << time_mcmc << endl;

	string logL_mcmc_file = "/net/mulan/home/yasheng/summAnnot/simulation/mcmc_result/eigen_result" + to_string(loop) + ".txt";
	ofstream outfile(logL_mcmc_file);
	outfile << "iter" << " " << "sigma2_beta" << " " << "nonzeros_beta" << " " << "logL" <<endl;
	for (size_t i = 0; i<(w_step+s_step); i++)  {
		
		outfile << i << " " << HIST_sigma2_beta[i] << " " << HIST_nonzerobeta[i] << " " << HIST_logLikelihood[i] << endl;
	}
	outfile.close();

	// *calculate posterior distribution
	POST_gamma = (POST_gamma.array() / (double)s_step).matrix();
	POST_beta = (POST_beta.array()/(double)s_step).matrix();
	POST_sigma2_beta /= (double)s_step;

	// *return the posterior distribution
	SSR = num_sample * (1.0 - 2.0 * POST_beta.dot(z) / sqrt(num_sample) + POST_beta.dot(R * POST_beta));
	cout << "SSR: " << SSR << endl;
	beta = POST_beta;
	sigma2_beta = POST_sigma2_beta;
	/*cout << "Posterior distribution is finished!" << endl;*/

	// *log likelihood of MCMC
	logLikelihood_hat = calcLogLikelihood_mcmc(num_sample, SSR, a_beta, b_beta, sigma2_beta,
	                                           sigma2_e, beta, POST_gamma, pi);
	VectorXd logLikelihood_hat2 = (HIST_logLikelihood.tail(s_step).array() - HIST_logLikelihood.tail(s_step).mean()).matrix();
	double pD1 = 2 * (logLikelihood_hat - (HIST_logLikelihood.tail(s_step).sum()) / (double)s_step);
	double pD2 = 2 * (logLikelihood_hat2.dot(logLikelihood_hat2)) / ((double)s_step - 1.0);
	if (pD1 < 0) { pD1 = 1.0; }
	double DIC1, DIC2, BIC1, BIC2;
	DIC1 = -2 * logLikelihood_hat + 2 * pD1;
	DIC2 = -2 * logLikelihood_hat + 2 * pD2;
	BIC1 = -2 * logLikelihood_hat + log(num_sample)*pD1;
	BIC2 = -2 * logLikelihood_hat + log(num_sample)*pD2;
	
	//cout << "MCMC is finished!" << endl;

}// *end function


   //*****************************************************************//
  //                   Updata u (Newton method)                      //
 //*****************************************************************//
void Mstep(MatrixXd A, VectorXd POST_gamma, VectorXd &u) {

	// *POST_gamma is posterior inclusion probability, PIP
	// *compute pi
	VectorXd u_old = u; // *u is coefficent of annotation
	size_t true_iter = 1;
	double citer = 1e-5;
	int nr_iter = 20; // maximum iteration of newton raphson

	//======================
	// Newton method
	while (true_iter) {

		ArrayXd pi = 1.0 / (1.0 + exp((-A * u_old).array()));
		// *Jaccobi matrix, second derivtive
		VectorXd pp = (pi * (1.0 - pi)).matrix();
		MatrixXd J = A.transpose() * col_multiple(pp, A);
		// *update tau
		u = u_old + J.inverse() * (A.transpose() * (POST_gamma - pi.matrix()));
		// cout << "u: " << u << endl;
		// *
		if ((u_old - u).norm() < citer || (++true_iter) > nr_iter) { break; }
		else { u_old = u; }
	}
	// cout << "NR time: " << true_iter << endl;
	return;
}

  //*****************************************************//
 //                  Louis method                       //
//*****************************************************//
MatrixXd calcLouisInfo(VectorXd beta, VectorXd POST_gamma, MatrixXd A, double sigma2_beta, double sigma2_e) {

	int num_snp = A.rows();
	MatrixXd info(A.cols(), A.cols());
	
	//normal distribution
	boost::math::normal_distribution<> normal_dist(0, sigma2_beta * sigma2_e);
	ArrayXd prob_beta(num_snp);
	for (int i = 0; i < num_snp; i++) {
		prob_beta(i) = cdf(normal_dist, beta(i));
	}
	ArrayXd numer = (1 - POST_gamma.array()) * POST_gamma.array() * prob_beta;
	ArrayXd denom = POST_gamma.array() * prob_beta + (1 - POST_gamma.array());
	//vec numer = (1 - POST_gamma) % POST_gamma % normcdf(beta, 0, sigma2_beta * sigma2_e);
	//vec denom = POST_gamma % normcdf(beta, 0, sigma2_beta * sigma2_e) + (1 - POST_gamma);
		
	info(0, 0) = ((numer / (denom * denom + 1e-20) * A.col(0).array() * A.col(0).array()).matrix()).sum();
	info(0, 1) = ((numer / (denom * denom + 1e-20) * A.col(0).array() * A.col(1).array()).matrix()).sum();
	info(1, 0) = info(0, 1);
	info(1, 1) = ((numer / (denom * denom + 1e-20) * A.col(1).array() * A.col(1).array()).matrix()).sum();
	//cout << "Louis method information: " << endl;
	//cout << info << endl;
	return info;
}


// [[Rcpp::export]]
SEXP scGWAS(SEXP zIn, SEXP RIn, SEXP AIn, SEXP sigma2_eIn, SEXP num_sampleIn, SEXP loopIn) {
	
	try {

		  ///////////
		 // INPUT //
		///////////
		VectorXd z = as<VectorXd>(zIn);   // *dim = num_snp x 1, z = X.t() * y / n
		SpMat R = as<SpMat>(RIn); // *dim = num_snp x num_snp, R = X.t() * X 
		MatrixXd A = as<MatrixXd>(AIn);  // *dim = num_snp x num_annot
		double sigma2_e = as<double>(sigma2_eIn);     // *sigma2_e = 1- h2
		int num_sample = Rcpp::as<int>(num_sampleIn);
		int loop = Rcpp::as<int>(loopIn);


		  //////////////////////////
		 // INITIALIZE PARAMETER //
		//////////////////////////
		int num_anno = A.cols() - 1;
		int num_snp = A.rows();
		int em_iter = 8;
		int mcmc_iter = 500;
		double sigma2_beta;

		// *beta 
		VectorXd beta(num_snp);
		boost::mt19937 rng;
		boost::random::normal_distribution<> norm_dist(0, 1e-3);
		boost::random::variate_generator<boost::mt19937&, boost::random::normal_distribution<> > randn(rng, norm_dist);
		for (int i = 0; i < num_snp; i++) { beta(i) = randn(); }
		cout << "Initial beta: " << beta.sum() << endl;
		// *u
		VectorXd u_old(num_anno + 1);
		u_old(0) = -5.0;
		for (int i = 1; i < num_anno; i++) { u_old(i) = -0.2; }
		VectorXd PRIOR_pi = (1.0 / (1.0 + exp((-A * u_old).array()))).matrix();
		cout << "Initial PRIOR_pi: " << PRIOR_pi.sum() << endl;
		VectorXd u = u_old;
		VectorXd u_sigma(num_anno + 1);
		VectorXd u_sigma_Louis(num_anno + 1);

		// *result
		VectorXd POST_gamma(num_snp);
		double SSR = 0.0;
		double logLikelihood_em_old = 0.0, logLikelihood_mcmc = 0.0, logLikelihood_em = 0.0;

		// cout << "MCMC time: " << mcmc_iter << endl;
		// *the main loop of EM algorithm
		for (int iter = 1; iter <= em_iter; iter++) {

			// * only store the last iteration results
			POST_gamma.fill(0);

			  //////////////////
			 // E-step: MCMC //
			//////////////////
			// *updata POST_gamma, sigma2_e and sigma2_beta
			// double tstart1 = clock();
			// cout << beta.sum() << endl;
			Estep(z, R, beta, POST_gamma, PRIOR_pi, sigma2_beta, sigma2_e, logLikelihood_mcmc, num_sample, mcmc_iter, iter);
			// cout << beta.sum() << endl;
			//beta = POST_beta;
			// double time_mcmc = (clock() - tstart1) / (double(CLOCKS_PER_SEC));
			// cout << "sum of POST_gamma: " << POST_gamma.sum() << endl;

			  ////////////////
			 // M-step: NR //
			////////////////			
			// *maximize u
			Mstep(A, POST_gamma, u);

			// *compute the pi
			PRIOR_pi = (1.0 / (1.0 + exp((-A * u).array()))).matrix();
			VectorXd pp = (PRIOR_pi.array() * (1 - PRIOR_pi.array())).matrix();
			MatrixXd info = A.transpose() * pp.asDiagonal() * A;
			MatrixXd invInfo = info.inverse();
			u_sigma = invInfo.diagonal();
			double wald = u(1) / sqrt(u_sigma(1));

			// *compute the Louis information matrix
			MatrixXd Louis_info_inv = calcLouisInfo(beta, POST_gamma, A, sigma2_beta, sigma2_e).inverse();
			u_sigma_Louis = Louis_info_inv.diagonal();

			// *compute the full likelihood here
			double logLikelihood_em = logLikelihood_mcmc + calcLogLikelihood_em(A, u, PRIOR_pi);

			//Converge condition
			double citer_likelihood = abs(logLikelihood_em - logLikelihood_em_old);
			cout << logLikelihood_em << " " << citer_likelihood << " " << POST_gamma.sum() << " " << sigma2_beta << " " << sigma2_e
				 << " " << u(0) << " " << u(1) << " " << u_sigma(1) << " " << u_sigma_Louis(1) << endl;
			if (citer_likelihood < 1) {
				break;
			}
			else {
				logLikelihood_em_old = logLikelihood_em;
				u_old = u;
			}

		}// end for EM iteration

		return List::create(Named("beta") = beta, Named("u") = u, Named("pip") = POST_gamma, Named("u_sigma") = u_sigma, Named("u_sigma_Louis") = u_sigma_Louis,
			                Named("sigma2_e") = sigma2_e, Named("sigma2_beta") = sigma2_beta);
	}
	catch (std::exception &ex) {
		forward_exception_to_r(ex);
	}
	catch (...) {
		::Rf_error("C++ exception (unknown reason)...");
	}
	return R_NilValue;
}
