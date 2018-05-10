#include <iostream>
#include <fstream>
#include <list>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

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
using namespace arma;
using namespace Rcpp;


#define ARMA_DONT_PRINT_ERRORS

//************************************************************//
//  Single Cell Matrix Factorization using ADMM   //
//************************************************************//
// [[Rcpp::export]]
SEXP scMF_cpp(SEXP Xin, SEXP Win, SEXP Hin, SEXP criteriain, SEXP rhoin, SEXP num_iterin){
	try {
		mat X = as<mat>(Xin);  // *dim = num_cell x num_gene
		mat W = as<mat>(Win);  // *dim = num_cell x num_pc
		mat H = as<mat>(Hin);  // *dim = num_pc x num_gene

		// *coefficient of regularization
		double rho = Rcpp::as<double>(rhoin);
		int criteria = Rcpp::as<int>(criteriain);
		int num_iter = Rcpp::as<int>(num_iterin);
		
		int num_pc = W.n_cols;
		mat E(num_pc, num_pc, fill::eye);

		// *initial values
		mat Xprime = W*H;
		mat Wplus = W;
		mat Hplus = H;
		
		mat Xdual = zeros<mat>(size(Xprime));
		mat Wdual = zeros<mat>(size(W));
		mat Hdual = zeros<mat>(size(H));
		
		int iter = 0;
		mat ZH = zeros<mat>(size(H));
		mat ZW = zeros<mat>(size(W));
		while(iter < num_iter){
	
			// *updating H
			H = inv_sympd(W.t()*W + E) * (W.t()*Xprime + Hplus + 1/rho *( W.t()*Xdual - Hdual) );
			
			
			// *updating W
			mat P = H*H.t() + E;
			W = (Xprime*H.t() + Wplus + 1/rho *(Xdual*H.t() - Wdual)) * inv_sympd(P);
			
			// *updating X
			mat Xtmp = W*H;
			if(criteria == 0){ // *KL divergence
				mat B = rho*Xtmp - Xdual -1;
				Xprime = ( B + sqrt(B%B + 4.0*rho*X) )/(2.0*rho);
			}else if(criteria == 1){ // *IS divergence
				mat A = Xdual/rho - Xtmp;
				mat B = 1.0/(3*rho) - A%A/9.0;
				mat C = -A%A%A/27.0 + A/(6.0*rho) + X/(2.0*rho);
				mat D = B%B%B + C%C;
				
				for(size_t i=0;i<X.n_rows;i++){
					for(size_t j=0;j<X.n_cols;j++){
						if(D(i,j)>=0){
							Xprime(i,j) = cbrt( C(i,j) + sqrt(D(i,j))) +
										  cbrt( C(i,j) - sqrt(D(i,j))) -
										  A(i,j)/3.0;
						}else{
							Xprime(i,j) = 2.0*sqrt( -B(i,j))*cos( acos(C(i,j)/sqrt(-B(i,j)*B(i,j)*B(i,j))) /3.0 ) - 
				                           A(i,j)/3.0;
						}
						
					}
				}
			}else if(criteria == 2){// *lee's multiplicative update with least square loss
				cout<<"add it later ..." <<endl;
			}
			
			// *updating Hplus, Wplus
			Hplus = arma::max( H + 1.0/rho*Hdual, ZH);
			Wplus = arma::max( W + 1.0/rho*Wdual, ZW);
			
			// *updating dual variables
			Xdual = Xdual + rho*(Xprime - Xtmp);
			Hdual = Hdual + rho*(H - Hplus);
			Wdual = Wdual + rho*(W - Wplus);
			
			// *update iterator
			iter++;
		}// *end while
		
		
		return List::create(Named("W") = Wplus, Named("H") = Hplus );
	} catch( std::exception &ex ) {
		forward_exception_to_r( ex );
	} catch(...) {
		::Rf_error( "C++ exception (unknown reason)..." );
	}
	return R_NilValue;
}
