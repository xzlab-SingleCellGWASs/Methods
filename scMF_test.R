
num_data = 1
num_pc = 5
num_iter = 100
X <- list(X1 = matrix(runif(20*100),ncol=20) );

# initializations
H0 <- list(H1 = matrix(runif(num_pc*20),ncol=20) );
W0 <- list(W1 = matrix(runif(num_pc*100),ncol=num_pc) );
U0 <- matrix(runif(num_pc*100,0,0),ncol=num_pc)
criteria = 0; # set criteria=1 for KL divergence, criteria=0 for IS divergence
rho = 1; # ADMM parameter
library("Rcpp")
sourceCpp("~/scMF.cpp")

res_s <- scMF_cpp(X$X1, W0$W1, H0$H1, criteria, rho, num_iter)
