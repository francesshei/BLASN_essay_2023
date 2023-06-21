library(tibble)
library(tidyr)
library(purrr)
library(rdist)

# Refined composite multivariate generalized multiscale fuzzy entropy computation
# Ported from MATLAB codes reported in http://dx.doi.org/10.1016/j.physa.2016.07.077 
# Ref:
#   [1] H. Azami and J. Escudero, "Refined Composite Multivariate Generalized Multiscale Fuzzy Entropy:
# A Tool for Complexity Analysis of Multichannel Signals", Physica A, 2016.
# If you use the code, please make sure that you cite reference [1].


RCmvMFE <- function(X, m, r, n, tau, max_scale_factor) {
  # Inputs:
  #   X: multivariate signal - a matrix of size N (the number of channels) x M (the number of sample points for each channel)
  #   m: embedding dimension - scalar 
  #   r: similarity tolerance (it is usually equal to 0.15) - scalar
  #   n: fuzzy power (it is usually equal to 2) - scalar
  #   tau: time delay (it is usually equal to 1) - scalar
  #   max_scale_factor: the number of scale factors - scalar
  
  # Output:
  #   RCmvMFE: RCmvMFE value at each scale factor - an array of size S (max. scale factor)
  
  # Center and scale the input data X, and initialize the algorithm quantities
  X <- scale(t(X))
  r <- r * sum(apply(X, 2, sd))
  M <- rep(m, ncol(X))
  tau <- rep(tau, ncol(X))
  X <- t(X)
  RCmvMFE <- array(NA, dim=max_scale_factor)
  
  # The first element (scale = 1) is the multivariate fuzzy entropy (mvFE)
  result <- mvFE(X, M, r, n, tau)
  RCmvMFE[1] <- result[[1]]
  cat("Finished computation with timescale factor: 1 \n")
  
  # Iteratively compute the mvFE at each time scale factor and append the result
  # to the output vector
  for (i in 2:max_scale_factor) {
    PHI_M <- vector("list", i)
    PHI_M1 <- vector("list", i)
    for (j in 1:i) {
      X_g <- make_coarsegrained(X[,j:ncol(X)], i)
      result <- mvFE(X_g, M, r, n, tau)
      phi_m <- result[[2]]
      phi_m1 <- result[[3]]
      PHI_M[[j]] <- phi_m
      PHI_M1[[j]] <- phi_m1
    }
    PHI_M <- sum(unlist(PHI_M))
    PHI_M1 <- sum(unlist(PHI_M1))
    RCmvMFE[i] <- log(PHI_M / PHI_M1)
    cat("Finished computation with timescale factor:", i)
    cat("\n")
  }
  
  return(RCmvMFE)
}


make_coarsegrained <- function(X, scale_factor) {
  # Generates the consecutive coarse-grained time series based on mean.
  # Inputs: 
  #   X: the original time series - a matrix of size N (the number of channels) x M (the number of sample points for each channel)
  #   scale_factor: the scale factor - scalar
  
  # Ouput:
  #   X_g: the coarse-grained time series - a matrix of size N (the number of channels) x J (lowest integer given by [n_samples / scale_factor])
  
  n_channels <- nrow(X)
  n_samples <- ncol(X)
  J <- floor(n_samples / scale_factor)
  X_g <- matrix(0, nrow = n_channels, ncol = J)
  
  for (j in 1:n_channels) {
    for (i in 1:J) {
      X_g[j, i] <- mean(X[j, ((i-1)*scale_factor + 1):(i*scale_factor)])
    }
  }
  
  return(X_g)
}


mvFE <- function(X, M, r, n, tau) {
  # Computes the multivariate fuzzy entropy (mvFE) of a multivariate signal
  # Inputs:
  #   X: multivariate signal - a matrix of size N (the number of channels) x M (the number of sample points for each channel)
  #   M: embedding vector
  #   r: similarity tolerance (it is usually equal to 0.15) - scalar
  #   n: fuzzy power (it is usually equal to 2) - scalar
  #   tau: time delay (it is usually equal to 1) - scalar
  
  # Output:
  #   mvFE: mvFE value - scalar
  #   phi_m: the global quantity in dimension m - scalar
  #   phi_m1 : the global quantity in dimension m+1 - scalar
  
  # Extract the necessary quantities from the input arguments
  M_max <- max(M)
  tau_max <- max(tau)
  nn <- M_max * tau_max
  
  num_channels <- nrow(X)
  num_samples <- ncol(X)
  N <- num_samples - nn
  
  # Compute the multivariate embedding of the time series in M
  A <- multivariate_embedding(X, M, tau)
  
  # Calculate the Chebyshev between the entries of the multivariate embedding vector
  y <- rdist(A, metric = "maximum") 
  y <- exp((-y^n) / r)
  # Compute the global quantity for M
  phi_m <- sum(y) * 2 / (N * (N - 1))
  
  # Clean unused intermediate variables
  rm(y, A)
  
  # Construct the M+1 vector to embed the time series in new (higher) dimension
  M <- kronecker(matrix(1, num_channels, 1), t(M))
  I <- diag(num_channels)
  M <- M + I
  
  B <- NULL
  
  for (h in 1:num_channels) {
    # Compute the new embedding for an incremental number of channels
    # and stack the results in a new embedding matrix
    B_h <- multivariate_embedding(X, M[h,], tau)
    B <- rbind(B, B_h)
  }
  
  # Calculate the Chebyshev between the entries of the multivariate embedding vector in M+1
  z <- rdist(B, metric = "maximum")
  z <- exp((-z^n) / r)
  # Compute the global quantity for M+1
  phi_m1 <- sum(z) * 2 / (num_channels * N * (num_channels * N - 1))
  
  # The mvFE is the logarithm of the global quantities (by Shannon's theorem)
  mvFE <- log(phi_m / phi_m1)
  
  return(list(mvFE, phi_m, phi_m1))
}


multivariate_embedding <- function(X, M, tau) {
  # Construct the multivariate delay embedded vectors.
  # Inputs:
  #   X: multivariate time series - a matrix of size N_channels X N_samples
  #   M: embedding vector parameter -  an array of size N_channels 
  #   tau: time delays vector -  an array of size N_channels 
  
  # Output:
  #   A: the multivariate embedding matrix
  
  # Ref: M. U. Ahmed and D. P. Mandic, "Multivariate multiscale entropy
  # analysis", IEEE Signal Processing Letters, vol. 19, no. 2, pp.91-94.2012
  
  n_channels <- nrow(X)
  n_samples <- ncol(X)
  
  A <- NULL
  
  for (j in 1:n_channels) {
    embedded_row <- list()
    for (i in 1:(n_samples - max(M))) {
      # Each row of the multivariate embedding matrix is constructed as 
      # xj, xj + tau[j] , . . . , xj + (M[j]−1)*tau[j] 
      embedded_row[[i]] <- X[j, seq(from=i, to=(i+M[j]-1), by=tau[j])]
    }
    embedded_row <- do.call(cbind, embedded_row)
    # Construct the matrix piling the rows
    A <- rbind(A, embedded_row)
  }
  return(t(A))
}


# Experimental optimized version (TODO: see profiling)
tibble_multivariate_embedding <- function(X, M, tau) {
  n_channels <- nrow(X)
  n_samples <- ncol(X)
  
  # Convert the data to a tibble to avoid looping
  x_t <- as.tibble(t(X))
  
  # This map function computes a nested list of lists.
  # One list for each channel j holding a list for each sample i so that: 
  # embedded_nested_list[j] = [[xi, xi + tau[j] , . . . , xi + (M[j]−1)*tau[j]], [...],]
  embedded_nested_list <- 
    purrr::map(rep(1:n_channels), 
               function(y) map(rep(1:(n_samples - max(M))), 
                               function(x) as.numeric(t(x_t[x:tau[y]:(x+M[y]-1),1]))
               )
    )
  # Each outer list of the nested list is stacked horizontally (each list becomes a column)
  # and the numerical values contained in each list are expanded horizontally so that
  # the total number of columns is len(sample_list) * num_channels
  unnested_t <- unnest_wider(as.tibble(do.call(cbind,embedded_nested_list)), everything(), names_sep = '-')
  
  # Convert the tibble to matrix and return it
  return (as.matrix(unnested_t))
}


