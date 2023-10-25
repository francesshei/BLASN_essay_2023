import numpy as np
import pandas as pd
from scipy.spatial import distance

# from fastdist import fastdist


def compute_subect_rcmvmfe(
    dataframe, cluster=None, samples_per_segment=5000, rcmvmfe_parameters={}
):
    # Helper function to compute the RCmvMFE values for a single patient. It receives
    # the tibble dataframe as input and returns a list of entropies, each of len equal to the max scale
    # factor

    # Read the algorithm parameteres from the dictionary passed as input
    m = rcmvmfe_parameters.get("m", 2)
    tau = rcmvmfe_parameters.get("tau", 1)
    n = rcmvmfe_parameters.get("n", 2)
    r = rcmvmfe_parameters.get("r", 0.15)
    max_scale_factor = rcmvmfe_parameters.get("max_scale_factor", 20)

    # Iterate through the patient's data frame and compute the entropy values
    df = pd.DataFrame.from_dict(dataframe)
    num_chunks = np.ceil(len(df) / samples_per_segment)

    entropies = []
    for i in range(int(num_chunks)):
        print(f"Starting a new chunk: {i + 1} out of {int(num_chunks)}", flush=True)
        data = (np.array_split(df[cluster], num_chunks)[i]).to_numpy()
        # Scale the matrix
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0)
        # Compute the entropy
        chunk_entropy = py_rcmvmfe(data, m, r, n, tau, max_scale_factor)
        entropies.append(chunk_entropy)
        print("Chunk terminated!", flush=True)

    return entropies


def py_rcmvmfe(X, m, r, n, tau, max_scale_factor, scale=False):
    # Inputs:
    #   X: multivariate signal - a matrix of size N (the number of channels) x M (the number of sample points for each channel)
    #   NOTE: it must be scaled and center before passing it as argument
    #   m: embedding dimension - scalar
    #   r: similarity tolerance (it is usually equal to 0.15) - scalar
    #   n: fuzzy power (it is usually equal to 2) - scalar
    #   tau: time delay (it is usually equal to 1) - scalar
    #   max_scale_factor: the number of scale factors - scalar

    # Output:
    #   RCmvMFE: RCmvMFE value at each scale factor - an array of size S (max. scale factor)

    # Initialize the algorithm quantities
    if scale:
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)

    r = r * np.sum(np.apply_along_axis(np.std, axis=0, arr=X))
    M = np.tile(m, (X.shape[1],)).astype(int)
    tau = np.tile(tau, (X.shape[1],)).astype(int)

    # Initialize the empty array holding the different values for each scale factor
    RCmvMFE = np.empty((int(max_scale_factor), 1))

    # The first element (scale = 1) is the multivariate fuzzy entropy (mvFE)
    rcmvmfe_1, _, _ = py_mvFE(X, M, r, n, tau)
    RCmvMFE[0] = rcmvmfe_1
    print("Finished computation with timescale factor: 1", flush=True)

    # Iteratively compute the mvFE at each time scale factor and append the result
    # to the output vector
    for i in range(2, int(max_scale_factor) + 1):
        PHI_M = []
        PHI_M1 = []
        for j in range(i):
            X_g = py_make_coarsegrained(X[j:, :], i)
            _, phi_m, phi_m1 = py_mvFE(X_g, M, r, n, tau)
            PHI_M.append(phi_m)
            PHI_M1.append(phi_m1)
        PHI_M = np.sum(np.array(PHI_M))
        PHI_M1 = np.sum(np.array(PHI_M1))
        RCmvMFE[i - 1] = np.log(PHI_M / PHI_M1)
        print(f"Finished computation with timescale factor: {i}", flush=True)

    return RCmvMFE


def py_make_coarsegrained(X, scale_factor):
    # Generates the consecutive coarse-grained time series based on mean.
    # Inputs:
    #   X: the original time series - a matrix of size N (the number of channels) x M (the number of sample points for each channel)
    #   scale_factor: the scale factor - scalar

    # Ouput:
    #   X_g: the coarse-grained time series - a matrix of size N (the number of channels) x J (lowest integer given by [n_samples / scale_factor])

    if len(X.shape) > 1:
        n_samples, n_channels = X.shape
    else:
        n_samples = len(X)
        n_channels = 1

    J = np.floor(n_samples / scale_factor).astype(int)
    X_g = np.zeros((n_channels, J))

    for j in range(n_channels):
        for i in range(1, J):
            X_g[j, i] = np.mean(X[((i - 1) * scale_factor) : (i * scale_factor), j])
    return np.transpose(X_g)


def py_mvFE(X, M, r, n, tau):
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
    M_max = np.max(M)
    tau_max = np.max(tau)
    nn = M_max * tau_max

    num_samples, num_channels = X.shape
    N = num_samples - nn

    # Compute the multivariate embedding of the time series in M
    A = py_multivariate_embedding(X, M, tau)

    # Calculate the Chebyshev between the entries of the multivariate embedding vector
    y = distance.pdist(A, "chebyshev")
    # y = fastdist.matrix_pairwise_distance(A, fastdist.chebyshev, "chebyshev")
    y = np.exp((-(y**n)) / r)
    # Compute the global quantity for M
    phi_m = np.sum(y) * 2 / (N * (N - 1))

    # Construct the M+1 vector to embed the time series in new (higher) dimension
    M = np.tile(M, (num_channels, 1))
    I = np.eye(num_channels, dtype=int)
    M += I

    B = np.vstack(
        [py_multivariate_embedding(X, M[h,], tau) for h in range(num_channels)]
    )

    # Calculate the Chebyshev between the entries of the multivariate embedding vector in M+1
    z = distance.pdist(B, "chebyshev")
    # z = fastdist.matrix_pairwise_distance(B, fastdist.chebyshev, "chebyshev")
    z = np.exp((-(z**n)) / r)
    # Compute the global quantity for M+1
    phi_m1 = np.sum(z) * 2 / (num_channels * N * (num_channels * N - 1))

    # The mvFE is the logarithm of the global quantities (by Shannon's theorem)
    mvFE = np.log(phi_m / phi_m1)

    return [mvFE, phi_m, phi_m1]


def py_multivariate_embedding(X, M, tau):
    # Construct the multivariate delay embedded vectors.
    # Inputs:
    #   X: multivariate time series - a matrix of size N_channels X N_samples
    #   M: embedding vector parameter -  an array of size N_channels
    #   tau: time delays vector -  an array of size N_channels

    # Output:
    #   Multivariate embedding matrix

    # Ref: M. U. Ahmed and D. P. Mandic, "Multivariate multiscale entropy
    # analysis", IEEE Signal Processing Letters, vol. 19, no. 2, pp.91-94.2012

    n_samples, n_channels = X.shape
    sample_ids = np.arange(n_samples - np.max(M)).astype(int)

    embedded_channels = []

    for n_ch in range(n_channels):
        time_lags = np.arange(stop=M[n_ch], step=tau[n_ch]).astype(int)
        embedding_indices = np.hstack([sample_ids[:, None] + lag for lag in time_lags])
        embedded_channels.append(X[embedding_indices, n_ch])

    return np.hstack(embedded_channels)
