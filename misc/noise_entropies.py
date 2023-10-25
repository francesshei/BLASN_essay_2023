import math
import matplotlib.pyplot as plt
import ordpy
import numpy as np
import EntropyHub as eh
from rcmvmfe import py_rcmvmfe, py_make_coarsegrained
from data_io import *
from multivariate_multiscale_permutation_entropy import mmpe


def compute_msse(data, scales=20):
    entropies = []

    Mobj = eh.MSobject("SampEn", r=0.15)

    for i in range(5):
        print("New entropies computation")
        msx, _ = eh.MSEn(data, Mobj, Scales=scales)
        entropies.append(msx)

    return np.mean(np.array(entropies), axis=0)


def compute_mspe(data, scales=20):
    entropies = []

    Mobj = eh.MSobject("PermEn", m=4, Norm=True)

    for i in range(5):
        print("New entropies computation")
        msx, _ = eh.MSEn(data, Mobj, Scales=scales)
        entropies.append(msx)

    return np.mean(np.array(entropies), axis=0)


def compute_mspe_ordpy(data, scales=20, n_trials=10):
    mean_mspe = []

    for trial in range(n_trials):
        print("New trial")
        trial_entropies = []
        for scale in range(1, (scales + 1)):
            # print(f"Computing scale: {scale}")
            coarse_data = py_make_coarsegrained(data.reshape(-1, 1), scale)
            trial_entropies.append(
                ordpy.permutation_entropy(coarse_data.T, dx=4, normalized=False)
            )
        mean_mspe.append(trial_entropies)

    return np.mean(np.array(mean_mspe), axis=0)


if __name__ == "__main__":
    # Initialize the noises
    num_samples = 40000
    WN = white_noise(num_samples)
    PN = pink_noise(num_samples)
    BN = brownian_noise(num_samples)

    # white_en = compute_mspe(WN)
    # pink_en = compute_mspe(PN)
    # brown_en = compute_mspe(BN)

    white_en = compute_mspe_ordpy(WN)
    pink_en = compute_mspe_ordpy(PN)
    brown_en = compute_mspe_ordpy(BN)

    plt.plot(range(20), white_en, range(20), pink_en, range(20), brown_en)
    plt.show()
