import numpy as np
import seaborn
from fastdist import fastdist
import matplotlib.pyplot as plt
from data_io import EegData, pink_noise, white_noise

seaborn.set(style="ticks")


def multiscale_sample_entropy(x, scale, m=2, r=0.15):
    # Coarse grain the process
    try:
        y = np.mean(np.reshape(x, (-1, scale)), axis=1)
    # pylint:disable=bare-except
    except:
        x = np.pad(x, (0, scale - (len(x) % scale)), mode="constant")
        y = np.mean(np.reshape(x, (-1, scale)), axis=1)

    # Multi-variate embedding
    X = np.array([y[i : i + m + 1] for i in range(len(y) - m)])
    # Compute the pairwise distances
    A = np.sum(
        fastdist.matrix_pairwise_distance(X, fastdist.chebyshev, "chebyshev")
        < r * np.nanstd(x, axis=0)
    )
    # matching m-element sequences
    X = X[:, :m]
    B = np.sum(
        fastdist.matrix_pairwise_distance(X, fastdist.chebyshev, "chebyshev")
        < r * np.nanstd(x, axis=0)
    )

    # take log
    if A == 0 or B == 0:
        e = np.nan
        return e
    else:
        e = np.log(B / A)
        return e


if __name__ == "__main__":
    # Generate white noise and pink noise time series
    pn = pink_noise(30000)
    wn = white_noise(30000)

    # Compute the multiscale entropy and plot the values
    wn_entropies = []
    pn_entropies = []
    for scale in range(1, 21):
        print(f"Computing scale n° {scale}")
        pn_entropies.append(multiscale_sample_entropy(pn, scale))
        wn_entropies.append(multiscale_sample_entropy(wn, scale))

    # Plot the result
    fig, ax = plt.subplots()
    ax.plot(np.array(range(20)), pn_entropies)
    ax.plot(np.array(range(20)), wn_entropies)
    ax.grid(True, which="both")
    seaborn.despine(ax=ax, offset=0)  # the important part here
    plt.xlabel("Scale")
    plt.ylabel("SE")
    plt.show()
