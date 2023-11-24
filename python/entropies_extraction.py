import csv
import h5py
from fastdist import fastdist
import numpy as np

# ANTERIOR = [0, 2, 16, 1, 3]
# CENTRAL = [4, 5, 17]
# TEMPORAL_LEFT = [12, 14, 10]
# TEMPORAL_RIGHT = [13, 15, 11]
# POSTERIOR = [6, 8, 18, 7, 9]
N_SCALES = 20
N_CHANNELS = 19


def multiscale_sample_entropy(x, scale, m=2, r=0.2):
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
    # Read the participants information
    with open("ds004504-download/participants.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        participants = list(tsv_file)[1:]

    # Open the preprocessed EEG here
    processed_eeg_file = h5py.File("data/preprocessed_subjects.hdf5", "r")
    # Initialize the entropies file
    entropies_file = h5py.File("data/extracted_mss_entropies.hdf5", "w")

    # processed_eeg_file = h5py.File("preprocessed_subjects.hdf5", "r")
    for participant in participants:
        subject_id = participant[0]
        print(f"Computing entropies for subject: {subject_id}")
        eeg_data = np.array(processed_eeg_file[subject_id])
        mss_entropy = []

        n_segments, n_channels, n_samples = eeg_data.shape
        for scale in range(1, N_SCALES + 1):
            print(f"Scale: {scale}")
            segment_entropies = []
            for segment in range(n_segments):
                print(f"Segment {segment} out of {n_segments}")
                channel_entropies = []
                for channel in range(N_CHANNELS):
                    channel_entropies.append(
                        multiscale_sample_entropy(
                            eeg_data[segment, channel, :], scale, r=0.1
                        )
                    )
                segment_entropies.append(channel_entropies)
            mss_entropy.append(np.mean(np.array(segment_entropies), axis=0))

        entropies_file.create_dataset(f"{subject_id}", data=np.array(mss_entropy))

        print(f"Finished entropies extraction for subject: {subject_id}")

    print("Entropies extracted")
