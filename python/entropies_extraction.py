import csv
import h5py
import ordpy
import numpy as np

ANTERIOR = [0, 2, 16, 1, 3]
CENTRAL = [4, 5, 17]
TEMPORAL_LEFT = [12, 14, 10]
TEMPORAL_RIGHT = [13, 15, 11]
POSTERIOR = [6, 8, 18, 7, 9]
N_SCALES = 20


def compute_mspe_ordpy(data, scales=20, channel=2):
    def make_coarsegrained(X, scale_factor):
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

    mean_mspe = []

    for i in range(data.shape[0]):
        # print("New segment")
        trial_entropies = []
        segment_data = data[i, channel, :]
        for scale in range(1, (scales + 1)):
            # print(f"Computing scale: {scale}")
            if isinstance(channel, int):
                segment_data = segment_data.reshape(-1, 1)
            coarse_data = make_coarsegrained(segment_data.T, scale)
            trial_entropies.append(
                ordpy.weighted_permutation_entropy(coarse_data.T, dx=5)
            )
        mean_mspe.append(trial_entropies)

    return np.mean(np.array(mean_mspe), axis=0)


if __name__ == "__main__":
    # Read the participants information
    with open("ds004504-download/participants.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        participants = list(tsv_file)[1:]

    # Open the preprocessed EEG here
    processed_eeg_file = h5py.File("data/preprocessed_subjects.hdf5", "r")
    # Initialize the entropies file
    entropies_file = h5py.File("data/extracted_entropies.hdf5", "w")

    # processed_eeg_file = h5py.File("preprocessed_subjects.hdf5", "r")
    for participant in participants:
        subject_id = participant[0]
        print(f"Computing entropies for subject: {subject_id}")
        eeg_data = np.array(processed_eeg_file[subject_id])

        anterior_mspe = compute_mspe_ordpy(eeg_data, scales=N_SCALES, channel=ANTERIOR)
        entropies_file.create_dataset(f"{subject_id}_anterior", data=anterior_mspe)

        central_mspe = compute_mspe_ordpy(eeg_data, scales=N_SCALES, channel=CENTRAL)
        entropies_file.create_dataset(f"{subject_id}_central", data=central_mspe)

        tl_mspe = compute_mspe_ordpy(eeg_data, scales=N_SCALES, channel=TEMPORAL_LEFT)
        entropies_file.create_dataset(f"{subject_id}_temporal_left", data=tl_mspe)

        tr_mspe = compute_mspe_ordpy(eeg_data, scales=N_SCALES, channel=TEMPORAL_RIGHT)
        entropies_file.create_dataset(f"{subject_id}_temporal_right", data=tr_mspe)

        posterior_mspe = compute_mspe_ordpy(
            eeg_data, scales=N_SCALES, channel=POSTERIOR
        )
        entropies_file.create_dataset(f"{subject_id}_posterior", data=posterior_mspe)

        print(f"Finished entropies extraction for subject: {subject_id}")

    print("Entropies extracted")
