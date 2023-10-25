import os
import ordpy
import csv
import numpy as np
import scipy.io


ANTERIOR = [0, 2, 16, 1, 3]
CENTRAL = [4, 5, 17]
TEMPORAL_LEFT = [12, 14, 10]
TEMPORAL_RIGHT = [13, 15, 11]
POSTERIOR = [6, 8, 18, 7, 9]

DERIVATIVES_FOLDER = "ds004504-download/derivatives"


def compute_permutation_entropy(time_serie, normalize=True, *args, **kwargs):
    if normalize:
        time_serie -= np.mean(time_serie, axis=0)
        time_serie /= np.std(time_serie, axis=0)
    return ordpy.permutation_entropy(time_serie, *args, **kwargs)


def compute_tsallis_entropy(time_serie, normalize=True, *args, **kwargs):
    if normalize:
        time_serie -= np.mean(time_serie, axis=0)
        time_serie /= np.std(time_serie, axis=0)
    return ordpy.tsallis_entropy(time_serie, *args, **kwargs)


def compute_renyi_entropy(time_serie, normalize=True, *args, **kwargs):
    if normalize:
        time_serie -= np.mean(time_serie, axis=0)
        time_serie /= np.std(time_serie, axis=0)
    return ordpy.renyi_entropy(time_serie, *args, **kwargs)


if __name__ == "__main__":
    # Read the participants information
    with open("ds004504-download/participants.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        participants = list(tsv_file)
    classes = {participant[0]: participant[-2] for participant in participants[1:]}
    available_sujbects = os.listdir(DERIVATIVES_FOLDER)
    clustered_files = {"A": [], "C": [], "F": []}
    for k, v in classes.items():
        clustered_files[v].append(
            f"{DERIVATIVES_FOLDER}/{k}/eeg/{k}_task-eyesclosed_eeg.set"
        )

    # sub-001_task-eyesclosed_eeg
    # Alzheimer's patients
    ad_files = clustered_files["A"]
    ad_entropies = np.zeros((len(ad_files),), dtype=float)
    for i, file in enumerate(ad_files):
        print("Extracting entropy for a new AD patient")
        eeg_data = scipy.io.loadmat(file)
        sources_data = eeg_data["data"][CENTRAL, :]
        ad_entropies[i] = compute_permutation_entropy(sources_data)

    control_files = clustered_files["C"]
    control_entropies = np.zeros((len(control_files),), dtype=float)
    for i, file in enumerate(control_files):
        print("Extracting entropy for a new control patient")
        eeg_data = scipy.io.loadmat(file)
        sources_data = eeg_data["data"][CENTRAL, :]
        control_entropies[i] = compute_permutation_entropy(sources_data)
