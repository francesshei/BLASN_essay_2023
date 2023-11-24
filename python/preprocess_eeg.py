import csv
import h5py

# from mpi4py import MPI
from data_io import EegData

# Data folder
DATA_FOLDER = "ds004504-download"

# rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
f = h5py.File("data/preprocessed_subjects.hdf5", "w")


# d = mne.filter.filter_data(a_eeg.data, 500, l_freq=1.0, h_freq=None, filter_length=2499)
# d = mne.filter.notch_filter(a_eeg.data, 500, 50.0)


def process_eeg(filename, *args, **kwargs):
    eeg_data = EegData.from_set_file(filename, *args, **kwargs)
    eeg_data.reject_artifacts()
    eeg_data.base_filter(low_cut_f=0.5, high_cut_f=45.0)
    subject_id = filename.split("/")[1]
    f.create_dataset(subject_id, data=eeg_data.data)
    print(f"Created dataset for {subject_id}")  # return eeg_data


if __name__ == "__main__":
    # Read the participants information
    with open(f"{DATA_FOLDER}/participants.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        participants = list(tsv_file)

    # Populate the list of available .set files
    files = [
        f"{DATA_FOLDER}/{p[0]}/eeg/{p[0]}_task-eyesclosed_eeg.set"
        for p in participants[1:]
    ]

    # Process the EEG and store the processing result in parallel
    for file in files:
        process_eeg(file, len_segments=10.0)
