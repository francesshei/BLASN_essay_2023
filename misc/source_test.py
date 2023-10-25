import numpy as np
import os.path as op
from data_io import EegData

# import nibabel as nib
from scipy import linalg

import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw


DERIVATIVES_FOLDER = "ds004504-download/derivatives"
file = f"{DERIVATIVES_FOLDER}/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")


if __name__ == "__main__":
    raw = EegData.from_set_file(file).eeg_data

    # Read and set the EEG electrode locations, which are already in fsaverage's
    # space (MNI space) for standard_1020:
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)  # needed for inverse modeling

    fwd = mne.make_forward_solution(
        raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
    )
    noise_cov = mne.compute_raw_covariance(raw)
    inverse_operator = make_inverse_operator(
        raw.info, fwd, noise_cov, loose=0.2, depth=0.8
    )

    method = "eLORETA"
    snr = 3.0
    lambda2 = 1.0 / snr**2
    stc = apply_inverse_raw(
        raw,
        inverse_operator,
        lambda2,
        stop=5000,
        method=method,
        pick_ori=None,
        buffer_size=1000,
        verbose=True,
    )
