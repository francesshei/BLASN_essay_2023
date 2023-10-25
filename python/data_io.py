import mne
import numpy as np
import matplotlib.pyplot as plt
from autoreject import AutoReject


def noise_psd(N, psd=lambda f: 1):
    X_white = np.fft.rfft(np.random.randn(N))
    S = psd(np.fft.rfftfreq(N))
    # Normalize S
    S = S / np.sqrt(np.mean(S**2))
    X_shaped = X_white * S
    return np.fft.irfft(X_shaped)


def PSDGenerator(f):
    return lambda N: noise_psd(N, f)


@PSDGenerator
def white_noise(f):
    return 1


@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)


@PSDGenerator
def violet_noise(f):
    return f


@PSDGenerator
def brownian_noise(f):
    return 1 / np.where(f == 0, float("inf"), f)


@PSDGenerator
def pink_noise(f):
    return 1 / np.where(f == 0, float("inf"), np.sqrt(f))


class EegData:
    """Utlity class to perform data processing and entropy computation on EEG signals"""

    def __init__(self, eeg_data, len_segments=None, preload_data=True):
        if len_segments is not None:
            eeg_data = mne.make_fixed_length_epochs(
                eeg_data, duration=len_segments, preload=preload_data
            )
        self.eeg_data = eeg_data
        self.ar = AutoReject()

    @property
    def shape(self):
        """Return the shape of the timeseries data,
        i.e. N x M, with N = num. of channels, M = num. of temporal samples

        :return: Shape of the data array
        :rtype: tuple
        """
        return self.data.shape

    @property
    def data(self):
        return self.eeg_data.get_data()

    @classmethod
    def from_set_file(cls, filename, *args, **kwargs):
        """Initialize a new EegData object from a .set file


        :param filename: string
        :type filename: the path to the desired .set file
        :return: a EegData object storing the data contained in the file
        :rtype: EegData
        """
        raw_data = mne.io.read_raw_eeglab(filename)
        return EegData(raw_data, *args, **kwargs)

    def filter(self, *args, low_cut_f=0.5, high_cut_f=50.0, in_place=True, **kwargs):
        filtered_data = self.eeg_data.filter(
            l_freq=low_cut_f, h_freq=high_cut_f, *args, **kwargs
        )
        if in_place:
            self.eeg_data = filtered_data
        else:
            return filtered_data

    def reject_artifacts(self, in_place=True):
        try:
            clean_data = self.ar.fit_transform(self.eeg_data)
        except Exception as e:
            print(
                f"Make sure to provide segmented raw data, instantiated with the `preload` flag set to `True`! Error: {e}"
            )
        if in_place:
            self.eeg_data = clean_data
        else:
            return clean_data

    def plot(self, channels=None, time=None, cmap_name="viridis"):
        from matplotlib.cm import get_cmap

        if channels is None:
            print("Please provide the indices of the desired channels to plot!")
        else:
            cmap = get_cmap(cmap_name)
            norm_values = np.linspace(0, 1, len(channels))
            _, ax = plt.subplots(len(channels), 1, sharex=True)
            for i, (ch, norm) in enumerate(zip(channels, norm_values)):
                if isinstance(time, int):
                    ax[i].plot(
                        np.array(range(time)),
                        self.data[ch, :time],
                        c=cmap(norm),
                    )
                elif isinstance(time, (tuple, list)):
                    ax[i].plot(
                        time[0] + np.array(range(time[1] - time[0])),
                        self.data[ch, time[0] : time[1]],
                        c=cmap(norm),
                    )
                else:
                    ax[i].plot(
                        np.array(range(self.shape[1])), self.data[ch, :], c=cmap(norm)
                    )
            plt.show()
