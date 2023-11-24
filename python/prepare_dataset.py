import h5py
import csv
import numpy as np

mss = h5py.File("data/extracted_mss_entropies.hdf5", "r")

with open("ds004504-download/participants.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    participants = list(tsv_file)[1:]

with open("ds004504-download/channels.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    channels = list(tsv_file)[1:]

# Prepare the pandas dataset
with open("data/dataset.csv", "w") as file:
    writer = csv.writer(
        file,
        delimiter=",",
        lineterminator="\n",
    )
    writer.writerow(
        [
            "id",
            "entropy",
            "scale",
            "channel",
            "sex",
            "age",
            "group",
            "MMSE",
        ]
    )
    for i in range(len(participants)):
        _id, sex, age, group, mmse = participants[i]
        print(_id)
        data = np.array(mss[_id])
        n_scales, n_channels = data.shape
        for j in range(n_scales):
            for k in range(n_channels):
                writer.writerow(
                    [
                        _id,
                        data[j][k],
                        j + 1,
                        channels[k][0],
                        sex,
                        int(age),
                        group,
                        int(mmse),
                    ]
                )
