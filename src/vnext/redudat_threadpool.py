import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_calibration(filename):
    data = np.loadtxt(filename)
    if data.shape[0] < data[:, 0].max():
        det_difc = np.zeros((int(data[:, 0].max() + 1), 3))
        det_difc[data[:, 0].astype(np.int64), 0] = np.squeeze(data[:, 0])
        det_difc[data[:, 0].astype(np.int64), 1] = np.squeeze(data[:, 1])
        det_difc[data[:, 0].astype(np.int64), 2] = np.squeeze(data[:, 2])
    return det_difc


def get_difc(event_idx):
    det_difc = load_calibration("B123456DIFCs-12Cross-3456Cal.txt")
    return det_difc[event_idx, 1]


def histogram(tof, raw_data):
    return np.histogram(raw_data, tof)[0]


def reduce_banks(bank, event, event_tof):
    tofbin = 1250
    event_tof = event_tof / get_difc(event)
    # print(event_tof)
    tofmin = event_tof.min()
    tofmax = event_tof.max()
    tof = np.linspace(tofmin, tofmax, tofbin + 1)
    # intsty = np.zeros(tofbin)
    intsy = histogram(tof, event_tof)
    tvi = np.zeros([tofbin, 3])
    tof_f = tof[:-1] - (tof[1] - tof[0]) / 2.0
    tvi[:, 0] = bank
    tvi[:, 1] = tof_f
    tvi[:, 2] = intsy
    return tvi


@contextmanager
def locked_file():
    with HDF_LOCK:
        with h5py.File(HDF_PATH, "r") as file:
            yield file


def process_files(rmin, rmax):
    with locked_file() as db:
        # timestamp = file.attrs.get('event_id')
        # dataset = file.get('entry')
        bankn = np.arange(1, 7)
        tmp = []
        with ProcessPoolExecutor(6) as executor:
            for bank in bankn:
                pulse_index = np.array(db["entry"][f"bank{bank}_events"]["event_index"][rmin:rmax])
                pulse_id = np.array(
                    db["entry"][f"bank{bank}_events"]["event_id"][int(pulse_index[0]) : int(pulse_index[-1])]
                )
                pulse_tofset = np.array(
                    db["entry"][f"bank{bank}_events"]["event_time_offset"][pulse_index[0] : pulse_index[-1]]
                )
                tmp.append(executor.submit(reduce_banks, bank, pulse_id, pulse_tofset))
        return np.stack([mex.result() for mex in tmp], axis=0)


if __name__ == "__main__":
    time_o = time.time()

    tstart = 0
    tend = 220
    invp = 1 / 60.0
    tsindex = int((tstart / invp))
    teindex = int((tend / invp))
    HDF_LOCK = threading.Lock()
    HDF_PATH = "VULCAN_217968.nxs.h5"
    data = process_files(tsindex, teindex)
    print(data.shape)
    for j in range(len(data)):
        plt.figure(data[j, 0, 0])
        plt.plot(data[j, :, 1], data[j, :, 2])

    print(f" hello time{time.time()-time_o}")
    plt.show()
