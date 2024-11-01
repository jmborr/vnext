# import h5rdmtoolbox as h5tbx
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI


def get_difc(event_idx):
    return det_difc[event_idx, 1]


def load_calibration(filename):
    if ".txt" in filename:
        data = np.loadtxt("B123456DIFCs-12Cross-3456Cal.txt")
        if data.shape[0] < data[:, 0].max():
            det_difc = np.zeros((int(data[:, 0].max() + 1), 3))
            det_difc[data[:, 0].astype(np.int32), 0] = np.squeeze(data[:, 0])
            det_difc[data[:, 0].astype(np.int32), 1] = np.squeeze(data[:, 1])
            det_difc[data[:, 0].astype(np.int32), 2] = np.squeeze(data[:, 2])
        return det_difc


def histogram(tof, raw_data):
    return np.histogram(raw_data, tof)[0]


def reduce_banks(event, event_tof, tofmin=None, tofmax=None, tofbin=1250):
    event_tof /= get_difc(event)
    # print(event_tof)
    if tofmin is None:
        tofmin = event_tof.min()
    if tofmax is None:
        tofmax = event_tof.max()
    tof = np.linspace(tofmin, tofmax, tofbin + 1)
    # intsty = np.zeros(tofbin)
    intsy = histogram(tof, event_tof)
    tvi = np.zeros([tofbin, 2])
    tof_f = tof[:-1] - (tof[1] - tof[0]) / 2.0

    tvi[:, 0] = tof_f
    tvi[:, 1] = intsy
    return tvi


def return_spectra(db, bank, rmin, rmax):
    print("bank", bank)
    pulse_index = np.array(db["entry"][f"bank{bank}_events"]["event_index"][rmin:rmax])
    # tmparr=int(np.arange(pulse_index.min(), pulse_index.max()))
    pulse_id = np.array(db["entry"][f"bank{bank}_events"]["event_id"][int(pulse_index[0]) : int(pulse_index[-1])])
    pulse_tofset = np.array(db["entry"][f"bank{bank}_events"]["event_time_offset"][pulse_index[0] : pulse_index[-1]])
    spec = reduce_banks(pulse_id, pulse_tofset)
    rspec = np.zeros([len(spec), 3])
    rspec[:, 0] = bank
    rspec[:, 1] = spec[:, 0]
    rspec[:, 2] = spec[:, 1]
    return rspec


def bankbreak(bank, size):
    return [bank[_i::size] for _i in range(size)]


# print(h5tbx.get_config())
time_o = time.time()
# db=h5py.File('VULCAN_246340.nxs.h5','r')

db = h5py.File("VULCAN_217968.nxs.h5", "r")
det_difc = load_calibration("B123456DIFCs-12Cross-3456Cal.txt")
det_difc = load_calibration("B123456DIFCs-12Cross-3456Cal.txt")
# print(det_difC)
# pulse_time=np.array(db['entry']['bank1_events']['event_time_zero'][()])
# print(' hello aa time{}'.format(time.time()-time_o))
tstart = 0
tend = 2205
invp = 1 / 60.0
tsindex = int((tstart / invp))
teindex = int((tend / invp))
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
allspec = []
if rank == 0:
    banks = np.arange(1, 7, 1)
    banks = bankbreak(banks, size)
else:
    banks = None
jobs = comm.scatter(banks, root=0)
for job in jobs:
    spec = return_spectra(db, job, tsindex, teindex)
    allspec.append(spec)
allspec = MPI.COMM_WORLD.gather(allspec, root=0)
if rank == 0:
    allspec = [_i for temp in allspec for _i in temp]
    specall = np.array(allspec)
    for i in range(len(specall)):
        banknumber = np.unique(specall[i, :, 0])[0]
        plt.figure(banknumber)
        plt.suptitle("Bank Number " + str(int(banknumber)))
        plt.plot(specall[i, :, 1], specall[i, :, 2])
    print(f" hello time{time.time()-time_o}")
    plt.show()
