# standard imports
import os
import time
from os.path import dirname

# third-party imports
import h5py
import numpy as np
from mpi4py import MPI

# vnext imports
#


REPO_DIR = os.path.dirname(dirname(dirname(os.path.abspath(__file__))))
VULCAN_BANK_NUMBERS = np.arange(1, 7, 1)


def get_difc(event_idx):
    return det_difc[event_idx, 1]


def load_calibration(filename):
    if ".txt" in filename:
        data = np.loadtxt(filename)
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
    bank_events = db["entry"][f"bank{bank}_events"]

    pulse_index = bank_events["event_index"][rmin:rmax]
    begin, end = int(pulse_index[0]), int(pulse_index[-1])

    pulse_id = np.array(bank_events["event_id"][begin:end])
    pulse_tofset = np.array(bank_events["event_time_offset"][begin:end])

    print(f"Size for bank {bank}: {pulse_id.size}")

    spec = reduce_banks(pulse_id, pulse_tofset)
    rspec = np.zeros([len(spec), 3])
    rspec[:, 0] = bank
    rspec[:, 1] = spec[:, 0]
    rspec[:, 2] = spec[:, 1]
    return rspec


def partition_banks(bank_numbers, processors_count):
    """
    Distributes bank numbers among MPI processes.

    This function takes a list of bank numbers and the total number of MPI processes,
    and returns a list of lists where each sublist contains the bank numbers assigned
    to a specific MPI process.

    Parameters
    ----------
    bank_numbers : numpy.ndarray
        An array of bank numbers to be distributed.
    processors_count : int
        The total number of MPI processes.

    Returns
    -------
    list
        A list of lists where each sublist contains the bank numbers assigned to a specific MPI process.
    """
    return [bank_numbers[_i::processors_count] for _i in range(processors_count)]


def bank_content_sizes(rmin, rmax):
    sizes = {}
    for bank in VULCAN_BANK_NUMBERS:
        pulse_index = np.array(db["entry"][f"bank{bank}_events"]["event_index"][rmin:rmax])
        pulse_id = np.array(db["entry"][f"bank{bank}_events"]["event_id"][int(pulse_index[0]) : int(pulse_index[-1])])
        sizes[bank] = pulse_id.size
    return sizes


time_o = time.time()
db = h5py.File(os.path.join(REPO_DIR, "tests", "data_large", "VULCAN_218738.nxs.h5"), "r")  # 25GB file size
det_difc = load_calibration(os.path.join(REPO_DIR, "tests", "data_large", "B123456DIFCs-12Cross-3456Cal.txt"))

tstart = 0
tend = 2205
invp = 1 / 60.0
ts_index = int((tstart / invp))
te_index = int((tend / invp))

comm = MPI.COMM_WORLD  # create a communicator
size = comm.Get_size()  # number of MPI processes requested
print(f"Number of MPI processes requested: {size}")
rank = comm.Get_rank()  # get the rank of the current process

# print(f"Bank content sizes = {bank_content_sizes(ts_index, te_index)}")
# print(f"Elapsed time = {time.time()-time_o}")

allspec = []
if rank == 0:
    banks = partition_banks(VULCAN_BANK_NUMBERS, size)
else:
    banks = None

jobs_per_mpi_process = comm.scatter(banks, root=0)
for job in jobs_per_mpi_process:
    print(f"Rank {rank} is processing bank {job}")
    spec = return_spectra(db, job, ts_index, te_index)
    allspec.append(spec)

allspec = MPI.COMM_WORLD.gather(allspec, root=0)

if rank == 0:
    print(f"Elapsed time: {time.time()-time_o}")

# if rank == 0:
#     allspec = [_i for temp in allspec for _i in temp]
#     specall = np.array(allspec)
#     for i in range(len(specall)):
#         bank_number = np.unique(specall[i, :, 0])[0]
#         plt.figure(bank_number)
#         plt.suptitle("Bank Number " + str(int(bank_number)))
#         plt.plot(specall[i, :, 1], specall[i, :, 2])
#     plt.show()
