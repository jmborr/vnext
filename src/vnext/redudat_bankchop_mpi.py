# standard imports
import argparse
import math
import os
import time
from collections import namedtuple
from os.path import dirname

# third-party imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

# vnext imports
#


REPO_DIR = os.path.dirname(dirname(dirname(os.path.abspath(__file__))))
VULCAN_BANK_NUMBERS = [1, 2, 3, 4, 5, 6]  # [4, 3, 1, 2, 5, 6]
PULSE_FREQUENCY = 60.0  # Hz
JobSpecs = namedtuple("JobSpecs", ["bank_number", "pulse_begin", "pulse_end"])


def get_difc(det_difc, event_idx):
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


def reduce_banks(event, event_tof, det_difc, tofmin=None, tofmax=None, tofbin=1250):
    event_tof /= get_difc(det_difc, event)
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


def return_spectra(db, det_difc, job_id):
    bank_events = db["entry"][f"bank{job_id.bank_number}_events"]
    event_index_list = bank_events["event_index"][job_id.pulse_begin : job_id.pulse_end]
    event_index_begin, event_index_end = int(event_index_list[0]), int(event_index_list[-1])
    pixel_id_list = bank_events["event_id"][event_index_begin:event_index_end]
    tof_list = bank_events["event_time_offset"][event_index_begin:event_index_end]
    # print(f"Size for bank {job_id.bank_number}: {pixel_id_list.size}")
    spec = reduce_banks(pixel_id_list, tof_list, det_difc)
    rspec = np.zeros([len(spec), 3])
    rspec[:, 0] = job_id.bank_number
    rspec[:, 1] = spec[:, 0]
    rspec[:, 2] = spec[:, 1]
    return rspec


def aportion_work(pulse_count, workers_count):
    """
    Distributes bank numbers among MPI processes.

    This function takes a list of bank numbers and the total number of MPI processes,
    and returns a list of lists where each sublist contains the bank numbers assigned
    to a specific MPI process.

    Parameters
    ----------
    pulse_count : int
        Number of pulses in the run.
    workers_count : int
        The total number of MPI worker processes.

    Returns
    -------
    list
        A list of lists where each sublist contains the bank numbers assigned to a specific MPI process.
    """

    # create a pool of jobs specs
    bank_count = len(VULCAN_BANK_NUMBERS)
    chunk_count = math.ceil(workers_count / bank_count)
    pulse_chunk = pulse_count // chunk_count
    jobs_pool = list()
    last_pulse_index = 0
    while last_pulse_index < pulse_count:
        first_pulse_index = 1 + last_pulse_index
        last_pulse_index = min(last_pulse_index + pulse_chunk, pulse_count)
        last_pulse_index = pulse_count if last_pulse_index + chunk_count >= pulse_count else last_pulse_index
        for bank in VULCAN_BANK_NUMBERS:
            jobs_pool.append(JobSpecs(bank, first_pulse_index, last_pulse_index))

    # distribute the jobs among the MPI processes
    jobs_per_mpi_process = list()
    for rank in range(workers_count):
        jobs = [job for job in jobs_pool[rank::workers_count]]
        jobs_per_mpi_process.append(jobs)
    return jobs_per_mpi_process


def bank_content_sizes(db, pulse_begin, pulse_end):
    sizes = {}
    for bank in VULCAN_BANK_NUMBERS:
        pulse_index = np.array(db["entry"][f"bank{bank}_events"]["event_index"][pulse_begin:pulse_end])
        pulse_id = np.array(db["entry"][f"bank{bank}_events"]["event_id"][int(pulse_index[0]) : int(pulse_index[-1])])
        sizes[bank] = pulse_id.size
    return sizes


def plot_spectra(spectra):
    spectra = [_i for temp in spectra for _i in temp]
    specall = np.array(spectra)
    for i in range(len(specall)):
        bank_number = np.unique(specall[i, :, 0])[0]
        plt.figure(bank_number)
        plt.suptitle("Bank Number " + str(int(bank_number)))
        plt.plot(specall[i, :, 1], specall[i, :, 2])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument("--file", "-f", required=True, help="Path to the data file")
    parser.add_argument("--calfile", "-c", required=True, help="Path to the calibration file")
    parser.add_argument("--plot", action="store_true", help="Plot the spectra if this flag is set")
    args = parser.parse_args()

    db = h5py.File(args.file, "r")
    pulse_count = int(db["entry"]["duration"][0] * PULSE_FREQUENCY)
    det_difc = load_calibration(args.calfile)

    time_o = time.time()
    comm = MPI.COMM_WORLD  # create a communicator
    mpi_workers_count = comm.Get_size()  # get the number of MPI processes
    rank = comm.Get_rank()  # get the rank of the current process
    if rank == 0:
        jobs = aportion_work(pulse_count, mpi_workers_count)
    else:
        jobs = None

    jobs_per_mpi_process = comm.scatter(jobs, root=0)
    allspec = []
    for job in jobs_per_mpi_process:
        print(f"Rank {rank} is processing job {job}")
        spec = return_spectra(db, det_difc, job)
        allspec.append(spec)
    allspec = MPI.COMM_WORLD.gather(allspec, root=0)

    if rank == 0:
        print(f"Elapsed time: {time.time()-time_o}")
        if args.plot:
            plot_spectra(allspec)


if __name__ == "__main__":
    main()
