VNEXT
=====

Exploration ((EWM 7563](chttps://ornlrse.clm.ibmcloud.com/ccm/web/projects/Neutron%20Data%20Project%20%28Change%20Management%29#action=com.ibm.team.workitem.viewWorkItem&id=7563)) of fast reduction of VULCAN data


Development environment setup
-----------------------------

At the root of the repo, create the conda environemnt `vnext-dev`

```
conda env create --file environment.yml --solver=libmamba
conda activate vnext-dev
```

Running the MPI script
----------------------

For instance, to run the script with 42 MPI workers and plot the histrograms for each bank:

```
cd ./src/vnext
mpiexec --map-by :OVERSUBSCRIBE: -n 42 python -m redudat_bankchop_mpi -f /path/to/VULCAN_XXXX.nxs.h5 -c /path/to/calibration_file.txt --plot
```

analysis nodes require argument `--map-by :OVERSUBSCRIBE:` if more than 24 workers are required.

Benchmarking the MPI script
---------------------------

The following runs were tried in analysis-node01.sns.gov using calibration file
/SNS/VULCAN/shared/Calibrationfiles/B123456DIFCs-12Cross-3456Cal.txt

- 25GB /SNS/VULCAN/IPTS-28883/nexus/VULCAN_218738.nxs.h5

| #Workers | Time (s) |
|----------|----------|
|  3       |  270     |
|  6       |  170     | 
| 12       |   91     |
| 18       |   80     |
| 24       |   82     |
| 30       |          |

- 61GB /SNS/VULCAN/IPTS-28883/nexus/VULCAN_217914.nxs.h5

| #Workers | Time (s) |
|----------|----------|
|  6       |  600     | 
| 12       |  310     |
| 18       |  229     |
| 24       |  187     |
| 30       |  162     |
| 36       |  168     |
| 42       |  137     |

- 113GB /SNS/VULCAN/IPTS-28883/nexus/VULCAN_219009.nxs.h5

| #Workers | Time (s) |
|----------|----------|
|  6       |  860     | 
| 12       |  448     |
| 18       |  364     |
| 24       |  319     |
| 30       |  253     |
| 36       |  254     |
| 42       |  262     |
