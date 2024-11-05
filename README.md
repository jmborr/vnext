VNEXT
=====

Exploration ([EWM 7563](chttps://ornlrse.clm.ibmcloud.com/ccm/web/projects/Neutron%20Data%20Project%20%28Change%20Management%29#action=com.ibm.team.workitem.viewWorkItem&id=7563)) of fast reduction of VULCAN data


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

analysis nodes and ndav.sns.gov require argument `--map-by :OVERSUBSCRIBE:` if more than 24 workers are required.

Benchmarking the MPI script
---------------------------

The following runs were tried in analysis-node01.sns.gov and ndav.sns.gov using calibration file
/SNS/VULCAN/shared/Calibrationfiles/B123456DIFCs-12Cross-3456Cal.txt

- 25GB /SNS/VULCAN/IPTS-28883/nexus/VULCAN_218738.nxs.h5

| #Workers | Time (s) | Time (s) |
|----------|----------|----------|
|          |  node01  |  ndav    |
|  3       |  270     |  289     |
|  6       |  170     |  170     |
| 12       |   91     |   88     |
| 18       |   80     |   61     |
| 24       |   82     |   57     |

- 61GB /SNS/VULCAN/IPTS-28883/nexus/VULCAN_217914.nxs.h5

| #Workers | Time (s) | Time (s) |
|----------|----------|----------|
|          |  node01  |  ndav    |
|  6       |  600     |  537     |
| 12       |  310     |  290     |
| 18       |  229     |  199     |
| 24       |  187     |  152     |
| 30       |  162     |  126     |
| 36       |  168     |  123     |
| 42       |  137     |  111     |

- 113GB /SNS/VULCAN/IPTS-28883/nexus/VULCAN_219009.nxs.h5

| #Workers | Time (s) | Time (s) |
|----------|----------|----------|
|          |  node01  |  ndav    |
|  6       |  860     |  623     |
| 12       |  448     |  406     |
| 18       |  364     |  336     |
| 24       |  319     |  262     |
| 30       |  253     |  246     |
| 36       |  254     |  229     |
| 42       |  262     |  237     |
