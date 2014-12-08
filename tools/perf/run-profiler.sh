# File:   run-profiler.sh
# Date:   Mon 08 Dec 2014 09:09:11 AM CET
# Author: Fabian Wermelinger
# Tag:    Use nvprof tool
# Copyright © 2014 Fabian Wermelinger. All Rights Reserved.
export OMP_NUM_THREADS=8

export PATH=/usr/local/cuda/bin/:$PATH

nvprof --profile-child-processes --print-gpu-trace \
    --metrics flops_sp,flops_sp_add,flops_sp_fma,flops_sp_mul,flops_sp_special,dram_read_throughput,dram_write_throughput,dram_utilization,dram_read_transactions,dram_write_transactions,ipc \
    ./mpcf-cluster-gpu \
    -sim SteadyStateMPI \
    -tend 1.0 -cfl 0.32 -nslices 256 \
    -dumpinterval 10.0 -saveperiod 8000 \
    -subcellsX 16 -IO false -SOSkernel qpx -UPkernel qpx \
    -nsteps 1 &>> profiler.metrics

nvprof --profile-child-processes --print-summary \
    ./mpcf-cluster-gpu \
    -sim SteadyStateMPI \
    -tend 1.0 -cfl 0.32 -nslices 256 \
    -dumpinterval 10.0 -saveperiod 8000 \
    -subcellsX 16 -IO false -SOSkernel qpx -UPkernel qpx \
    -nsteps 1 &>> profiler.summary
