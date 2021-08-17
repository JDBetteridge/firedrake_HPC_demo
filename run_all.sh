#!/bin/bash

## Single core
$VIRTUAL_ENV/bin/python singlecore.py

## Weak scaling
# Warmup on 2 cores
srun -n 2 $VIRTUAL_ENV/bin/python weakscaling.py
for NCPU in 1 2 4 8 16 32 64 128
    do
    srun -n $NCPU $VIRTUAL_ENV/bin/python weakscaling.py
done

## Strong Scaling
CPU_PER_NODE=128
# Warmup on 8 nodes
for NODES in 8 8 4 2 1
    do
    NCPU=$(python -c "print($CPU_PER_NODE*$NODES)")
    srun -n $NCPU $VIRTUAL_ENV/bin/python strongscaling.py --solver "CG + AMG"
    srun -n $NCPU $VIRTUAL_ENV/bin/python strongscaling.py --solver "CG + full GMG"
    srun -n $NCPU $VIRTUAL_ENV/bin/python strongscaling.py --solver "Matfree CG + telescoped full GMG" --telescope $NODES
done
