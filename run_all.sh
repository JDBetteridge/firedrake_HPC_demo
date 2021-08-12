#!/bin/bash

## Single core
python singlecore.py

## Weak scaling
for NCPU in 1 2 4 8 16 32 64 128
    do
    srun -n $NCPU $VIRTUAL_ENV/bin/python weakscaling.py
done

## Strong Scaling
CPU_PER_NODE=128
for NODES in 8 #4 2 1 #8
    do
    NCPU=$(python -c "print($CPU_PER_NODE*$NODES)")
    srun -n $NCPU $VIRTUAL_ENV/bin/python strongscaling.py --solver "Full MG" --telescope $NODES
    srun -n $NCPU $VIRTUAL_ENV/bin/python strongscaling.py --solver "Matfree FMG" --telescope $NODES
    srun -n $NCPU $VIRTUAL_ENV/bin/python strongscaling.py --solver "Telescoped matfree FMG" --telescope $NODES
done
