#!/bin/bash

start=$(date +%s.%N)

mpirun -np 8 python ../pyradexnest_multimol.py

end=$(date +%s.%N)    
runtime=$(python -c "print(${end} - ${start})")

echo "Runtime was $runtime"
