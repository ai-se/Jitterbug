#! /bin/tcsh
# chmod +x run_hpc.sh
rm err/*
rm out/*
foreach NUM (`seq 0 1 9`)
  bsub -q standard -W 5000 -n 2 -o ./out/$NUM.out.%J -e ./err/$NUM.err.%J /share/tjmenzie/zyu9/miniconda2/bin/python2.7 runner.py exp_HPC $NUM
end
