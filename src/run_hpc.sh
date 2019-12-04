#! /bin/tcsh
# chmod +x run_hpc.sh
rm err/*
rm out/*
foreach NUM (`seq 0 1 29`)
  bsub -q standard -W 5000 -n 2 -o ./out/$NUM.out.%J -e ./err/$NUM.err.%J /share/tjmenzie/zyu9/miniconda2/bin/python2.7 error.py exp_transfer $NUM
end
