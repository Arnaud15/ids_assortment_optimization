#!/usr/bin/env bash
python main.py --agent ets -n 5 -k 2 --horizon 500 --nruns 200
python main.py --agent etscs -n 5 -k 2 --horizon 500 --nruns 200
python main.py --agent eids -n 5 -k 2 --horizon 500 --nruns 200 --ids_samples 10
python main.py --agent eids -n 5 -k 2 --horizon 500 --nruns 200 --ids_samples 100
python main.py --agent eids -n 5 -k 2 --horizon 500 --nruns 200 --ids_samples 1000
