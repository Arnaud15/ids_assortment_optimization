#!/usr/bin/env bash
python main.py --agent ets -n 5 -k 2 --horizon 500 --nruns 50
python main.py --agent etscs -n 5 -k 2 --horizon 500 --nruns 50
python main.py --agent eids -n 5 -k 2 --horizon 500 --nruns 50
python main.py --agent eidscs -n 5 -k 2 --horizon 500 --nruns 50
