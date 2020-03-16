#!/usr/bin/env bash
#python main.py --name big --agent etscs -n 1000 -k 10 --horizon 5000 --nruns 100 --prior uniform 
#python main.py --name big --agent ets -n 1000 -k 10 --horizon 5000 --nruns 20 --prior uniform 
python main.py --name big --agent evids -n 1000 -k 10 --horizon 5000 --nruns 50 --prior uniform --ids_action_selection greedy --greedy_scaler 0.00789
