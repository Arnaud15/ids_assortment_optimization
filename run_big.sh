#!/usr/bin/env bash
#python main.py --name big --agent ets -n 1000 -k 10 --horizon 5000 --nruns 2 --prior uniform 
#python main.py --name big --agent etscs -n 1000 -k 10 --horizon 5000 --nruns 2 --prior uniform 
python main.py --name big --agent evids -n 1000 -k 10 --horizon 5000 --nruns 2 --prior uniform --ids_action_selection greedy
