#!/usr/bin/env bash
#python main.py --name gap --agent ets -n 10 -k 3 --horizon 1000 --nruns 100 --prior restricted
#python main.py --name gap --agent etscs -n 10 -k 3 --horizon 1000 --nruns 100 --prior restricted
#python main.py --name gap --agent evids -n 10 -k 3 --horizon 1000 --nruns 100 --prior restricted --ids_action_selection greedy --best_scaler_h 250 --best_scaler_n 20
#python main.py --name gap --agent ets -n 100 -k 3 --horizon 2500 --nruns 100 --prior restricted
#python main.py --name gap --agent etscs -n 100 -k 3 --horizon 2500 --nruns 100 --prior restricted
#python main.py --name gap --agent evids -n 100 -k 3 --horizon 2500 --nruns 100 --prior restricted --ids_action_selection greedy --greedy_scaler 0.015 --find_best_scaler 0
#python main.py --name gap --agent etscs -n 1000 -k 3 --horizon 5000 --nruns 100 --prior restricted
python main.py --name gap --agent ets -n 1000 -k 3 --horizon 5000 --nruns 100 --prior restricted
python main.py --name gap --agent evids -n 1000 -k 3 --horizon 5000 --nruns 100 --prior restricted --ids_action_selection greedy
