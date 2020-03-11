#!/usr/bin/env bash
python main.py --name analysis --agent evids -n 5 -k 3 --horizon 1000 --nruns 100 --prior uniform --ids_action_selection greedy
python main.py --name analysis --agent eids -n 5 -k 3 --horizon 1000 --nruns 100 --prior uniform --ids_action_selection exact
python main.py --name analysis --agent eids -n 5 -k 3 --horizon 1000 --nruns 100 --prior uniform --ids_action_selection approximate
python main.py --name analysis --agent eids -n 5 -k 3 --horizon 1000 --nruns 100 --prior uniform --ids_action_selection greedy
