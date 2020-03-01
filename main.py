from env import AssortmentEnvironment
from base_agents import RandomAgent
from ts_agents import EpochSamplingTS, HypermodelTS
from ids_agents import EpochSamplingIDS, HypermodelIDS
from utils import save_experiment_data, print_actions_posteriors
from scipy.stats import uniform
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--agent", type=str, required=True, help="choice of ts, ids, rd, ets, eids")
parser.add_argument("-n", type=int, default=5, help="number of items available")
parser.add_argument("-k", type=int, default=3, help="size of the assortments")
parser.add_argument("--horizon", type=int, default=499, help="number of random simulations to carry out with agent")
parser.add_argument("--nruns", type=int, default=50, help="number of random simulations to carry out with agent")
parser.add_argument("--fixed_preferences", type=int, default=0,
                    help="if you want episodes running with pre-defined preferences")
parser.add_argument("--verbose", type=int, default=0, help="print intermediate info during episodes or not")
parser.add_argument("--ids_action", type=str, default='greedy', help="action selection: exact IDS, approximate (linear in O(A)), greedy")
parser.add_argument("--greedy_scaler", type=float, default=1.0, help="scaling factor for greedy action selection")
parser.add_argument("--ids_samples", type=int, default=100,
                    help="if you want episodes running with pre-defined preferences")
parser.add_argument("--reg_weight", type=float, default=0.1)
parser.add_argument("--training_sigmaobs", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=1e-2)# 1e-3 and reg 0.1 for mlp
parser.add_argument("--model_input_dim", type=int, default=10)
parser.add_argument("--nsteps", type=int, default=25)
parser.add_argument("--printinterval", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--nzsamples", type=int, default=32)

AGENTS = {"rd": RandomAgent,
        #   "ts": ThompsonSamplingAgent,
        #   "ids": InformationDirectedSamplingAgent,
          "ets": EpochSamplingTS,
          "eids": EpochSamplingIDS,
          "evids":EpochSamplingIDS,
          "hts": HypermodelTS,
          "hids": HypermodelIDS}

FIXED_PREFERENCES = np.concatenate([np.array([0.1, 0.2, 0.5, 0.5, 0.3]),
                                    np.array([1.])])


def run_episode(envnmt, actor, n_steps, verbose=False):
    """
    :param envnmt: instance from the AssortmentEnvironment class
    :param actor: instance from the Agent class
    :param n_steps: horizon of the run in the environment
    :return: (observations history = list of (assortment one-hot array of size N+1, 0<=index<=N of item picked),
    rewards = historical rewards)
    """
    actor.reset()  # Resets the internal state of the agent at the start of simulations (prior beliefs, etc...)
    rewards = np.zeros(n_steps)
    obs = [0] * n_steps
    iterator = tqdm(range(n_steps)) if not verbose else range(n_steps)
    for ix in iterator:
        assortment = actor.act()
        item_selected = envnmt.step(assortment)
        obs[ix] = (assortment, item_selected)
        reward = actor.update(item_selected)  # agent observes item selected, perceive reward and updates its beliefs
        if verbose and ((ix > n_steps - 2) or ((ix + 1) % 25 == 0) or (not ix)):
            print_actions_posteriors(agent=actor, past_observations=obs[:ix+1])
        rewards[ix] = reward
    prefs_str = [f'{run_preference:.2f}' for run_preference in envnmt.preferences]
    if verbose:
        print(f'Initial preferences were :{prefs_str}')
        print(f'Best action was: {run_preferences.argsort()[-(args.k + 1):][::-1][1:]}')
    return obs, rewards


def summarize_run(observations):
    run_assortments = sum([assortment for assortment, item_picked in observations])
    run_picks = {item_ix: 0 for item_ix in range(args.n + 1)}
    for assortment, item_picked in observations:
        run_picks[item_picked] += 1
    return {"assortments": run_assortments,
            "picks": run_picks}


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"arguments are {args}")
    print(f"Fixed preferences are: {FIXED_PREFERENCES}.")
    print(f"Preferences used are {'fixed' if args.fixed_preferences else 'random'}")

    # Experiment identifier in storage
    correlated_sampling = args.agent[-2:] == "cs"
    info_type = "VIDS" if 'vids' in args.agent else "IDS"
    agent_key = args.agent[:-2] if correlated_sampling else args.agent

    agent_name = f"{args.agent}_{args.ids_samples}_{args.ids_action}" if 'ids' in args.agent else args.agent
    exp_keys = [agent_name,
                args.n,
                args.k,
                args.horizon]
    exp_id = '_'.join([str(elt) for elt in exp_keys])
    print(f"Name of the agent is {exp_id}")

    # Agent init
    agent_class = AGENTS[agent_key]
    agent = agent_class(k=args.k,
                        n=args.n,
                        correlated_sampling=correlated_sampling,
                        horizon=args.horizon,
                        n_samples=args.ids_samples,
                        info_type=info_type,
                        action_type=args.ids_action,
                        scaling_factor=args.greedy_scaler,
                        params=args)
    
    # Hyperparameters search
    # step 1 is below, step 2 is too look into the best sigma_obs 
    # best params = 0.1 / 10 / 1. for linear model
    # best params = ??? for neural model.
    # IDEA: encode how many times items have occured together in the model?
    # lrs = [1e-1, 1e-2, 1e-3, 1e-4]
    # model_dims = [5, 10, 15, 25]
    # regularizations = [0.1, 0.5, 1., 2.]
    # best_gap = np.inf 
    # best_parameters = {}
    # for given_lr in lrs:
    #     for given_dim in model_dims:
    #         for given_reg in regularizations:
    #             args.lr = given_lr
    #             args.model_input_dim = given_dim
    #             args.reg_weight = given_reg
    #             agent = agent_class(k=args.k,
    #                 n=args.n,
    #                 correlated_sampling=correlated_sampling,
    #                 horizon=args.horizon,
    #                 n_samples=args.ids_samples,
    #                 info_type=info_type,
    #                 params=args)
    #             gap_params = 0.
    #             for _ in range(args.nruns):
    #                 run_preferences = np.concatenate([uniform.rvs(size=args.n),
    #                                                 np.array([1.])])
    #                 env = AssortmentEnvironment(n=args.n, v=run_preferences)
    #                 obs_run, rewards_run = run_episode(envnmt=env, actor=agent, n_steps=args.horizon)
    #                 gap_params += np.mean((run_preferences[:-1] - agent.sample_from_posterior(1000).mean(0)) ** 2)
    #             gap_params = gap_params / args.nruns
    #             if gap_params < best_gap:
    #                 best_gap = gap_params
    #                 best_parameters = {'lr':given_lr, 'dim':given_dim, 'regularization':given_reg}
    #                 print(f"new best parameters: {best_parameters}, with gap: {np.sqrt(best_gap)}")
                    
    # import pickle 
    # with open('./outputs/hyper_params.pickle', 'wb') as handle:
    #     pickle.dump(best_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('best parameters saved')
    # with open('./outputs/hyper_params.pickle', 'rb') as handle:
    #     params = pickle.load(handle)
    #     print(f'best parameters loaded: {params}')
                    

    # Actual experiments with logging
    experiment_data = []
    for _ in range(args.nruns):
        run_preferences = FIXED_PREFERENCES
        if not args.fixed_preferences:
            selected_item = np.random.randint(args.n)
            run_preferences = np.zeros(args.n+1)
            run_preferences[selected_item] = 1.
            run_preferences[args.n] = 1.
            run_preferences = np.concatenate([uniform.rvs(size=args.n),
                                              np.array([1.])])
        env = AssortmentEnvironment(n=args.n, v=run_preferences)
        top_preferences = np.sort(run_preferences)[-(args.k + 1):]
        top_preferences = top_preferences / top_preferences.sum()
        expected_reward_from_best_action = top_preferences[:args.k].sum()

        run_data = {"best_reward": expected_reward_from_best_action}
        obs_run, rewards_run = run_episode(envnmt=env, actor=agent, n_steps=args.horizon, verbose=args.verbose)

        run_data["rewards"] = rewards_run
        run_data.update(summarize_run(obs_run))

        experiment_data.append(run_data)

    save_experiment_data(exp_id, experiment_data)
    print(f"Experiment successfully terminated")

else:
    import sys
    sys.exit("main should be used as a script")

# TODO: hp search for hts
# TODO: hts vs hids experiments
# TODO: 5 choose 3 setting experiments

# TODO: greedy algorithm
# TODO: variational inference

# TODO: check mcmc code