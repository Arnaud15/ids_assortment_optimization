from env import AssortmentEnvironment
from base_agents import RandomAgent
from ts_agents import EpochSamplingTS, HypermodelTS
from ids_agents import EpochSamplingIDS, HypermodelIDS
from utils import save_experiment_data, print_actions_posteriors, get_prior
from scipy.stats import uniform
import numpy as np
import pickle 
from tqdm import tqdm
from args import get_experiment_args


# LIST OF SUPPORTED AGENT KEYS WITH THE CORRESPONDING CLASSES
AGENTS = {"rd": RandomAgent,
          "ets": EpochSamplingTS,
          "eids": EpochSamplingIDS,
          "evids":EpochSamplingIDS,
          "hts": HypermodelTS,
          "hids": HypermodelIDS}


def run_episode(envnmt, actor, n_steps, verbose=False):
    """
    :param envnmt: instance from the AssortmentEnvironment class
    :param actor: instance from the Agent class
    :param n_steps: horizon of the run in the environment
    :return: (observations history = list of (assortment one-hot array of size N+1, 0<=index<=N of item picked),
    rewards = 1D numpy array of size (horizon,) with entries = expected_reward from action taken given env parameters
    """
    # Initialization of observations and agent
    actor.reset()  
    rewards = np.zeros(n_steps)
    obs = [0] * n_steps
    iterator = tqdm(range(n_steps)) if not verbose else range(n_steps)
    for ix in iterator:
        # act / step / update
        assortment = actor.act()
        item_selected = envnmt.step(assortment)
        actor.update(item_selected) 
        # Store expected reward, observation
        obs[ix] = (assortment, item_selected)
        unnormalized_pick_proba = envnmt.preferences[assortment].sum()
        rewards[ix] = unnormalized_pick_proba / (1. + unnormalized_pick_proba)
        # print(rewards[ix])
        # Print current posterior belief of the agent if asked
        if verbose and ((ix > n_steps - 2) or ((ix + 1) % 50 == 0) or (ix == 0)):
            print_actions_posteriors(agent=actor, past_observations=obs[:ix+1])
    prefs_str = [f'{run_preference:.2f}' for run_preference in envnmt.preferences]

    # Print environment model parameters if asked
    if verbose:
        print(f'Initial preferences were :{prefs_str}')
        print(f'Best action was: {env.preferences.argsort()[-(args.k + 1):][::-1][1:]}')
    return obs, rewards


def summarize_run(observations):
    """
    param: observations = [obs=(k-sparse assortment given, index of item selected) for obs in observations]
    return: {assortments: 1D array of size K with how many times each item is proposed,
             picks: 1D array with how many times each item if picked}
    """
    run_assortments = sum([assortment for assortment, item_picked in observations])
    run_picks = {item_ix: 0 for item_ix in range(args.n + 1)}
    for assortment, item_picked in observations:
        run_picks[item_picked] += 1
    return {"assortments": run_assortments,
            "picks": run_picks}

# TODO refactor that when we come back to work with hypermodels
def hypermodels_search(args, step1, step2):
    # Hyperparameters search
    # step 1 is below, step 2 is too look into the best sigma_obs 
    # best params = ??? for neural model.
    # IDEA: encode how many times items have occured together in the model?
    print("Hyperparameters search begins...")
    if step1:
        lrs = [1e-1, 1e-2, 1e-3, 1e-4]
        model_dims = [5, 10, 15, 25]
        regularizations = [0.1, 0.5, 1., 2.]
        best_gap = np.inf 
        best_parameters = {}
        for given_lr in lrs:
            for given_dim in model_dims:
                for given_reg in regularizations:
                    args.lr = given_lr
                    args.model_input_dim = given_dim
                    args.reg_weight = given_reg
                    agent = AGENTS['hts'](k=args.k,
                                            n=args.n,
                                            correlated_sampling=False,
                                            horizon=args.horizon,
                                            n_samples=args.ids_samples,
                                            params=args)
                    gap_params = 0.
                    for _ in range(args.nruns):
                        run_preferences = np.concatenate([uniform.rvs(size=args.n),
                                                        np.array([1.])])
                        env = AssortmentEnvironment(n=args.n, v=run_preferences)
                        obs_run, rewards_run = run_episode(envnmt=env, actor=agent, n_steps=args.horizon)
                        gap_params += np.mean((run_preferences[:-1] - agent.sample_from_posterior(1000).mean(0)) ** 2)
                    gap_params = gap_params / args.nruns
                    if gap_params < best_gap:
                        best_gap = gap_params
                        best_parameters = {'lr':given_lr, 'dim':given_dim, 'regularization':given_reg}
                        print(f"new best parameters: {best_parameters}, with gap: {np.sqrt(best_gap)}")
                    
        with open('./outputs/hyper_params.pickle', 'wb') as handle:
            pickle.dump(best_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Best params step 1 saved')
        print("Step 1 done.")
    if step2:
        params_loaded = None
        with open('./outputs/hyper_params.pickle', 'rb') as handle:
            params_loaded = pickle.load(handle)
            print(f'Best parameters step 1 loaded: {params_loaded}')
            args.lr = params_loaded['lr']
            args.model_input_dim = params_loaded['dim']
            args.reg_weight = params_loaded['regularization']
        sigma_obss = [0.01, 0.1, 0.3, 0.5]
        agent = AGENTS[args.agent](k=args.k,
            n=args.n,
            correlated_sampling=False,
            horizon=args.horizon,
            params=args)
        best_cumulated_rewards = - np.inf
        for sigma_obs in sigma_obss:
            param_rewards = 0.
            for _ in range(args.nruns):
                run_preferences = np.concatenate([uniform.rvs(size=args.n),
                                                np.array([1.])])
                env = AssortmentEnvironment(n=args.n, v=run_preferences)
                obs_run, rewards_run = run_episode(envnmt=env, actor=agent, n_steps=args.horizon)
                param_rewards += np.sum(rewards_run) 
            param_rewards /= args.nruns
            if param_rewards > best_cumulated_rewards:
                best_cumulated_rewards = param_rewards
                params_loaded['sigma_obs'] = sigma_obs
                print(f'new best rewards with sigma obs = {sigma_obs:.3f}, rewards: {param_rewards:.2f}')
        with open('./outputs/hyper_params2.pickle', 'wb') as handle:
            pickle.dump(params_loaded, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Best parameters step 2 saved') 
        print("Step 2 done")
    print('HP search done.')
    return


def scaler_search(args, agent_class, info_type):
    """
    args = Namespace object with the parameters for the experiment
    agent_class = class for the IDS agent for which we find the best epsilon parameter for greedy IDS action selection
    """
    print("Scale search begins:")
    # Grid search over the following possible values
    scales = np.linspace(start=0., stop=0.006, num=10) 
    best_rewards = - np.inf
    for scale in scales:
        # Instantiate the agent
        agent = agent_class(k=args.k,
                            n=args.n,
                            correlated_sampling=False,
                            horizon=args.horizon,
                            n_samples=args.ids_samples,
                            limited_prefs=True if args.prior == "restricted" else False,
                            info_type=info_type,
                            action_type=args.ids_action_selection,
                            scaling_factor=scale,
                            params=args)
        # Examine the sum of rewards accumulated on average over args.best_scaler_n runs
        sum_of_rewards = 0.
        for _ in range(args.best_scaler_n):
            run_preferences = get_prior(n_items=args.n, prior_type=args.prior) 
            env = AssortmentEnvironment(n=args.n, v=run_preferences)
            top_preferences = np.sort(run_preferences)[-(args.k + 1):]
            top_preferences = top_preferences / top_preferences.sum()
            expected_reward_from_best_action = top_preferences[:args.k].sum()

            obs_run, rewards_run = run_episode(envnmt=env, actor=agent, n_steps=args.best_scaler_h)
            sum_of_rewards += (rewards_run - expected_reward_from_best_action).sum()
        sum_of_rewards = sum_of_rewards / args.best_scaler_n
        if sum_of_rewards > best_rewards:
            best_rewards = sum_of_rewards
            best_scale = scale
            print(f"New best scaling factor:{scale} with average cumulative regret: {sum_of_rewards:.2f} over horizon: {args.best_scaler_h}") 
    return best_scale
            

if __name__ == "__main__":
    args = get_experiment_args(run_or_plot='run')
    print(f"Arguments are {args}")
    print(f"Prior on environment parameters is: {args.prior}.")

    # Agent name parsing to pick the right agent class and parameters 
    correlated_sampling = args.agent[-2:] == "cs"
    info_type = "VIDS" if 'vids' in args.agent else "IDS"
    agent_key = args.agent[:-2] if correlated_sampling else args.agent

    # Agent name (string used in plots to identify it compared to other agents)
    agent_name = f"{args.agent}_{args.ids_samples}_{args.ids_action_selection}" if 'ids' in args.agent else args.agent
    exp_keys = [agent_name,
                args.n,
                args.k,
                args.horizon,
                args.name]
    exp_id = '_'.join([str(elt) for elt in exp_keys])
    print(f"Name of the agent is {exp_id}")

    # Picking the correct agent class
    agent_class = AGENTS[agent_key]

    # Looking for the best approximation to true IDS action selection here
    if ('ids' in agent_key) and (args.ids_action_selection == 'greedy') and args.find_best_scaler:
        args.greedy_scaler = scaler_search(args, agent_class, info_type)

    # Instantiating the agent before experiments
    agent = agent_class(k=args.k,
                        n=args.n,
                        correlated_sampling=correlated_sampling,
                        horizon=args.horizon,
                        limited_prefs=True if args.prior == "restricted" else False,
                        n_samples=args.ids_samples,
                        info_type=info_type,
                        action_type=args.ids_action_selection,
                        scaling_factor=args.greedy_scaler,
                        params=args)

    # Actual experiments with printing
    experiment_data = []
    for _ in range(args.nruns):
        run_preferences = get_prior(n_items=args.n, prior_type=args.prior) 
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