from mcmc import sample_from_posterior
from scipy.stats import uniform
from utils import act_optimally, possible_actions, expected_reward, optimized_ratio
from base_agents import Agent, EpochSamplingAgent
from collections import defaultdict
import numpy as np
from functools import partial

N_SAMPLES_IDS = 16
DISCRETE_IDS_OPTIMIZATION = True


def delta_full(action, sampled_preferences, r_star):
    return r_star - expected_reward(action=action, preferences=sampled_preferences)


def g_full(action, sampled_preferences, opt_actions):
    """
    :param action:
    :param sampled_preferences: sampled posterior thetas
    :param opt_actions: dictionary {action_tuple:p_action, theta_indices}
    :return:
    """
    g_a = 0.
    probs = 0.
    probas_given_action = sampled_preferences[:, action]
    probas_given_action = probas_given_action / (1 + np.expand_dims(probas_given_action.sum(1), axis=-1))
    no_pick_given_action = 1 - probas_given_action.sum(1)
    p_no_item_action = no_pick_given_action.mean()
    probs += p_no_item_action
    for action_star, (p_star, theta_indices) in opt_actions.items():
        p_no_item_a_star_action = np.mean([no_pick_given_action[theta_indice] for theta_indice in theta_indices])
        g_a += p_no_item_a_star_action * np.log(p_no_item_a_star_action / (p_star * p_no_item_action))

    for action_ix, item_ix in enumerate(action):
        p_item_action = probas_given_action[:, action_ix].mean()
        if p_item_action:
            probs += p_item_action
            for action_star, (p_star, theta_indices) in opt_actions.items():
                p_item_a_star_action = np.mean(
                    [probas_given_action[theta_indice, action_ix] for theta_indice in theta_indices])
                if p_item_a_star_action:
                    g_a += p_item_a_star_action * np.log(p_item_a_star_action / (p_star * p_item_action))
    assert probs > 0.99, f"{probs}"
    return g_a


def v_full(action, sampled_preferences, opt_actions):
    if len(opt_actions.keys()) > 1:
        r_a_t_given_a_star = np.array(
            [expected_reward(sampled_preferences[thetas_a_star, :], action) for a_star, (p_a_star, thetas_a_star) in
             opt_actions.items()])
        probas_a_star = np.array([p_a_star for a_star, (p_a_star, thetas_a_star) in opt_actions.items()])
        return probas_a_star.dot(r_a_t_given_a_star ** 2) - (probas_a_star.dot(r_a_t_given_a_star)) ** 2
    else:
        return 0.


# TODO fix with greedy
def approximate_ids_action_selection(n, k, delta_, v_):
    v_information_ratios_items = - np.array([delta_([i]) ** 2 / v_([i]) for i in range(n)])
    if np.isinf(v_information_ratios_items.min()):
        v_information_ratios_items = - np.array([delta_([i]) for i in range(n)])
        return np.sort(np.argpartition(v_information_ratios_items, -k)[-k:])
    else:
        return np.sort(np.argpartition(v_information_ratios_items, -k)[-k:])


def ids_action_selection(n, k, delta_, g_):
    actions_set = possible_actions(n_items=n, assortment_size=k)
    min_information_ratio = np.inf
    deltas = [None, None]
    gains = [None, None]
    ids_action = actions_set[0]
    for action1 in actions_set:
        for action2 in actions_set:
            g_a1, g_a2 = g_(action1), g_(action2)
            if np.isnan(g_a1):
                import ipdb;
                ipdb.set_trace()
                g_a1 = g_(action1)
            delta_1, delta_2 = delta_(action1), delta_(action2)
            if (not g_a1) or (not g_a2):
                if delta_1 < delta_2:
                    value = delta_1
                    action_picked = action1
                else:
                    value = delta_2
                    action_picked = action2
            else:
                value, rho = optimized_ratio(d1=delta_1,
                                             d2=delta_2,
                                             g1=g_a1,
                                             g2=g_a2,
                                             discrete=DISCRETE_IDS_OPTIMIZATION)

                action_picked = action1 if np.random.rand() <= rho else action2
            if value < min_information_ratio:
                deltas = delta_1, delta_2
                gains = g_a1, g_a2
                min_information_ratio = value
                ids_action = action_picked

    # print(f"min information ratio obtained is {min_information_ratio:.4f}")
    # print(f"with deltas: {[f'{delt:.2f}' for delt in deltas]}")
    # print(f"and information gains: {[f'{gain:.2f}' for gain in gains]}")

    return ids_action


# TODO why are the improvements observed from CS not clear with TS? Experiments too small? Confirm that
# Also try out their exact formula for STD in CS on a large problem instance and check that we get similar results
# If both of the above are positive, implement approximate IDS for large problem instance and compare

# TODO confirm results with IDS on the small scenario
# TODO observe bound and think it through

# TODO MCMC algorithms on the small scenario with horizon 300/500
# TODO find a faster package for multinomial logistic regression / alternative to pymc3
# TODO explore approx Bayes method for our setting
# TODO check out ICLR paper for approximating posterior distributions

# TODO new experiments section in the Overleaf
class InformationDirectedSamplingAgent(Agent):
    def __init__(self, k, n, **kwargs):
        """
        :param k: assortment size
        :param n: number of items available
        :param n_ids_samples: number of posterior samples for IDS
        """
        super().__init__(k, n)
        self.prior_belief = uniform.rvs(size=(N_SAMPLES_IDS, n))
        self.n_samples = N_SAMPLES_IDS
        self.assortments_given = []
        self.item_picks = []
        self.optimal_actions = None
        self.g_ = None
        self.compute_g()
        self.r_star = 0.
        self.delta_ = None
        self.compute_delta()

    def update_r_star(self):
        sorted_beliefs = np.sort(self.prior_belief, axis=1)[:, -self.n_items:]  # shape (m, k)
        picking_probabilities = sorted_beliefs.sum(1)
        self.r_star = (picking_probabilities / (1 + picking_probabilities)).mean()

    def update_optimal_actions(self):
        """
        :return: dictionary of informations about optimal action for each posterior sample of the model parameters
        # keys: actions = sorted tuple of items to propose in the assortment
        # values: (p(action = a*), [thetas such that action is optimal for theta]
        """

        posteriors_actions = act_optimally(self.prior_belief, self.assortment_size)
        posteriors_actions = [tuple(posteriors_actions[ix, :]) for ix in range(self.n_samples)]
        optimal_actions_information = defaultdict(list)
        for ix, action in enumerate(posteriors_actions):
            optimal_actions_information[action].append(ix)

        self.optimal_actions = {action: (len(theta_idxs) / self.n_samples, theta_idxs) for
                                action, theta_idxs in optimal_actions_information.items()}

    def compute_delta(self):
        """
        :return:
        """
        self.update_r_star()
        self.delta_ = partial(delta_full,
                              sampled_preferences=self.prior_belief,
                              r_star=self.r_star)

    def compute_g(self):
        self.update_optimal_actions()
        self.g_ = partial(g_full,
                          sampled_preferences=self.prior_belief,
                          opt_actions=self.optimal_actions)

    def act(self):
        """
        3 steps:
        - loop over M posterior samples and:
            get p(a*)
        - loop over items + no_items and:
            compu
        :return:
        """
        action = ids_action_selection(n=self.n_items,
                                      k=self.assortment_size,
                                      delta_=self.delta_,
                                      g_=self.g_)
        assortment = np.zeros(self.n_items + 1)
        assortment[self.n_items] = 1.
        for item in action:
            assortment[item] = 1.
        self.assortments_given.append(assortment)
        return np.array(action)

    def reset(self):
        self.prior_belief = uniform.rvs(size=(self.n_samples, self.n_items))
        self.assortments_given = []
        self.item_picks = []
        self.optimal_actions = None
        self.g_ = None
        self.compute_g()
        self.r_star = 0.
        self.delta_ = None
        self.compute_delta()

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        self.item_picks.append(item_selected)
        self.prior_belief = sample_from_posterior(n_samples=self.n_samples,
                                                  assortments=np.array(self.assortments_given),
                                                  item_picks=np.array(self.item_picks),
                                                  n_observations=len(self.item_picks),
                                                  n_items=self.n_items)
        self.compute_delta()
        self.compute_g()
        return reward


# TODO multiple inheritance here?
class EpochSamplingIDS(EpochSamplingAgent):
    def __init__(self, k, n, horizon, correlated_sampling):
        super().__init__(k, n, horizon=horizon, correlated_sampling=correlated_sampling)
        self.optimal_actions = None
        self.g_ = None
        self.r_star = 0.
        self.delta_ = None
        self.n_samples = N_SAMPLES_IDS
        self.prior_belief = self.sample_from_posterior(self.n_samples)

    def update_r_star(self):
        sorted_beliefs = np.sort(self.prior_belief, axis=1)[:, -self.n_items:]  # shape (m, k)
        picking_probabilities = sorted_beliefs.sum(1)
        self.r_star = (picking_probabilities / (1 + picking_probabilities)).mean()

    def update_optimal_actions(self):
        """
        :return: dictionary of informations about optimal action for each posterior sample of the model parameters
        # keys: actions = sorted tuple of items to propose in the assortment
        # values: (p(action = a*), [thetas such that action is optimal for theta]
        """

        posteriors_actions = act_optimally(self.prior_belief, self.assortment_size)
        posteriors_actions = [tuple(posteriors_actions[ix, :]) for ix in range(self.n_samples)]
        optimal_actions_information = defaultdict(list)
        for ix, action in enumerate(posteriors_actions):
            optimal_actions_information[action].append(ix)

        self.optimal_actions = {action: (len(theta_idxs) / self.n_samples, theta_idxs) for
                                action, theta_idxs in optimal_actions_information.items()}

    def compute_delta(self):
        self.update_r_star()
        self.delta_ = partial(delta_full,
                              sampled_preferences=self.prior_belief,
                              r_star=self.r_star)

    def compute_g(self):
        self.g_ = partial(g_full,
                          sampled_preferences=self.prior_belief,
                          opt_actions=self.optimal_actions)

    def proposal(self):
        self.prior_belief = self.sample_from_posterior(self.n_samples)
        # print(f"belief sampled is: {self.prior_belief}")
        self.update_r_star()
        self.compute_delta()
        self.update_optimal_actions()
        # print(f"optimal actions are: {self.optimal_actions}")
        self.compute_g()
        action = np.array(ids_action_selection(n=self.n_items,
                                               k=self.assortment_size,
                                               delta_=self.delta_,
                                               g_=self.g_))
        self.current_action = action
        # print("-" * 15)
        return action

    def reset(self):
        self.epoch_ended = True
        self.current_action = self.n_items
        self.epoch_picks = defaultdict(int)
        self.posterior_parameters = [(1, 1) for _ in range(self.n_items)]
        self.optimal_actions = None
        self.g_ = None
        self.r_star = 0.
        self.delta_ = None
        self.n_samples = N_SAMPLES_IDS
        self.prior_belief = self.sample_from_posterior(self.n_samples)

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        if item_selected == self.n_items:
            self.epoch_ended = True
            # print(f"former posterior parameters where: {self.posterior_parameters}")
            n_is = [int(ix in self.current_action) for ix in range(self.n_items)]
            # print("current action", self.current_action)
            # print("nis", n_is)
            v_is = [self.epoch_picks[i] for i in range(self.n_items)]
            # print("epoch picks", self.epoch_picks)
            # print("vis", v_is)
            self.posterior_parameters = [(a + n_is[ix], b + v_is[ix]) for ix, (a, b) in
                                         enumerate(self.posterior_parameters)]
            # print(f"Now they are {self.posterior_parameters}")
            self.epoch_picks = defaultdict(int)
        else:
            self.epoch_picks[item_selected] += 1
            self.epoch_ended = False
        return reward


if __name__ == "__main__":
    pref = np.random.rand(2, 4)
    opt_act = {(0, 1): [1., [0, 1]]}
    actions = possible_actions(4, 2)
    for action in actions:
        print(g_full(action, pref, opt_act))
