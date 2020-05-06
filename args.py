import argparse
import os

# EPOCH BASED SAMPLING PARAMS
# Whether to employ the exploration bonus introduced in the paper
PAPER_EXPLORATION_BONUS = False
# Whether to employ the paper's faulty gaussian approximations
PAPER_UNDEFINED_PRIOR = True
BETA_RVS = True

# SOFT_SPARSE SETTING PARAMS
BAD_ITEM_CONSTANT = 0.5  # soft_sparse preference for bad items
TOP_ITEM_CONSTANT = 1.0  # preference for (know) top item in soft-sparse
OUTPUTS_FOLDER = "outputs"
if not os.path.isdir(OUTPUTS_FOLDER):
    os.makedirs(OUTPUTS_FOLDER)


def get_experiment_args(run_or_plot):
    parser = argparse.ArgumentParser()

    if run_or_plot == "run":
        # AGENT SELECTED TO RUN EXP
        parser.add_argument(
            "--agent",
            type=str,
            required=True,
            help="choice of ts, ids, rd, ets, eids",
        )
    elif run_or_plot == "plot":
        # AGENTS TO APPEAR ON PLOT
        parser.add_argument(
            "--agents",
            type=str,
            required=True,
            help="select agents appearing on the plot",
            nargs="+",
        )
    else:
        raise ValueError("Incorrect argument: choice of 'run' or 'plot'")

    # EXP NAME
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="identifier for the experiment, beyond agent_env_params",
    )

    # ENV PARAMETERS
    parser.add_argument(
        "-n", type=int, default=10, help="number of items available"
    )
    parser.add_argument(
        "-k", type=int, default=3, help="size of the assortments"
    )
    parser.add_argument(
        "--prior",
        type=str,
        default="full_sparse",
        help="possible values: 'uniform', 'soft_sparse', 'full_sparse'",
    )
    parser.add_argument(
        "-p",
        type=float,
        default=0.5,
        help="proba for the fallback item to be picked",
    )
    # TODO env prior and agents must be compatible

    # BASIC EXP PARAMETERS
    parser.add_argument(
        "--horizon",
        type=int,
        default=5000,
        help="number of random simulations to carry out with agent",
    )
    parser.add_argument(
        "--nruns",
        type=int,
        default=100,
        help="number of random simulations to carry out with agent",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="print intermediate info during episodes or not",
    )

    # Thompson Sampling PARAMETERS
    parser.add_argument(
        "--correlated_sampling",
        type=int,
        default=1,
        help="correlated sampling or no",
    )

    # Information Directed Sampling PARAMETERS
    parser.add_argument(
        "--info_type",
        type=str,
        default="gain",
        help="choice of 'gain' and 'variance'",
    )
    parser.add_argument(
        "--ids_samples",
        type=int,
        default=100,
        help="number of posterior samples for IDS",
    )
    parser.add_argument(
        "--ids_action_selection",
        type=str,
        default="exact",
        help="action selection: exact O(A**2), approximate O(A), greedy O(NK)",
    )
    parser.add_argument(
        "--greedy_scaler",
        type=float,
        default=0.00316,
        help="scaling factor for greedy action selection",
    )
    parser.add_argument(
        "--find_best_scaler",
        type=int,
        default=0,
        help="execute grid search for best greedy scaler",
    )
    parser.add_argument(
        "--best_scaler_h",
        type=int,
        default=1000,
        help="horizon for runs in the grid search",
    )
    parser.add_argument(
        "--best_scaler_n",
        type=int,
        default=10,
        help="number of runs in the grid search",
    )
    parser.add_argument(
        "-m",
        type=int,
        default=25,
        help="number of actions in action space reduction",
    )
    return parser.parse_args()
