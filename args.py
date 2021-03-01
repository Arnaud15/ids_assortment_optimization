import argparse
import os

RAW_OUTPUTS_FOLDER = "raw_outputs"
if not os.path.isdir(RAW_OUTPUTS_FOLDER):
    os.makedirs(RAW_OUTPUTS_FOLDER)
AGG_OUTPUTS_FOLDER = "aggregated_outputs"
if not os.path.isdir(AGG_OUTPUTS_FOLDER):
    os.makedirs(AGG_OUTPUTS_FOLDER)
PLOTS_FOLDER = "plots"
if not os.path.isdir(PLOTS_FOLDER):
    os.makedirs(PLOTS_FOLDER)
SUPPORTED_AGENTS = ["ts", "ids", "rd", "cids"]


def get_experiment_args():
    parser = argparse.ArgumentParser()

    # SCRIPT TYPE
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Choice of 'run', 'agg', 'plot'",
        choices=["run", "agg", "plot"],
    )

    # AGENT SELECTED TO RUN EXP
    parser.add_argument(
        "--agent",
        type=str,
        required=False,
        help="Choice of ts, ids, rd, cids",
        choices=SUPPORTED_AGENTS,
    )

    # AGENT to plot / aggregate data from
    parser.add_argument(
        "--agents",
        type=str,
        required=False,
        help="Select agents for plot / aggregation.",
        nargs="+",
    )

    # ENV PARAMETERS
    parser.add_argument(
        "-N", type=int, default=100, help="Number of items available."
    )
    parser.add_argument(
        "-K", type=int, default=3, help="size of the assortments"
    )
    parser.add_argument(
        "-T",
        type=int,
        default=1000,
        help="Number of random simulations to carry out with agent.",
    )
    parser.add_argument(
        "--prior",
        type=str,
        default="uniform",
        help="Possible values: 'uniform', 'soft_sparse', 'full_sparse'.",
        choices=["uniform", "soft_sparse", "full_sparse"],
    )
    parser.add_argument(
        "-p",
        type=float,
        default=0.15,
        help="Proba for the fallback item to be picked.",
    )

    # BASIC EXP PARAMETERS
    parser.add_argument(
        "--nruns",
        type=int,
        default=1,
        help="Number of random simulations to carry out with agent.",
    )

    # Thompson Sampling PARAMETERS
    parser.add_argument(
        "--sampling",
        type=int,
        default=0,
        help="0: default, 1: correlated sampling, 2: optimistic sampling",
        choices=[0, 1, 2],
    )
    parser.add_argument(
        "--optim_prob",
        type=float,
        default=None,
        help="Optimism probability for the TSCS sparse agent.",
    )

    # Information Directed Sampling PARAMETERS
    parser.add_argument(
        "--info_type",
        type=str,
        default="gain",
        help="Choice of 'gain' and 'variance'.",
        choices=["gain", "variance"],
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="exact",
        help="Choice of 'exact' and 'lambda'.",
        choices=["exact", "lambda"],
    )
    parser.add_argument(
        "--dynamics",
        type=str,
        default="step",
        help="Choice of 'step' and 'epoch'.",
        choices=["step", "epoch"],
    )
    parser.add_argument(
        "--M",
        type=int,
        default=1000,
        help="Number of posterior samples for IDS.",
    )
    parser.add_argument(
        "--scaling",
        type=str,
        default="autoreg",
        help="Choice of 'autoreg' and 'time'.",
        choices=["autoreg", "time"],
    )
    parser.add_argument(
        "--D",
        type=float,
        default=0.0,
        help="Regret threshold for satisficing IDS",
    )
    return parser.parse_args()


"""
Danger Zone:
The following global constants should (almost)
never be touched.
"""

# EPOCH BASED SAMPLING PARAMS
# Whether to employ the exploration bonus introduced in the paper
PAPER_EXPLORATION_BONUS = False
# Whether to employ the paper's faulty gaussian approximations
PAPER_UNDEFINED_PRIOR = True

# SOFT_SPARSE SETTING PARAMS
BAD_ITEM_CONSTANT = 0.5  # soft_sparse preference for bad items
TOP_ITEM_CONSTANT = 1.0  # preference for (know) top item in soft-sparse
