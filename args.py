import argparse


def get_experiment_args(run_or_plot):
    parser = argparse.ArgumentParser()

    if run_or_plot == "run":
        # AGENT SELECTED TO RUN EXP
        parser.add_argument(
            "--agent",
            type=str,
            required=True,
            help="choice of ts, ids, rd, ets, eids, vts",
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
        help="identifier for the experiment, beyond agent_n_k_horizon",
    )

    # ENV PARAMETERS
    parser.add_argument("-n", type=int, default=10, help="number of items available")
    parser.add_argument("-k", type=int, default=3, help="size of the assortments")
    parser.add_argument(
        "--prior",
        type=str,
        default="uniform",
        help="possible values: 'restricted', 'uniform'",
    )

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

    # Information Directed Sampling PARAMETERS
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
        default=1,
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

    # HYPERMODEL PARAMETERS
    parser.add_argument("--reg_weight", type=float, default=1.0)
    parser.add_argument("--training_sigmaobs", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-1)  # 1e-3 and reg 0.1 for mlp
    parser.add_argument("--model_input_dim", type=int, default=10)
    parser.add_argument("--nsteps", type=int, default=25)
    parser.add_argument("--printinterval", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--nzsamples", type=int, default=32)

    return parser.parse_args()
