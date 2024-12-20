#!/usr/bin/env python3
import argparse
import logging
import multiprocessing
import os
import pickle
import random
import sys
from functools import partial
from test import (test_attack_action, test_defense_action, test_defense_newest,
                  test_defense_suricata, test_model_fraud)
from typing import Callable, List, Union

import numpy as np
import tensorflow as tf
from scipy import optimize as op

from config import config
from ddpg import AttackerOracle, DefenderOracle
# from listutils import flatten_lists
from model import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

"""Implementation of double-oracle algorithms."""

##################### CONFIGURATION ######################
MAX_EPISODES = config.getint("parameter", "max_episodes")
MAX_STEPS = config.getint("parameter", "max_ep_steps")
GAMMA = config.getfloat("parameter", "gamma")
MAX_ITERATION = config.getint("double-oracle", "max_iteration")
N_TRIAL = config.getint("double-oracle", "n_trial")
##########################################################


def find_mixed_NE(payoff: np.ndarray):
    """
    Function for returning mixed strategies of the first step of double oracle iterations.
    :param payoff: Two dimensinal array. Payoff matrix of the players.
                    The row is defender and column is attacker.
    :return: List, mixed strategy of the attacker and defender at NE by solving maxmini problem.
    """
    # This implementation is based on page 88 of the book multiagent systems (Shoham etc.)
    n_action = payoff.shape[0]
    c = np.zeros(n_action)
    c = np.append(c, 1)
    A_ub = np.concatenate((payoff, np.full((n_action, 1), -1)), axis=1)
    b_ub = np.zeros(n_action)
    A_eq = np.ones(n_action)
    A_eq = np.append(A_eq, 0)
    # => A_eq = 1 - c
    A_eq = np.expand_dims(A_eq, axis=0)
    b_eq = np.array([1])
    bound = ()
    for i in range(n_action):
        bound += ((0, None),)
    bound += ((None, None),)
    res_attacker = op.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bound)
    c = -c
    A_ub = np.concatenate((-payoff.T, np.full((n_action, 1), 1)), axis=1)
    res_defender = op.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bound)
    return (
        list(res_attacker.x[0:n_action]),
        list(res_defender.x[0:n_action]),
        res_attacker.fun,
    )


def core_LP_defender(payoffs: List[np.ndarray], attacker_weights: List):
    """
    Function for returning mixed strategies of the first step of double oracle iterations.
    :param payoff: Two dimensinal array. Payoff matrix of the players. The row is defender and column is attcker.
    :return: List, mixed strategy of the attacker and defender at NE by solving maxmini problem.
    """
    # This implementation is based on page 88 of the book multiagent systems (Shoham etc.)
    assert len(payoffs) > 0
    assert len(payoffs) == len(attacker_weights)

    n_action = payoffs[0].shape[0]
    c = np.zeros(n_action)
    c = np.append(c, -1)
    c = -c
    A_ub = np.sum(
        [
            np.concatenate([weight * -payoff.T, np.full((n_action, 1), 1)], axis=1)
            for payoff, weight in zip(payoffs, attacker_weights)
        ]
    )
    b_ub = np.zeros(n_action)
    A_eq = np.full(n_action, 1)
    A_eq = np.append(A_eq, 0)
    A_eq = np.expand_dims(A_eq, axis=0)
    b_eq = np.array([1])
    bounds = [
        *[(0, None)] * n_action,  # Gives: [(0, None), ...n_action times..., (0, None)]
        (None, None),
    ]  # Gives: [(0, None), ...n_action times..., (0, None), (None, None)]

    res_defender = op.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds)

    # return list(res_attacker.x[0:n_action]), list(res_defender.x[0:n_action]), res_attacker.fun
    return list(res_defender.x[0:n_action]), res_defender.fun


def core_LP_attacker(payoff: np.ndarray):
    n_action = payoff.shape[1]
    c = np.zeros(n_action)
    c = np.append(c, 1)
    A_ub = np.concatenate((payoff, np.full((n_action, 1), -1)), axis=1)
    b_ub = np.zeros(n_action)
    A_eq = np.full(n_action, 1)
    A_eq = np.append(A_eq, 0)
    A_eq = np.expand_dims(A_eq, axis=0)
    b_eq = np.array([1])
    bounds = [
        *[(0, None)] * n_action,  # Gives: [(0, None), ...n_action times..., (0, None)]
        (None, None),
    ]  # Gives: [(0, None), ...n_action times..., (0, None), (None, None)]

    res_attacker = op.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds)
    return res_attacker.x[0:n_action], res_attacker.fun


def mixed_NE(
    payoffs: List[np.ndarray], attacker_weights: Union[List[float], np.ndarray]
):
    defender_strategy, defender_utility = core_LP_defender(payoffs, attacker_weights)
    attacker_strategies, attacker_utilities = [], []

    for attacker_idx in len(attacker_weights):
        attacker_strategy, attacker_utility = core_LP_attacker(payoffs[attacker_idx])
        attacker_strategies.append(attacker_strategy)
        attacker_utilities.append(attacker_utility)

    logging.info(f"Utility diff: {defender_utility - np.sum(attacker_utilities)}")
    return defender_strategy, attacker_strategies, defender_utility


def get_payoff_mixed(
    model: Model,
    attack_profile: List[Callable],
    defense_profile: List[Callable],
    attack_strategy: List[float],
    defense_strategy: List[float],
) -> float:
    """
    Function for computing the payoff of the defender given its mixed strategy and the mixed strategy of the attacker.
    :param model: Model of the alert prioritization problem (i.e., Model object).
    :param attack_profile: List of attack policies given a model and a state.
    :param defense_profile: List of defense policies.
    :param attack_strategy: List of probablities of choosing policy from the attack profile
    :param defense_strategy: List of probablities of choosing policy from the defense profile
    :return: The expected discounted reward.
    """
    total_discount_reward = 0

    attack_policies = np.random.choice(attack_profile, MAX_EPISODES, p=attack_strategy)
    defense_policies = np.random.choice(
        defense_profile, MAX_EPISODES, p=defense_strategy
    )

    initial_state = Model.State(model)

    for episode_num in range(MAX_EPISODES):
        state = initial_state
        episode_reward = 0.0
        defense_policy = defense_policies[episode_num]
        attack_policy = attack_policies[episode_num]
        for step_num in range(MAX_STEPS):
            next_state = model.next_state("old", state, defense_policy, attack_policy)
            loss = next_state.U - state.U
            state = next_state
            step_reward = -1.0 * loss
            episode_reward += GAMMA**step_num * step_reward
        total_discount_reward += episode_reward

    avg_discount_reward = total_discount_reward / MAX_EPISODES
    return avg_discount_reward

def get_payoff_mixed(
    model: Model,
    attack_profile: List[Callable],
    defense_profile: List[Callable],
    attack_strategy: List[float],
    defense_strategy: List[float],
) -> float:
    """
    Function for computing the payoff of the defender given its mixed strategy and the mixed strategy of the attacker.
    :param model: Model of the alert prioritization problem (i.e., Model object).
    :param attack_profile: List of attack policies given a model and a state.
    :param defense_profile: List of defense policies.
    :param attack_strategy: List of probablities of choosing policy from the attack profile
    :param defense_strategy: List of probablities of choosing policy from the defense profile
    :return: The expected discounted reward.
    """
    total_discount_reward = 0

    attack_policies = np.random.choice(attack_profile, MAX_EPISODES, p=attack_strategy)
    defense_policies = np.random.choice(
        defense_profile, MAX_EPISODES, p=defense_strategy
    )

    initial_state = Model.State(model)

    for episode_num in range(MAX_EPISODES):
        state = initial_state
        episode_reward = 0.0
        defense_policy = defense_policies[episode_num]
        attack_policy = attack_policies[episode_num]
        for step_num in range(MAX_STEPS):
            next_state = model.next_state("old", state, defense_policy, attack_policy)
            loss = next_state.U - state.U
            state = next_state
            step_reward = -1.0 * loss
            episode_reward += GAMMA**step_num * step_reward
        total_discount_reward += episode_reward

    avg_discount_reward = total_discount_reward / MAX_EPISODES
    return avg_discount_reward


def get_payoff(
    model: Model, attack_policy: Callable, defense_policy: Callable
) -> float:
    """
    Function for computing the payoff of the defender given its strategy and the strategy of the attacker.
    :param model: Model of the alert prioritization problem (i.e., Model object).
    :param attack_policies: Function takes a model and a state, returns the portion of budget allocated for each type of attacks given a model and a state.
    :param defense_policy: Function which takes a model and a state, returns the portion of budget allocated for each type of alerts with all ages given a model and a state.
    :return: The expected discounted reward.
    """
    avg_discount_reward = get_payoff_mixed(
        model, [attack_policy], [defense_policy], [1.0], [1.0]
    )
    return avg_discount_reward


def update_profile(
    model: Model,
    payoff: List[List[float]],
    attack_profile: List[Callable],
    defense_profile: List[Callable],
    attack_policy: Callable[[Model, Model.State], List],
    defense_policy: Callable[[Model, Model.State], List],
):
    """
    Function for updating the payoff matrix and the action profile of defender and attacker
    :param model: Model of the alert prioritization problem (i.e., Model object).
    :param payoff: Two dimensinal array. Payoff matrix of the players. The row is defender and column is attacker.
    :param attack_profile: List of attack policies.
    :param defense_profile: List of defense policies.
    :param attack_policy: New pure strategy of the attacker
    :param defense_policy: New pure strategy of the defender
    :return: updated payoff matrix, attack_profile and defense_profile
    """
    n_action = payoff.shape[0]

    # A new row and column will be added to the payoff matrix
    new_payoff_col = np.array([])
    new_payoff_row = np.array([])

    # First get the new column
    for i in range(len(defense_profile)):
        new_payoff = get_payoff(model, attack_policy, defense_profile[i])
        new_payoff_col = np.append(new_payoff_col, new_payoff)
    new_payoff_col = np.expand_dims(new_payoff_col, axis=0)
    attack_profile.append(attack_policy)
    payoff = np.concatenate((payoff, new_payoff_col.T), axis=1)

    # Second, get the new row
    for j in range(len(attack_profile)):
        new_payoff = get_payoff(model, attack_profile[j], defense_policy)
        new_payoff_row = np.append(new_payoff_row, new_payoff)
    new_payoff_row = np.expand_dims(new_payoff_row, axis=0)
    defense_profile.append(defense_policy)
    payoff = np.concatenate((payoff, new_payoff_row), axis=0)
    return payoff, attack_profile, defense_profile


def double_oracle(model: Model, exper_index: int):
    # print("Initializing...")
    num_attackers = len(model.adv_weights)
    attack_profile = []
    defense_profile = []
    payoff = []
    payoff_record = []

    # Initialize the payoff matrix
    initial_payoff = get_payoff(model, test_attack_action, test_defense_newest)
    payoff = np.array([[initial_payoff]])
    # Initialize the action profile
    attack_profile = [test_attack_action]
    defense_profile = [test_defense_newest]
    initial_defense_size, initial_attack_size = payoff.shape

    # Compute new strategies and actions
    for i in range(MAX_ITERATION):
        attack_strategy, defense_strategy, utility = find_mixed_NE(payoff)
        payoff_record.append(utility)

        print(
            "###########################################################################################"
        )
        print("Iteration", i)

        print("Current payoff matrix:")
        print(payoff)

        print("Attacker's mixed strategy:")
        print(attack_strategy)
        print("Defender's mixed strategy:")
        print(defense_strategy)

        logging.info("Iteration {}: {}".format(i, payoff_record))
        utility_defender_newest = np.dot(payoff[0], attack_strategy)
        utility_attacker_uniform = np.dot(payoff[:, 0], defense_strategy)

        # Get new response to the mixed strategy
        attack_response: List[AttackerOracle] = []
        attack_utility: List[float] = []
        defense_response: List[DefenderOracle] = []
        defense_utility: List[float] = []
        for k in range(N_TRIAL):
            attack_response.append(
                AttackerOracle(
                    model, defense_profile, defense_strategy, exper_index, i, k
                )
            )
            defense_response.append(
                DefenderOracle(
                    model, attack_profile, attack_strategy, exper_index, i, k
                )
            )
            attack_utility.append(attack_response[k].agent.utility)
            defense_utility.append(defense_response[k].agent.utility)

        # Get the best defense and attack policy
        attack_index = attack_utility.index(max(attack_utility))
        defense_index = defense_utility.index(max(defense_utility))

        # FIXME: What if this policy is already in the matrix?
        attack_policy = attack_response[attack_index].agent.policy
        defense_policy = defense_response[defense_index].agent.policy

        # Delete the models that are not the selected one
        for k in range(N_TRIAL):
            if k != defense_index:
                cmd = "rm -rf ../model/defender-{}-{}-{}".format(exper_index, i, k)
                os.system(cmd)
            if k != attack_index:
                cmd = "rm -rf ../model/attacker-{}-{}-{}".format(exper_index, i, k)
                os.system(cmd)

        # Update profile
        payoff, attack_profile, defense_profile = update_profile(
            model,
            payoff,
            attack_profile,
            defense_profile,
            attack_policy,
            defense_policy,
        )

        # Test the terminate condition
        attack_pure_utility = -np.dot(
            payoff[:, i + initial_attack_size][
                0 : (len(payoff[:, i + initial_attack_size]) - 1)
            ],
            defense_strategy,
        )
        defense_pure_utility = np.dot(
            payoff[i + initial_defense_size][
                0 : (len(payoff[i + initial_defense_size]) - 1)
            ],
            attack_strategy,
        )

        if -utility >= attack_pure_utility and utility >= defense_pure_utility:
            # Save the mixed strategies
            pickle.dump(
                defense_strategy,
                open("../model/defender-strategy-{}.pickle".format(exper_index), "wb"),
            )
            pickle.dump(
                attack_strategy,
                open("../model/attacker-strategy-{}.pickle".format(exper_index), "wb"),
            )
            break
        if i == MAX_ITERATION - 1:
            pickle.dump(
                defense_strategy,
                open("../model/defender-strategy-{}.pickle".format(exper_index), "wb"),
            )
            pickle.dump(
                attack_strategy,
                open("../model/attacker-strategy-{}.pickle".format(exper_index), "wb"),
            )
    return payoff_record[-1]


def multi_double_oracle(model: Model, exper_indx: int):
    num_attackers = len(model.adv_weights)

    attack_profile = []
    defense_profile = []
    payoff = []
    payoff_record = []

    initial_payoffs = np.array([
        get_payoff(
            model,
            partial(
                test_attack_action,
            ),
        )
        for n in num_attackers
    ])





def test_mixed_NE():
    """
    Test find_mixed_NE with the Matching Pennies game and Rock-Paper-Scissors on page 58 of Multiagent Systems.
    The mixed strategy at NE should be 0.5 for each player
    """
    # payoff = np.array([[1,-1],[-1,1]]) # Matching Pennies
    payoff = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])  # Rock, Paper, Scissors
    attack_strategy, defense_strategy, utility = find_mixed_NE(payoff)
    print(attack_strategy, defense_strategy)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s / %(levelname)s: %(message)s", level=logging.DEBUG
    )
    logging.info("Experiment starts.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="suricata")
    parser.add_argument("--def-budget", type=float)
    parser.add_argument("--adv-budgets", nargs="*", type=float)
    parser.add_argument("--num-exp", type=int)

    args = parser.parse_args()

    if args.model_name == "fraud":
        model = test_model_fraud(args.def_budget, args.adv_budget)
    # elif args.model_name == "suricata":
    #     model = test_model_suricata(args.def_budget, args.adv_budget)
    else:
        raise NotImplementedError()

    def evaluation(exper_index):
        random_seed = exper_index
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        do_utility = double_oracle(model, exper_index)
        return do_utility

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    utilities = []
    for do_utility in pool.imap(evaluation, range(args.num_exp)):
        utilities.append(do_utility)
    logging.info("The utility of the agent:")
    print(utilities)
