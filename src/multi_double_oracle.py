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


def find_mixed_NE(
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
    attack_profiles: List[List[Callable]],
    defense_profile: List[Callable],
    attack_strategies: List[np.ndarray],
    defense_strategy: List[float],
):
    """
    Returns the payoff for each defender x attacker combination
    i.e. Return U_D^l (\pi_D, \pi_l) forall l in L
    :return: List of U_D^l
    """
    assert len(attack_profiles) == len(attack_strategies)

    total_discount_reward = 0

    attackers_policies = [
        np.random.choice(attack_profile, MAX_EPISODES, p=attack_strategy)
        for attack_profile, attack_strategy in zip(attack_profiles, attack_strategies)
    ]

    defender_policies = np.random.choice(
        defense_profile, MAX_EPISODES, p=defense_strategy
    )

    initial_state = Model.State(model)

    for episode_num in range(MAX_EPISODES):
        state = initial_state
        episode_reward = 0.0
        defender_policy = defender_policies[episode_num]
        attackers_policy = [
            attacker_policies[episode_num] for attacker_policies in attackers_policies
        ]

        for step_num in range(MAX_STEPS):
            next_state = model.next_state("old", state, defender_policy, attackers_policy)
            loss = next_state.U - state.U # FIXME: Need utility diff of each attacker, 
            state = next_state
            step_reward = -loss # FIXME: Will be -np.sum(losses) over all attackers
            episode_reward += GAMMA ** step_num * step_reward

        total_discount_reward += episode_reward

    avg_discounted_reward = total_discount_reward / MAX_EPISODES
    return avg_discounted_reward # FIXME: Need average disc reward of each attacker as well?

def get_payoff(
    model: Model, attack_policy: Callable, defense_policy: Callable
):
    avg_discount_reward = get_payoff_mixed(
        model, [attack_policy], [defense_policy], [1.0], [1.0]
    )
    return avg_discount_reward

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
