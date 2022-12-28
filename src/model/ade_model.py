"""
Model of the alert prioritization problem, including alert and attack types.

Author: Aditya Chopra
Using sources from Aron Laszka, Liang Tong
17 Dec, 2022
"""

import copy
import logging
from random import Random
from typing import Callable, List, Optional, Union

import numpy as np
import numpy.typing as npt

from model.alert_type import AlertType
from model.attack_type import AttackType

EPSILON = 1e-5


def prob_investigation(
    n: int, m: float, k: Union[int, float], r: Union[bool, int, float]
):
    """
    Get the the probability of an attack being investigated.
    It is the probability that at least one true alert is investigated.
    :param n: Number of alert type t.
    :param m: Number of true alerts if an attack a is mounted.
    :param k: Number of investigations.
    :param r: A boolean value showing whether an attack a raises alert t.
    """
    result = 0
    # print(n, m, k, r)
    if n == 0 or r == 0:
        result = 0
    else:
        result = 1 - np.product([1 - np.ceil(m) * 1.0 / (n - i) for i in range(int(k))])
    return result

class Model:
    """Game-theoretic model of the alert prioritization problem."""

    def __init__(
        self,
        horizon: int,
        alert_types: List[AlertType],
        attack_types: List[AttackType],
        def_budget: float,
        adv_budgets: npt.ArrayLike[np.float64],
        adv_weights: npt.ArrayLike[np.float64],
    ):
        """
        Construct a model object.
        :param horizon: Time horizon (i.e., H).
        :param alert_types: Types of alerts (i.e., T). Single-dimensional list of AlertType objects.
        :param attack_types: Types of attacks (i.e., A). Single-dimensional list of AttackType objects.
        :param def_budget: Budget of the defender (i.e., B).
        :param adv_budgets: Budget of the adversarys (i.e., D).
        """
        self.horizon = horizon
        self.alert_types = alert_types
        self.attack_types = attack_types
        self.def_budget = def_budget
        self.adv_budgets = np.array(adv_budgets)
        self.adv_weights = np.array(adv_weights)
        self.rnd = Random(0)  # FIXME: Is this needed?

        # Validation
        self._validate()

        self.num_alert_types = len(self.alert_types)
        self.num_attack_types = len(self.attack_types)
        self.num_attackers = len(self.adv_budgets)

    class State:
        """State of the game in a certain time step."""

        def __init__(
            self,
            model: "Model",
            N: Optional[npt.NDArray[np.int]] = None,
            M: Optional[npt.NDArray[np.bool]] = None,
            R: Optional[npt.NDArray[np.bool]] = None,
            U: Optional[float] = None,
        ):
            """
            Construct a state object.
            :param model: Model of the alert prioritization problem (i.e., Model object).
            :param N: Number of yet uninvestigated alerts. Two-dimensional list, N[h][t] is the number of uninvestigated alerts of type t raised h time steps ago.
            :param M: Indicator of undetected attacks. Two-dimensional list, M[h][a][l] == 1 if an attack of type a was mounted by attacker l, h time steps ago, M[h][a][l] == 0 otherwise.
            :param R: Indicator of true alerts. Three-dimensional list, R[h][a][t][l] == 1 if an alert of type t was raised due to an attack of type a, mounted by attacker l, h time steps ago, R[h][a][t][l] == 0 otherwise.
            :param U: Cumulative loss sustained by the defender.
            """
            if N is None:
                N = np.zeros((model.horizon, model.num_alert_types))
                M = np.zeros(
                    (model.horizon, model.num_attack_types, model.num_attackers),
                    dtype=np.bool,
                )
                R = np.zeros(
                    (
                        model.horizon,
                        model.num_attack_types,
                        model.num_alert_types,
                        model.num_attackers,
                    ),
                    dtype=np.bool,
                )
                U = 0.0

            assert N.shape == (model.horizon, model.num_alert_types)
            assert M.shape == (
                model.horizon,
                model.num_attack_types,
                model.num_attackers,
            )
            assert R.shape == (
                model.horizon,
                model.num_attack_types,
                model.num_alert_types,
                model.num_attackers,
            )

            # self.model = model # This is not used anywhere. Just remove it.
            self.N = N
            self.M = M
            self.R = R
            self.U = U

        def __str__(self):
            return "N: {}, M: {}, R: {}, U: {}".format(self.N, self.M, self.R, self.U)

    def _validate(self):
        assert self.horizon > 0
        for a in self.attack_types:
            assert len(a.loss) == self.horizon
            assert len(a.pr_alert) == len(self.alert_types)
        assert self.def_budget > 0
        assert np.all(self.adv_budgets > 0)

        assert self.adv_budgets.shape == self.adv_weights.shape
        assert np.all(self.adv_weights > 0)
        assert self.adv_weights.sum() == 1

    def is_feasible_investigation(
        self, N: npt.NDArray[np.int], delta: npt.NDArray[np.int]
    ):
        """
        Determine if a given investigation action is feasible in a given state.
        :param N: Number of yet uninvestigated alerts. Two-dimensional list, N[h][t] is the number of uninvestigated alerts of type t raised h time steps ago.
        :param delta: Number of alerts to investigate. Two-dimensional list, delta[h][t] is the number of alerts to investigate of type t raised h time steps ago.
        :return: True if the investigation action is feasible, false otherwise.
        """

        # validate passed params
        assert N.shape == (self.horizon, self.num_alert_types)
        assert delta.shape == (self.horizon, self.num_alert_types)

        cost = 0.0
        for h in range(self.horizon):
            for t in range(len(self.alert_types)):
                if delta[h][t] > N[h][t]:
                    return False
                cost += self.alert_types[t].cost * delta[h][t]

        if cost > self.def_budget:
            logging.debug(
                f"cost of investigation > self.def_budge: {cost}, {self.def_budget}"
            )

        return cost <= self.def_budget

    def make_investigation_feasible(
        self, N: npt.NDArray[np.int], delta: npt.NDArray[np.int]
    ):
        """
        Compute an investigation action that is feasible in a given state and resembles the given investigation action.

        :param N: Number of yet uninvestigated alerts. Two-dimensional list, N[h][t] is the number of uninvestigated alerts of type t raised h time steps ago.
        :param delta: Number of alerts to investigate. Two-dimensional list, delta[h][t] is the number of alerts to investigate of type t raised h time steps ago.
        :return: Feasible investigation action.
        """
        assert N.shape == (self.horizon, self.num_alert_types)
        assert delta.shape == (self.horizon, self.num_alert_types)

        cost = 0.0
        for h in range(self.horizon):
            for t in range(len(self.alert_types)):
                cost += self.alert_types[t].cost * delta[h][t]

        if cost == 0:
            return delta

        factor = self.def_budget / cost
        delta_feasible = np.minimum(np.floor(delta * factor), N)

        # delta_feasible = []
        # for h in range(self.horizon):
        #     # delta_feasible.append([min(int(delta[h][t] * factor-1), N[h][t]) for t in range(len(self.alert_types))])
        #     delta_feasible.append(
        #         [
        #             min(int(delta[h][t] * factor), N[h][t])
        #             for t in range(len(self.alert_types))
        #         ]
        #     )

        return delta_feasible

    def is_feasible_attack(self, alpha: npt.NDArray[np.float]):
        """
        Determine if a given attack action is feasible.
        :param alpha: Probability of mounting attacks. One-dimensional list, alpha[a] is the probability of mounting an attack of type a.
        :return: True if the attack action is feasible, false otherwise.
        """
        assert alpha.shape[0] == self.attack_types

        cost = 0.0
        for a in range(len(self.attack_types)):
            cost += self.attack_types[a].cost * alpha[a]

        if cost > self.adv_budget:
            logging.debug(
                f"cost of investigation > self.adv_budge: {cost}, {self.adv_budget}"
            )

        return cost <= self.adv_budget

    def make_attack_feasible(self, alpha: npt.NDArray[np.float]):
        """
        Compute an attack action that is feasible and resembles the given attack action.
        :param alpha: Attack action (see function is_feasible_attack).
        :return: Feasible attack action.
        """
        assert alpha.shape[0] == self.attack_types

        cost_alpha = 0.0
        for a in range(len(self.attack_types)):
            cost_alpha += self.attack_types[a].cost * alpha[a]

        if cost_alpha == 0:
            return alpha

        factor = self.adv_budget / cost_alpha
        alpha_feasible = np.maximum(alpha * factor - EPSILON, 0)

        # alpha_feasible = []
        # for a in range(len(self.attack_types)):
        #     alpha_feasible.append(max(alpha[a] * factor - EPSILON, 0))

        return alpha_feasible

    def split_alpha(self, delta: npt.NDArray, splitting: str = "uniform"):
        split_alphas_shape = (
            self.horizon,
            self.num_attackers,
            self.num_alert_types,
        )

        # assert delta.shape == (self.horizon, self.num_alert_types)

        if splitting == "greedy":
            raise NotImplementedError
        elif splitting == "uniform":
            split_alphas = np.random.uniform(0, 1, size=split_alphas_shape)
            # Normalize accross number of attackers.
            split_alphas = (split_alphas / np.expand_dims(split_alphas.sum(axis=1), axis=1)) 
            # Rescale so that sum on axis 1 is the original delta values.
            split_alphas = np.expand_dims(delta, axis=1) * split_alphas

        else:
            raise NotImplementedError

        return split_alphas

    def next_state(
        self,
        mode,
        state: State,
        delta: Union[
            npt.NDArray, List[List[float]], Callable[["Model", State], npt.NDArray]
        ],
        alpha,
        rnd=None,
    ):
        """
        Compute the next state of the game given a defense action and an adversarial strategy.
        Note that the defense action delta is the specific number of alerts delta to investigate,
        while the adversary strategy alpha is a policy that returns the attack action given the state of the game.
        :param state: State of the alert prioritization problem (i.e., Model.State object).
        :param delta: Number of alerts to investigate. Two-dimensional list, delta[h][t] is the number of alerts to investigate of type t raised h time steps ago.
        :param alpha: Attack policy. Function, takes a model and a state, returns the probability of mounting attacks (one-dimensional list) given a model and a state.
        :param rnd: Random number generator.
        :return: Next state (i.e., Model.State object).
        """
        if isinstance(delta, list):
            delta = np.array(delta, copy=True)
        elif isinstance(
            delta,
        ):
            delta = copy.deepcopy(delta)
        else:
            delta = delta(self, state)
        delta = self.make_investigation_feasible(state.N, delta)
        assert self.is_feasible_investigation(state.N, delta)

        # FIXME: Where is rnd even used?
        if rnd is None:
            rnd = self.rnd
        next = Model.State(self)

        # print("state.M before investigation:", state.M)
        # 1. Attack investigation
        M_now = copy.deepcopy(state.M)
        split_alphas = self.split_alpha(delta)
        for l in range(self.num_attackers):
            for a in range(self.num_attack_types):
                for h in range(self.horizon):
                    coin = np.random.random()  # Uniform random number in [0, 1)
                    
                    fact = np.product(
                        [
                            1
                            # Why not just use R directly since it is an indicator var? Why do a sign?
                            - state.R[h][a][t][l]
                            * prob_investigation(
                                state.N[h][t],
                                self.attack_types[a].pr_alert[t],
                                split_alphas[h][l][t],
                                state.R[h][a][t][l],
                            )
                            for t in range(self.num_alert_types)
                        ]
                    )
                    
                    # fact = product([scipy.special.comb(state.N[h][t]-state.R[h][a][t], delta[h][t], exact=True) / scipy.special.comb(state.N[h][t], delta[h][t], exact=True) for t in range(len(self.alert_types))])
                    # print("attack: {} coin: {} fact :{}".format(a, coin, fact))
                    if (state.M[h][a] == 1) and (coin < fact):
                        state.M[h][a] = 1
                    else:
                        state.M[h][a] = 0
            # print("state.M after investigation", state.M)

        def_loss = 0.0
        if mode == "new":
            for a in range(self.num_attack_types):
                if M_now[0][a] == 1 and state.M[0][a] == 1:
                    def_loss += self.attack_types[a].loss[0]
                if M_now[0][a] == 1 and state.M[0][a] == 0:
                    def_loss -= self.attack_types[a].loss[0]
        elif mode == "old":
            for a in range(len(self.attack_types)):
                def_loss += self.attack_types[a].loss[0] * state.M[0][a]
        else:
            raise NotImplementedError

        # 2. Attacks
        if isinstance(alpha, list):
            pr_attacks = copy.deepcopy(alpha)
        else:
            pr_attacks = alpha(self, state)
        pr_attacks = self.make_attack_feasible(pr_attacks)
        if not self.is_feasible_attack(pr_attacks):
            print(pr_attacks)
        assert self.is_feasible_attack(pr_attacks)
        for a in range(len(self.attack_types)):
            if np.random.random() < pr_attacks[a]:
                next.M[0][a] = 1
            else:
                next.M[0][a] = 0

        # print(pr_attacks, delta)
        next.U = state.U + def_loss

        # 3. True alerts
        for a in range(len(self.attack_types)):
            for t in range(len(self.alert_types)):
                # next.R[0][a][t] = self.attack_types[a].pr_alert[t] * next.M[0][a]
                if np.random.random() < self.attack_types[a].pr_alert[t] * next.M[0][a]:
                    next.R[0][a][t] = np.ceil(self.attack_types[a].pr_alert[t])
                else:
                    next.R[0][a][t] = 0

        # 4. Alerts
        for t in range(len(self.alert_types)):
            next.N[0][t] = self.alert_types[t].false_alerts.generate() + sum(
                (next.R[0][a][t] for a in range(len(self.attack_types)))
            )
        return next
