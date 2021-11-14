# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Estimators."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
from sklearn.utils import check_scalar

from .helper import estimate_high_probability_upper_bound_bias
from ..utils import (
    estimate_confidence_interval_by_bootstrap,
    check_ope_inputs,
    check_ope_inputs_tensor,
)


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators."""

    @abstractmethod
    def _estimate_round_rewards(self) -> Union[np.ndarray, torch.Tensor]:
        """Estimate round-wise (or sample-wise) rewards."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value of evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value_tensor(self) -> torch.Tensor:
        """
        Estimate the policy value of evaluation policy and return PyTorch Tensor.
        This is intended for being used with NNPolicyLearner.
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure."""
        raise NotImplementedError


@dataclass
class ReplayMethod(BaseOffPolicyEstimator):
    """Relpay Method (RM).

    Note
    -------
    Replay Method (RM) estimates the policy value of evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{RM}} (\\pi_e; \\mathcal{D}) :=
        \\frac{\\mathbb{E}_{\\mathcal{D}}[\\mathbb{I} \\{ \\pi_e (x_t) = a_t \\} r_t ]}{\\mathbb{E}_{\\mathcal{D}}[\\mathbb{I} \\{ \\pi_e (x_t) = a_t \\}]},

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`\\pi_e: \\mathcal{X} \\rightarrow \\mathcal{A}` is the function
    representing action choices by the evaluation policy realized during offline bandit simulation.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.

    Parameters
    ----------
    estimator_name: str, default='rm'.
        Name of the estimator.

    References
    ------------
    Lihong Li, Wei Chu, John Langford, and Xuanhui Wang.
    "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms.", 2011.

    """

    estimator_name: str = "rm"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        behavior_action_dist: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (must be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards of each round estimated by the Replay Method.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        action_match = np.array(
            action_dist[np.arange(action.shape[0]), action, position] == 1
        )
        estimated_rewards = np.zeros_like(action_match)
        if action_match.sum() > 0.0:
            estimated_rewards = action_match * reward / action_match.mean()
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        behavior_action_dist: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (must be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")

        check_ope_inputs(
            action_dist=action_dist, position=position, action=action, reward=reward
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
            behavior_action_dist=behavior_action_dist,
        ).mean()

    def estimate_policy_value_tensor(
        self,
        **kwargs,
    ) -> torch.Tensor:
        """
        Estimate the policy value of evaluation policy and return PyTorch Tensor.
        This is intended for being used with NNPolicyLearner.
        This is not implemented for RM because it is indifferentiable.
        """
        raise NotImplementedError(
            "This is not implemented because RM is indifferentiable"
        )

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (must be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")

        check_ope_inputs(
            action_dist=action_dist, position=position, action=action, reward=reward
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class InverseProbabilityWeighting(BaseOffPolicyEstimator):
    """Inverse Probability Weighting (IPW) Estimator.

    Note
    -------
    Inverse Probability Weighting (IPW) estimates the policy value of evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{IPW}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{\\mathcal{D}} [ w(x_t,a_t) r_t],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    When the weight-clipping is applied, a large importance weight is clipped as :math:`\\hat{w}(x,a) := \\min \\{ \\lambda, w(x,a) \\}`
    where :math:`\\lambda (>0)` is a hyperparameter that decides a maximum allowed importance weight.

    IPW re-weights the rewards by the ratio of the evaluation policy and behavior policy (importance weight).
    When the behavior policy is known, IPW is unbiased and consistent for the true policy value.
    However, it can have a large variance, especially when the evaluation policy significantly deviates from the behavior policy.

    Parameters
    ------------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.

    estimator_name: str, default='ipw'.
        Name of the estimator.

    References
    ------------
    Alex Strehl, John Langford, Lihong Li, and Sham M Kakade.
    "Learning from Logged Implicit Exploration Data"., 2010.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambda_: float = np.inf
    estimator_name: str = "ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("lambda_ must not be nan")

    def _estimate_round_rewards(
        self,
        reward: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        pscore: Union[np.ndarray, torch.Tensor],
        action_dist: Union[np.ndarray, torch.Tensor],
        behavior_action_dist: Optional[Union[np.ndarray, torch.Tensor]] = None,
        position: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards of each round estimated by IPW.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)
        return reward * iw

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        behavior_action_dist: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            behavior_action_dist=behavior_action_dist,
        ).mean()

    def estimate_policy_value_tensor(
        self,
        reward: torch.Tensor,
        action: torch.Tensor,
        pscore: torch.Tensor,
        action_dist: torch.Tensor,
        behavior_action_dist: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Estimate the policy value of evaluation policy and return PyTorch Tensor.
        This is intended for being used with NNPolicyLearner.

        Parameters
        ----------
        reward: Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: Tensor, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: Tensor, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        V_hat: Tensor
            Estimated policy value (performance) of a given evaluation policy.

        """
        if not isinstance(reward, torch.Tensor):
            raise ValueError("reward must be Tensor")
        if not isinstance(action, torch.Tensor):
            raise ValueError("action must be Tensor")
        if not isinstance(pscore, torch.Tensor):
            raise ValueError("pscore must be Tensor")

        check_ope_inputs_tensor(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            behavior_action_dist=behavior_action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the MSE score of a given clipping hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given clipping hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n_rounds = reward.shape[0]
        # estimate the sample variance of IPW with clipping
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                position=position,
            )
        )
        sample_variance /= n_rounds

        # estimate the (high probability) upper bound of the bias of IPW with clipping
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        bias_upper_bound = estimate_high_probability_upper_bound_bias(
            reward=reward,
            iw=iw,
            iw_hat=np.minimum(iw, self.lambda_),
        )
        estimated_mse_score = sample_variance + (bias_upper_bound ** 2)

        return estimated_mse_score


@dataclass
class SelfNormalizedInverseProbabilityWeighting(InverseProbabilityWeighting):
    """Self-Normalized Inverse Probability Weighting (SNIPW) Estimator.

    Note
    -------
    Self-Normalized Inverse Probability Weighting (SNIPW) estimates the policy value of evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{SNIPW}} (\\pi_e; \\mathcal{D}) :=
        \\frac{\\mathbb{E}_{\\mathcal{D}} [w(x_t,a_t) r_t]}{ \\mathbb{E}_{\\mathcal{D}} [w(x_t,a_t)]},

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.

    SNIPW re-weights the observed rewards by the self-normalized importance weihgt.
    This estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the true policy value and increases the stability in some senses.
    See the references for the detailed discussions.

    Parameters
    ----------
    estimator_name: str, default='snipw'.
        Name of the estimator.

    References
    ----------
    Adith Swaminathan and Thorsten Joachims.
    "The Self-normalized Estimator for Counterfactual Learning.", 2015.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "snipw"

    def _estimate_round_rewards(
        self,
        reward: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        pscore: Union[np.ndarray, torch.Tensor],
        action_dist: Union[np.ndarray, torch.Tensor],
        behavior_action_dist: Optional[Union[np.ndarray, torch.Tensor]] = None,
        position: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards of each round estimated by the SNIPW estimator.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        return reward * iw / iw.mean()


@dataclass
class DirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM).

    Note
    -------
    DM first learns a supervised machine learning model, such as ridge regression and gradient boosting,
    to estimate the mean reward function (:math:`q(x,a) = \\mathbb{E}[r|x,a]`).
    It then uses it to estimate the policy value as follows.

    .. math::

        \\hat{V}_{\\mathrm{DM}} (\\pi_e; \\mathcal{D}, \\hat{q})
        &:= \\mathbb{E}_{\\mathcal{D}} \\left[ \\sum_{a \\in \\mathcal{A}} \\hat{q} (x_t,a) \\pi_e(a|x_t) \\right],    \\\\
        & =  \\mathbb{E}_{\\mathcal{D}}[\\hat{q} (x_t,\\pi_e)],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`, which supports several fitting methods specific to OPE.

    If the regression model (:math:`\\hat{q}`) is a good approximation to the true mean reward function,
    this estimator accurately estimates the policy value of the evaluation policy.
    If the regression function fails to approximate the mean reward function well,
    however, the final estimator is no longer consistent.

    Parameters
    ----------
    estimator_name: str, default='dm'.
        Name of the estimator.

    References
    ----------
    Alina Beygelzimer and John Langford.
    "The offset tree for learning with partial labels.", 2009.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name: str = "dm"

    def _estimate_round_rewards(
        self,
        action_dist: Union[np.ndarray, torch.Tensor],
        estimated_rewards_by_reg_model: Union[np.ndarray, torch.Tensor],
        behavior_action_dist: Optional[Union[np.ndarray, torch.Tensor]] = None,
        position: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards of each round estimated by the DM estimator.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        n_rounds = position.shape[0]
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]

        if isinstance(action_dist, np.ndarray):
            return np.average(
                q_hat_at_position,
                weights=pi_e_at_position,
                axis=1,
            )
        elif isinstance(action_dist, torch.Tensor):
            return torch.sum(q_hat_at_position * pi_e_at_position, dim=1)
        else:
            raise ValueError("action must be ndarray or Tensor")

    def estimate_policy_value(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        behavior_action_dist: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        if not isinstance(estimated_rewards_by_reg_model, np.ndarray):
            raise ValueError("estimated_rewards_by_reg_model must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            position=position,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
        ).mean()

    def estimate_policy_value_tensor(
        self,
        action_dist: torch.Tensor,
        estimated_rewards_by_reg_model: torch.Tensor,
        position: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Estimate the policy value of evaluation policy and return PyTorch Tensor.
        This is intended for being used with NNPolicyLearner.

        Parameters
        ----------
        action_dist: Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: Tensor, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        V_hat: Tensor
            Estimated policy value (performance) of a given evaluation policy.

        """
        if not isinstance(estimated_rewards_by_reg_model, torch.Tensor):
            raise ValueError("estimated_rewards_by_reg_model must be Tensor")

        check_ope_inputs_tensor(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            position=position,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        if not isinstance(estimated_rewards_by_reg_model, np.ndarray):
            raise ValueError("estimated_rewards_by_reg_model must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            position=position,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DoublyRobust(BaseOffPolicyEstimator):
    """Doubly Robust (DR) Estimator.

    Note
    -------
    Similar to DM, DR first learns a supervised machine learning model, such as ridge regression and gradient boosting,
    to estimate the mean reward function (:math:`q(x,a) = \\mathbb{E}[r|x,a]`).
    It then uses it to estimate the policy value as follows.

    .. math::

        \\hat{V}_{\\mathrm{DR}} (\\pi_e; \\mathcal{D}, \\hat{q})
        := \\mathbb{E}_{\\mathcal{D}}[\\hat{q}(x_t,\\pi_e) +  w(x_t,a_t) (r_t - \\hat{q}(x_t,a_t))],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    When the weight-clipping is applied, a large importance weight is clipped as :math:`\\hat{w}(x,a) := \\min \\{ \\lambda, w(x,a) \\}`
    where :math:`\\lambda (>0)` is a hyperparameter that decides a maximum allowed importance weight.

    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.

    DR mimics IPW to use a weighted version of rewards, but DR also uses the estimated mean reward
    function (the regression model) as a control variate to decrease the variance.
    It preserves the consistency of IPW if either the importance weight or
    the mean reward estimator is accurate (a property called double robustness).
    Moreover, DR is semiparametric efficient when the mean reward estimator is correctly specified.

    Parameters
    ----------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.
        DoublyRobust with a finite positive `lambda_` corresponds to the Doubly Robust with pessimistic shrinkage of Su et al.(2020).

    estimator_name: str, default='dr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh.
    "More Robust Doubly Robust Off-policy Evaluation.", 2018.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudík.
    "Doubly robust off-policy evaluation with shrinkage.", 2020.

    """

    lambda_: float = np.inf
    estimator_name: str = "dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("lambda_ must not be nan")

    def _estimate_round_rewards(
        self,
        reward: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        pscore: Union[np.ndarray, torch.Tensor],
        action_dist: Union[np.ndarray, torch.Tensor],
        estimated_rewards_by_reg_model: Union[np.ndarray, torch.Tensor],
        behavior_action_dist: Optional[Union[np.ndarray, torch.Tensor]] = None,
        position: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model or Tensor: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards of each round estimated by the DR estimator.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]

        if isinstance(reward, np.ndarray):
            estimated_rewards = np.average(
                q_hat_at_position,
                weights=pi_e_at_position,
                axis=1,
            )
        elif isinstance(reward, torch.Tensor):
            estimated_rewards = torch.sum(q_hat_at_position * pi_e_at_position, dim=1)
        else:
            raise ValueError("reward must be ndarray or Tensor")

        estimated_rewards += iw * (reward - q_hat_factual)
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        behavior_action_dist: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Policy value estimated by the DR estimator.

        """
        if not isinstance(estimated_rewards_by_reg_model, np.ndarray):
            raise ValueError("estimated_rewards_by_reg_model must be ndarray")
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            behavior_action_dist=behavior_action_dist,
            position=position,
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            context=context,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            behavior_action_dist=behavior_action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        ).mean()

    def estimate_policy_value_tensor(
        self,
        reward: torch.Tensor,
        action: torch.Tensor,
        pscore: torch.Tensor,
        action_dist: torch.Tensor,
        estimated_rewards_by_reg_model: torch.Tensor,
        behavior_action_dist: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Estimate the policy value of evaluation policy and return PyTorch Tensor.
        This is intended for being used with NNPolicyLearner.

        Parameters
        ----------
        reward: Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: Tensor, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: Tensor, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: Tensor, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        V_hat: Tensor
            Policy value estimated by the DR estimator.

        """
        if not isinstance(estimated_rewards_by_reg_model, torch.Tensor):
            raise ValueError("estimated_rewards_by_reg_model must be Tensor")
        if not isinstance(reward, torch.Tensor):
            raise ValueError("reward must be Tensor")
        if not isinstance(action, torch.Tensor):
            raise ValueError("action must be Tensor")
        if not isinstance(pscore, torch.Tensor):
            raise ValueError("pscore must be Tensor")

        check_ope_inputs_tensor(
            action_dist=action_dist,
            behavior_action_dist=behavior_action_dist,
            position=position,
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            context=context,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            behavior_action_dist=behavior_action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        if not isinstance(estimated_rewards_by_reg_model, np.ndarray):
            raise ValueError("estimated_rewards_by_reg_model must be ndarray")
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the MSE score of a given clipping hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given clipping hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n_rounds = reward.shape[0]
        # estimate the sample variance of DR with clipping
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )
        )
        sample_variance /= n_rounds

        # estimate the (high probability) upper bound of the bias of DR with clipping
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        bias_upper_bound = estimate_high_probability_upper_bound_bias(
            reward=reward,
            iw=iw,
            iw_hat=np.minimum(iw, self.lambda_),
            q_hat=estimated_rewards_by_reg_model[np.arange(n_rounds), action, position],
        )
        estimated_mse_score = sample_variance + (bias_upper_bound ** 2)

        return estimated_mse_score


@dataclass
class SelfNormalizedDoublyRobust(DoublyRobust):
    """Self-Normalized Doubly Robust (SNDR) Estimator.

    Note
    -------
    Self-Normalized Doubly Robust estimates the policy value of evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{SNDR}} (\\pi_e; \\mathcal{D}, \\hat{q}) :=
        \\mathbb{E}_{\\mathcal{D}} \\left[\\hat{q}(x_t,\\pi_e) +  \\frac{w(x_t,a_t) (r_t - \\hat{q}(x_t,a_t))}{\\mathbb{E}_{\\mathcal{D}}[ w(x_t,a_t) ]} \\right],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    Similar to Self-Normalized Inverse Probability Weighting, SNDR estimator applies the self-normalized importance weighting technique to
    increase the stability of the original Doubly Robust estimator.

    Parameters
    ----------
    estimator_name: str, default='sndr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "sndr"

    def _estimate_round_rewards(
        self,
        reward: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        pscore: Union[np.ndarray, torch.Tensor],
        action_dist: Union[np.ndarray, torch.Tensor],
        estimated_rewards_by_reg_model: Union[np.ndarray, torch.Tensor],
        behavior_action_dist: Optional[Union[np.ndarray, torch.Tensor]] = None,
        position: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards of each round estimated by the SNDR estimator.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]

        if isinstance(reward, np.ndarray):
            estimated_rewards = np.average(
                q_hat_at_position,
                weights=pi_e_at_position,
                axis=1,
            )
        elif isinstance(reward, torch.Tensor):
            estimated_rewards = torch.sum(q_hat_at_position * pi_e_at_position, dim=1)
        else:
            raise ValueError("reward must be ndarray or Tensor")

        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        estimated_rewards += iw * (reward - q_hat_factual) / iw.mean()
        return estimated_rewards


@dataclass
class SwitchDoublyRobust(DoublyRobust):
    """Switch Doubly Robust (Switch-DR) Estimator.

    Note
    -------
    Switch-DR aims to reduce the variance of the DR estimator by using direct method when the importance weight is large.
    This estimator estimates the policy value of evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{SwitchDR}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\tau)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w(x_t,a_t) (r_t - \\hat{q}(x_t,a_t)) \\mathbb{I} \\{ w(x_t,a_t) \\le \\tau \\}],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\tau (\\ge 0)` is a switching hyperparameter, which decides the threshold for the importance weight.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    Parameters
    ----------
    tau: float, default=np.inf
        Switching hyperparameter. When importance weight is larger than this parameter, DM is applied, otherwise DR is used.
        This hyperparameter should be larger than or equal to 0., otherwise it is meaningless.

    estimator_name: str, default='switch-dr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudík.
    "Optimal and Adaptive Off-policy Evaluation in Contextual Bandits", 2016.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    tau: float = np.inf
    estimator_name: str = "switch-dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.tau,
            name="tau",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.tau != self.tau:
            raise ValueError("tau must not be nan")

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        behavior_action_dist: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards of each round estimated by the Switch-DR estimator.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        # import pdb; pdb.set_trace()
        switch_indicator = np.array(iw <= self.tau, dtype=int)
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += switch_indicator * iw * (reward - q_hat_factual)
        return estimated_rewards

    def estimate_policy_value_tensor(
        self,
        **kwargs,
    ) -> torch.Tensor:
        """
        Estimate the policy value of evaluation policy and return PyTorch Tensor.
        This is intended for being used with NNPolicyLearner.
        This is not implemented because switching is indifferentiable.
        """
        raise NotImplementedError(
            "This is not implemented for Switch-DR because it is indifferentiable."
        )

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        behavior_action_dist: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the MSE score of a given switching hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given switching hyperparameter `tau`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n_rounds = reward.shape[0]
        # estimate the sample variance of Switch-DR (Eq.(8) of Wang et al.(2017))
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )
        )
        sample_variance /= n_rounds

        # estimate the (high probability) upper bound of the bias of Switch-DR
        # iw = action_dist[np.arange(n_rounds), action, position] / pscore
        # ## From 20-Su+
        # bias_upper_bound = estimate_high_probability_upper_bound_bias(
        #     reward=reward,
        #     iw=iw,
        #     iw_hat=iw * np.array(iw <= self.tau, dtype=int),
        #     q_hat=estimated_rewards_by_reg_model[np.arange(n_rounds), action, position],
        # )
        ### From 17-Wang+
        # import pdb; pdb.set_trace()
        iw = action_dist[np.arange(n_rounds), :, position] / behavior_action_dist
        switch_indicator = np.array(iw > self.tau, dtype=int)
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        bias_upper_bound = np.average(
            switch_indicator,
            weights=pi_e_at_position,
            axis=1,
        )
        bias_upper_bound = np.average(bias_upper_bound)

        estimated_mse_score = sample_variance + (bias_upper_bound ** 2)

        return estimated_mse_score


@dataclass
class dribt(DoublyRobust):
    """ {D}oubly {R}obust with {I}nformation borrowing 
                                    and {C}ontext-based switching (DR-IC) estimator

    Parameters
    ----------
    tau: float, default=np.inf
        Switching hyperparameter. 

    sigma: float, default=1.0
        Bandwidth hyperparameter. 

    estimator_name: str, default='dr-ibt'.
        Name of the estimator.

    References
    ----------
    THIS WORK. DR-IBT estimator == DR-IC estimator

    """

    tau: float = np.inf
    sigma: float = 1.0
    estimator_name: str = "dr-ibt"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.tau,
            name="tau",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.tau != self.tau:
            raise ValueError("tau must not be nan")
        check_scalar(
            self.sigma,
            name="sigma",
            target_type=(int, float),
            min_val=0.01,
        )
        if self.sigma != self.sigma:
            raise ValueError("sigma must not be nan")

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        context: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        behavior_action_dist: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards of each round estimated by the Switch-DR estimator.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        # estimate r_{IB} 
        r_ib = q_hat_at_position
        # adding cross product terms
        r_estimate_diff = reward - r_ib[np.arange(n_rounds), action]
        # Sigma_r_inverse = np.diagflat(np.repeat(1/np.var(r_estimate_diff), n_rounds ))
        Sigma_r_inverse = np.diagflat(np.repeat(1.0, n_rounds ))
        Sigma_r_truereward = self._cor_r_truereward(context, action,
                                    pi_e_at_position, behavior_action_dist)
        # @ is used for matrix multiplication
        cross_term = Sigma_r_truereward @ Sigma_r_inverse @ r_estimate_diff
        r_ib += np.reshape(cross_term, r_ib.shape) # n_rounds x num_actions
        # clipping r_ib in [0,1]
        r_ib[r_ib<0] = 0
        r_ib[r_ib>1] = 1

        # estimate r_{IB} for first term in the estimate
        r_ib_factual = r_ib[np.arange(n_rounds), action]

        # KL based switch indictor: size of (n_rounds, 1)
        kl = self._kl_contexts(pi_e_at_position, behavior_action_dist)
        switch_indicator = np.array(kl < self.tau, dtype=int)

        # r_aug: augmented r_ib
        # added indicator to switch between r_ib and direct rewards
        r_aug = r_ib
        r_aug[np.arange(n_rounds), action] = switch_indicator * r_aug[np.arange(n_rounds), action] \
                                            + (1.0-switch_indicator) * reward

        # estimator value term 2 and term 3 combined
        estimated_rewards = np.average(
            r_aug,
            weights=pi_e_at_position,
            axis=1,
        )
        # estimator value term 1
        estimated_rewards += switch_indicator * iw * (reward - r_ib_factual)
        return estimated_rewards

    def _kl_contexts(
        self,
        eval_policy: np.ndarray,
        behavior_policy: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        # import pdb; pdb.set_trace()
        kl = np.average(
            np.log(eval_policy / behavior_policy),
            weights=eval_policy,
            axis=1,
        )
        return kl
    
    def _cor_r_truereward(
        self,
        context: np.ndarray,
        action: np.ndarray,
        eval_policy: np.ndarray,
        behavior_policy: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        # import pdb; pdb.set_trace()
        n_data = eval_policy.shape[0]
        n_actions = eval_policy.shape[1]

        cor = np.zeros((n_data*n_actions, n_data))
        for j in range(n_data):
            row_idx = np.arange(action[j], n_data*n_actions, n_actions, dtype=int)
            weights_inv = behavior_policy[:,action[j]] / eval_policy[:,action[j]]
            weights_inv *= behavior_policy[j,action[j]] / eval_policy[j,action[j]]
            weights_inv *= 1 / (2 * self.sigma**2)
            cor[row_idx, j] = (np.sqrt(weights_inv)/np.sqrt(np.pi)) * \
                                np.exp( -1.0 * (np.linalg.norm(context-context[j]))**2 * weights_inv )

        # row normalization
        row_sum = np.sum(cor, axis = 1, keepdims = True)
        row_sum[row_sum==0] = 1.0
        cor /= np.repeat(row_sum, n_data, axis=1)
        cor *= 0.99999
        
        return cor

    def estimate_policy_value_tensor(
        self,
        **kwargs,
    ) -> torch.Tensor:
        """
        Estimate the policy value of evaluation policy and return PyTorch Tensor.
        This is intended for being used with NNPolicyLearner.
        This is not implemented because switching is indifferentiable.
        """
        raise NotImplementedError(
            "This is not implemented for Switch-DR because it is indifferentiable."
        )

    # hyperparameter tuning with dr-ibt specific inputs like behavior_action_dist
    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        behavior_action_dist: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the MSE score of a given switching hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given switching hyperparameter `tau`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n_rounds = reward.shape[0]
        # estimate the sample variance of Switch-DR (Eq.(8) of Wang et al.(2017))
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
                behavior_action_dist=behavior_action_dist,
                context=context,
            )
        )
        sample_variance /= n_rounds

        # estimate the (high probability) upper bound of the bias of Switch-DR
        # iw = action_dist[np.arange(n_rounds), action, position] / pscore
        ### From 17-Wang+
        # import pdb; pdb.set_trace()
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        kl = self._kl_contexts(pi_e_at_position, behavior_action_dist)
        switch_indicator = np.array(kl > self.tau, dtype=int)
        bias_upper_bound = np.average(switch_indicator)
        estimated_mse_score = sample_variance + (bias_upper_bound ** 2)

        return estimated_mse_score



@dataclass
class DoublyRobustWithShrinkage(DoublyRobust):
    """Doubly Robust with optimistic shrinkage (DRos) Estimator.

    Note
    ------
    DR with (optimistic) shrinkage replaces the importance weight in the original DR estimator with a new weight mapping
    found by directly optimizing sharp bounds on the resulting MSE.

    .. math::

        \\hat{V}_{\\mathrm{DRos}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w_o(x_t,a_t;\\lambda) (r_t - \\hat{q}(x_t,a_t))],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    :math:`w_{o} (x_t,a_t;\\lambda)` is a new weight by the shrinkage technique which is defined as

    .. math::

        w_{o} (x_t,a_t;\\lambda) := \\frac{\\lambda}{w^2(x_t,a_t) + \\lambda} w(x_t,a_t).

    When :math:`\\lambda=0`, we have :math:`w_{o} (x,a;\\lambda)=0` corresponding to the DM estimator.
    In contrast, as :math:`\\lambda \\rightarrow \\infty`, :math:`w_{o} (x,a;\\lambda)` increases and in the limit becomes equal to
    the original importance weight, corresponding to the standard DR estimator.

    Parameters
    ----------
    lambda_: float
        Shrinkage hyperparameter.
        This hyperparameter should be larger than or equal to 0., otherwise it is meaningless.

    estimator_name: str, default='dr-os'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambda_: float = 0.0
    estimator_name: str = "dr-os"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("lambda_ must not be nan")

    def _estimate_round_rewards(
        self,
        reward: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        pscore: Union[np.ndarray, torch.Tensor],
        action_dist: Union[np.ndarray, torch.Tensor],
        estimated_rewards_by_reg_model: Union[np.ndarray, torch.Tensor],
        behavior_action_dist: Optional[Union[np.ndarray, torch.Tensor]] = None,
        position: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards of each round estimated by the DRos estimator.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        if self.lambda_ < np.inf:
            iw_hat = (self.lambda_ * iw) / (iw ** 2 + self.lambda_)
        else:
            iw_hat = iw
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]

        if isinstance(reward, np.ndarray):
            estimated_rewards = np.average(
                q_hat_at_position,
                weights=pi_e_at_position,
                axis=1,
            )
        elif isinstance(reward, torch.Tensor):
            estimated_rewards = torch.sum(q_hat_at_position * pi_e_at_position, dim=1)
        else:
            raise ValueError("reward must be ndarray or Tensor")

        estimated_rewards += iw_hat * (reward - q_hat_factual)
        return estimated_rewards

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the MSE score of a given shrinkage hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given shrinkage hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n_rounds = reward.shape[0]
        # estimate the sample variance of DRos
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )
        )
        sample_variance /= n_rounds

        # estimate the (high probability) upper bound of the bias of DRos
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        if self.lambda_ < np.inf:
            iw_hat = (self.lambda_ * iw) / (iw ** 2 + self.lambda_)
        else:
            iw_hat = iw
        bias_upper_bound = estimate_high_probability_upper_bound_bias(
            reward=reward,
            iw=iw,
            iw_hat=iw_hat,
            q_hat=estimated_rewards_by_reg_model[np.arange(n_rounds), action, position],
        )
        estimated_mse_score = sample_variance + (bias_upper_bound ** 2)

        return estimated_mse_score
