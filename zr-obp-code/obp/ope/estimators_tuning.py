# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Estimators with built-in hyperparameter tuning."""
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import numpy as np
from sklearn.utils import check_scalar

from .estimators import (
    BaseOffPolicyEstimator,
    InverseProbabilityWeighting,
    DoublyRobust,
    SwitchDoublyRobust,
    DoublyRobustWithShrinkage,
    dribt,
)
from ..utils import check_ope_inputs


@dataclass
class BaseOffPolicyEstimatorTuning:
    """Base Class for Off-Policy Estimator with built-in hyperparameter tuning

    base_ope_estimator: BaseOffPolicyEstimator
        An OPE estimator with a hyperparameter
        (such as IPW/DR with clipping, Switch-DR, and DR with Shrinkage).

    candidate_hyperparameter_list: List[float]
        A list of candidate hyperparameter values.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    base_ope_estimator: BaseOffPolicyEstimator = field(init=False)
    candidate_hyperparameter_list: List[float] = field(init=False)

    def __new__(cls, *args, **kwargs):
        dataclass(cls)
        return super().__new__(cls)

    def _check_candidate_hyperparameter_list(self, hyperparam_name: str) -> None:
        """Check type and value of candidate_hyperparameter_list."""
        if isinstance(self.candidate_hyperparameter_list, list):
            if len(self.candidate_hyperparameter_list) == 0:
                raise ValueError(f"{hyperparam_name} must not be empty")
            for hyperparam_ in self.candidate_hyperparameter_list:
                check_scalar(
                    hyperparam_,
                    name=f"an element of {hyperparam_name}",
                    target_type=(int, float),
                    min_val=0.0,
                )
                if hyperparam_ != hyperparam_:
                    raise ValueError(f"an element of {hyperparam_name} must not be nan")
        else:
            raise TypeError(f"{hyperparam_name} must be a list")

    def _tune_hyperparam(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        behavior_action_dist: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ) -> None:
        """Find the best hyperparameter value from the given candidate set."""
        self.estimated_mse_score_dict = dict()
        for hyperparam_ in self.candidate_hyperparameter_list:
            # import pdb; pdb.set_trace()
            if self.base_ope_estimator == dribt:
                estimated_mse_score = self.base_ope_estimator(
                    tau=hyperparam_,
                    sigma=self.sigma,
                )._estimate_mse_score(
                    reward=reward,
                    action=action,
                    pscore=pscore,
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                    position=position,
                    behavior_action_dist=behavior_action_dist,
                    context=context,
                )
            elif self.base_ope_estimator == SwitchDoublyRobust:
                estimated_mse_score = self.base_ope_estimator(
                    tau=hyperparam_,
                )._estimate_mse_score(
                    reward=reward,
                    action=action,
                    pscore=pscore,
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                    position=position,
                    behavior_action_dist=behavior_action_dist,
                    # context=context,
                )
            else:
                estimated_mse_score = self.base_ope_estimator(
                    hyperparam_
                )._estimate_mse_score(
                    reward=reward,
                    action=action,
                    pscore=pscore,
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                    position=position,
                )

            self.estimated_mse_score_dict[hyperparam_] = estimated_mse_score
        self.best_hyperparam = min(
            self.estimated_mse_score_dict.items(), key=lambda x: x[1]
        )[0]

    def estimate_policy_value_with_tuning(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        behavior_action_dist: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

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

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.

        Returns
        ----------
        V_hat: float
            Policy value estimated by the DR estimator.

        """
        # tune hyperparameter if necessary
        if not hasattr(self, "best_hyperparam"):
            self._tune_hyperparam(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
                behavior_action_dist=behavior_action_dist,
                context=context,
            )

        return self.base_ope_estimator(self.best_hyperparam).estimate_policy_value(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            behavior_action_dist=behavior_action_dist,
            context=context,
        )

    def estimate_interval_with_tuning(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
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

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.

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
        # tune hyperparameter if necessary
        if not hasattr(self, "best_hyperparam"):
            self._tune_hyperparam(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )

        return self.base_ope_estimator(self.best_hyperparam).estimate_interval(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


class InverseProbabilityWeightingTuning(BaseOffPolicyEstimatorTuning):
    """Inverse Probability Weighting (IPW) with built-in hyperparameter tuning.

    Parameters
    ----------
    lambdas: List[float]
        A list of candidate clipping hyperparameters.
        The automatic hyperparameter tuning proposed by Su et al.(2020)
        will choose the best hyperparameter value from the data.

    estimator_name: str, default='ipw'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambdas: List[float] = None
    estimator_name: str = "ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.base_ope_estimator = InverseProbabilityWeighting
        self.candidate_hyperparameter_list = self.lambdas
        super()._check_candidate_hyperparameter_list(hyperparam_name="lambdas")

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
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

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
        )

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
            Action choice probabilities
            by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.

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

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DoublyRobustTuning(BaseOffPolicyEstimatorTuning):
    """Doubly Robust (DR) with built-in hyperparameter tuning.

    Parameters
    ----------
    lambdas: List[float]
        A list of candidate clipping hyperparameters.
        The automatic hyperparameter tuning proposed by Su et al.(2020)
        will choose the best hyperparameter value from the data.

    estimator_name: str, default='dr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambdas: List[float] = None
    estimator_name: str = "dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.base_ope_estimator = DoublyRobust
        self.candidate_hyperparameter_list = self.lambdas
        super()._check_candidate_hyperparameter_list(hyperparam_name="lambdas")

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

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
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

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

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SwitchDoublyRobustTuning(BaseOffPolicyEstimatorTuning):
    """Switch Doubly Robust (Switch-DR) with build-in hyperparameter tuning.

    Parameters
    ----------
    taus: List[float]
        A list of candidate switching hyperparameters.
        The automatic hyperparameter tuning proposed by Su et al.(2020)
        will choose the best hyperparameter value from the data.

    estimator_name: str, default='switch-dr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudík.
    "Optimal and Adaptive Off-policy Evaluation in Contextual Bandits", 2016.

    """

    taus: List[float] = None
    estimator_name: str = "switch-dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.base_ope_estimator = SwitchDoublyRobust
        self.candidate_hyperparameter_list = self.taus
        super()._check_candidate_hyperparameter_list(hyperparam_name="taus")

    def estimate_policy_value(
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
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

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
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            behavior_action_dist=behavior_action_dist,
            context=context,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            behavior_action_dist=behavior_action_dist,
            context=context,
        )

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

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

@dataclass
class DRIBTTuning(BaseOffPolicyEstimatorTuning):
    """DR-IBT with build-in hyperparameter tuning.

    Parameters
    ----------
    taus: List[float]
        A list of candidate switching hyperparameters.
        The automatic hyperparameter tuning proposed by Su et al.(2020)
        will choose the best hyperparameter value from the data.

    estimator_name: str, default='switch-dr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudík.
    "Optimal and Adaptive Off-policy Evaluation in Contextual Bandits", 2016.

    """

    taus: List[float] = None
    estimator_name: str = "dr-ibt (tuned)"
    sigma: float = 1.0

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.base_ope_estimator = dribt
        self.candidate_hyperparameter_list = self.taus
        super()._check_candidate_hyperparameter_list(hyperparam_name="taus")

    def estimate_policy_value(
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
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

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
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            behavior_action_dist=behavior_action_dist,
            context=context,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            behavior_action_dist=behavior_action_dist,
            context=context,
        )

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

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )




@dataclass
class DoublyRobustWithShrinkageTuning(BaseOffPolicyEstimatorTuning):
    """Doubly Robust with optimistic shrinkage (DRos) with built-in hyperparameter tuning.

    Parameters
    ----------
    lambdas: List[float]
        A list of candidate shrinkage hyperparameters.
        The automatic hyperparameter tuning proposed by Su et al.(2020)
        will choose the best hyperparameter value from the data.

    estimator_name: str, default='dr-os'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambdas: List[float] = None
    estimator_name: str = "dr-os"

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.base_ope_estimator = DoublyRobustWithShrinkage
        self.candidate_hyperparameter_list = self.lambdas
        super()._check_candidate_hyperparameter_list(hyperparam_name="lambdas")

    def estimate_policy_value(
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
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

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
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            behavior_action_dist=behavior_action_dist,
            context=context,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            behavior_action_dist=behavior_action_dist,
            context=context,
        )

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

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
