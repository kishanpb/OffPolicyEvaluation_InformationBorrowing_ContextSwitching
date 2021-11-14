import argparse
import yaml
from pathlib import Path

import numpy as np
from pandas import DataFrame, read_csv
from joblib import Parallel, delayed
# from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize

from obp.dataset import MultiClassToBanditReduction
from obp.ope import (
    RegressionModel,
    OffPolicyEvaluation,
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust,
    dribt,
    SelfNormalizedDoublyRobust,
    SwitchDoublyRobust,
    DoublyRobustWithShrinkage,
    SwitchDoublyRobustTuning,
    DoublyRobustWithShrinkageTuning,
    DRIBTTuning,
)

# hyperparameters of the regression model used in model dependent OPE estimators
with open("./examples/multiclass/conf/hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)

def load_uci_dataset(full_path, to_normalize=False):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    if to_normalize:
        X = normalize(X)
    # label encode the target variable to have the classes 0 --- n_classes-1
    y = LabelEncoder().fit_transform(y)
    return X, y


dataset_dict = dict(
    # breast_cancer=load_breast_cancer(return_X_y=True),
    # digits=load_digits(return_X_y=True),
    # iris=load_iris(return_X_y=True),
    # wine=load_wine(return_X_y=True),
    glass=load_uci_dataset("./../dataset/glass.csv"),
    yeast=load_uci_dataset("./../dataset/yeast.csv"),
    ecoli=load_uci_dataset("./../dataset/ecoli.csv"),
    optdigits=load_uci_dataset("./../dataset/optdigits.csv",to_normalize=True),
    wdbc=load_uci_dataset("./../dataset/wdbc.csv"),
    vehicle=load_uci_dataset("./../dataset/vehicle.csv",to_normalize=True),
    page_blocks=load_uci_dataset("./../dataset/page_blocks.csv",to_normalize=True),
    satimage=load_uci_dataset("./../dataset/satimage.csv"),
    pendigits=load_uci_dataset("./../dataset/pendigits.csv",to_normalize=True),
    letter=load_uci_dataset("./../dataset/letter.csv",to_normalize=True),
)

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=HistGradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

# compared OPE estimators


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate off-policy estimators with multi-class classification data."
    )
    parser.add_argument(
        "--n_runs", type=int, default=1, help="number of simulations in the experiment."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["glass", "yeast", "ecoli", "optdigits", "wdbc",
                "vehicle", "page_blocks", "letter", "satimage", "pendigits"],
        required=True,
        help="the name of the multi-class classification dataset.",
    )
    parser.add_argument(
        "--eval_size",
        type=float,
        default=0.7,
        help="the proportion of the dataset to include in the evaluation split.",
    )
    parser.add_argument(
        "--base_model_for_behavior_policy",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base ML model for behavior policy, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument(
        "--alpha_b",
        type=float,
        default=0.8,
        help="the ratio of a uniform random policy when constructing an behavior policy.",
    )
    parser.add_argument(
        "--beta_b",
        type=float,
        default=0.0,
        help="beta from 20-Su+ when constructing an behavior policy.",
    )
    parser.add_argument(
        "--base_model_for_evaluation_policy",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base ML model for evaluation policy, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument(
        "--alpha_e",
        type=float,
        default=0.9,
        help="the ratio of a uniform random policy when constructing an evaluation policy.",
    )
    parser.add_argument(
        "--base_model_for_reg_model",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base ML model for regression model, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="the maximum number of concurrently running jobs.",
    )
    parser.add_argument(
        "--stoc_reward",
        type=bool,
        default=False,
        help="Send 'True' for enabling Stochastic Rewards.",
    )
    parser.add_argument("--random_state", type=int, default=12345)
    parser.add_argument("--save_logs", type=bool, default=False)
    args = parser.parse_args()
    print(args)

    # configurations
    n_runs = args.n_runs
    dataset_name = args.dataset_name
    stoc_reward = args.stoc_reward
    eval_size = args.eval_size
    base_model_for_behavior_policy = args.base_model_for_behavior_policy
    alpha_b = args.alpha_b
    beta_b = args.beta_b
    base_model_for_evaluation_policy = args.base_model_for_evaluation_policy
    alpha_e = args.alpha_e
    base_model_for_reg_model = args.base_model_for_reg_model
    n_jobs = args.n_jobs
    random_state = args.random_state
    np.random.seed(random_state)

    # load raw data
    X, y = dataset_dict[dataset_name]
    # convert the raw classification data into a logged bandit dataset
    beta_b *= np.random.uniform(-0.5, 0.5)
    dataset = MultiClassToBanditReduction(
        X=X,
        y=y,
        base_classifier_b=base_model_dict[base_model_for_behavior_policy](
            **hyperparams[base_model_for_behavior_policy]
        ),
        alpha_b=alpha_b,
        beta_b=beta_b,
        dataset_name=dataset_name,
        stoc_reward=stoc_reward,
    )


    ## evaluation policy training
    # split the original data into training and evaluation sets
    dataset.split_train_eval(eval_size=eval_size, random_state=random_state)
    # obtain logged bandit feedback generated by behavior policy
    bandit_feedback, base_classifier_b, _ = dataset.obtain_batch_bandit_feedback(random_state=random_state)
    
    # obtain action choice probabilities and classifier of an evaluation policy
    action_dist, base_classifier_e = dataset.obtain_action_dist_by_eval_policy(
        base_classifier_e=base_model_dict[base_model_for_evaluation_policy](
            **hyperparams[base_model_for_evaluation_policy]
        ),
        alpha_e=alpha_e,
    )
    # calculate the ground-truth performance of the evaluation policy
    ground_truth_policy_value = dataset.calc_ground_truth_policy_value(
        action_dist=action_dist
    )
    print("=" * 45)
    print("Ground truth value is ", ground_truth_policy_value)

    ######## Find best sigma

    sigmas = np.geomspace(0.01,15.0,num=30)
    if dataset_name == "optdigits" or dataset_name == "satimage" or dataset_name == "letter" or\
        dataset_name == "pageblocks" or dataset_name == "pendigits":
        sigmas = np.geomspace(0.01,15.0,num=10)
    ope_estimators = []
    for sigma in sigmas:
        ope_estimators.append(dribt(sigma=sigma, 
                            estimator_name="dr-ibt (tau=inf,sigma="+str(sigma)+")"))
    def process(i: int, num_data):
        # from the original data generate validation sets: BOOTSTRAP SAMPLING
        # sklearn.train_test_split accepts range (0,1); hence -0.01 below
        validation_size = num_data / y.shape[0]
        if validation_size >= 1.0: validation_size-=0.01
        dataset.split_train_eval(eval_size=validation_size, random_state=i, 
                                validation_test_bool=True)
        # obtain logged bandit feedback generated by behavior policy for validation data
        validation_bandit_feedback, _, action_dist_validation_behavior = dataset.obtain_batch_bandit_feedback(random_state=i,
                                        base_classifier_b=base_classifier_b,
                                        validation_test_bool=True)
    
        # obtain action choice probabilities for validation data
        action_dist_validation, _ = dataset.obtain_action_dist_by_eval_policy(
                                        base_classifier_e=base_classifier_e,
                                        alpha_e=alpha_e, 
                                        validation_test_bool=True
                                    )

        # estimate the mean reward function of the evaluation set of multi-class classification data with ML model
        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
        )
        estimated_rewards_by_reg_model = regression_model.fit_predict(
            context=validation_bandit_feedback["context"],
            action=validation_bandit_feedback["action"],
            reward=validation_bandit_feedback["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )

        # MRDR model and estimate
        regression_model_mrdr = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
            fitting_method='mrdr',
        )
        estimated_rewards_by_reg_model_mrdr = regression_model_mrdr.fit_predict(
            action_dist=action_dist_validation,
            context=validation_bandit_feedback["context"],
            action=validation_bandit_feedback["action"],
            reward=validation_bandit_feedback["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )

        # evaluate estimators' performances using squared error
        ope = OffPolicyEvaluation(
            bandit_feedback=validation_bandit_feedback,
            ope_estimators=ope_estimators,
        )
        relative_ee_i = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            action_dist=action_dist_validation,
            behavior_action_dist=action_dist_validation_behavior,
            # estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            metric="se", # earlier, it was relative_ee
            estimated_rewards_by_reg_model={
                "mrdr": estimated_rewards_by_reg_model_mrdr,
                "common": estimated_rewards_by_reg_model,
            }
        )

        return relative_ee_i


    ind = 1
    num_data_list = [100]
    while True:
        if (2**ind)*100 > y.shape[0]: break
        num_data_list.append(int((2**ind)*100))
        ind+=1
    num_data_list.append(y.shape[0])
    # num_data_list = [100, 200, 500, 1000]
    if dataset_name == "optdigits":
        num_data_list = [200, 500, 2000, y.shape[0]]
        # num_data_list = [200, 500, 2000]
        # num_data_list = [y.shape[0]]
    if dataset_name == "page_blocks":
        num_data_list = [200, 500, 2000, 4000]
        # num_data_list = [200, 500, 2000]
        # num_data_list = [4000]
    if dataset_name == "satimage":
        num_data_list = [200, 500, 2000, 4000]
        # num_data_list = [200, 500, 2000]
        # num_data_list = [4000]
    if dataset_name == "pendigits":
        num_data_list = [200, 500, 2000, 4000]
        # num_data_list = [200, 500, 2000]
        # num_data_list = [4000]
    if dataset_name == "letter":
        num_data_list = [200, 500, 2000, 4000]
        # num_data_list = [200, 500, 2000]
        # num_data_list = [4000]

    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    print("Experiment date and time =", dt_string)

    best_sigma = []
    best_est = []
    for num_data in num_data_list:
        print("=" * 45)
        print(y.shape[0], num_data)

        processed = Parallel(
            n_jobs=n_jobs,
            verbose=0,
        )([delayed(process)(i, num_data) for i in np.arange(min(n_runs, 200))])
        
        relative_ee_dict = {est.estimator_name: dict() for est in ope_estimators}
        for i, relative_ee_i in enumerate(processed):
            for (
                estimator_name,
                relative_ee_,
            ) in relative_ee_i.items():
                relative_ee_dict[estimator_name][i] = relative_ee_
                # import pdb; pdb.set_trace()
        relative_ee_df = DataFrame(relative_ee_dict).describe().T.round(6)

        min_mse_idx = np.argmin(np.array(relative_ee_df["mean"]))
        # best_est.append(relative_ee_df.iloc[min_mse_idx].name)
        best_est=relative_ee_df.iloc[min_mse_idx].name
        # print("=" * 45)
        sigma = best_est[best_est.find('sigma=')+6: best_est.find(')')]
        best_sigma.append(float(sigma))

    print("*" * 45)
    print(f"n={num_data_list}")
    print(f"best sigma = {best_sigma}")
    print("*" * 45)

    ############ FIND ORACLE DRIBT and DRIBT-tuned here:
    def kl_contexts(
            eval_policy,
            behavior_policy
        ):
        # import pdb; pdb.set_trace()
        kl = np.average(
            np.log(eval_policy / behavior_policy),
            weights=eval_policy,
            axis=1,
        )
        return kl

    def process_final(i: int, num_data, sigma):
        # from the original data generate validation sets: BOOTSTRAP SAMPLING
        # sklearn.train_test_split accepts range (0,1); hence -0.01 below
        validation_size = num_data / y.shape[0]
        if validation_size >= 1.0: validation_size-=0.01
        dataset.split_train_eval(eval_size=validation_size, random_state=i, 
                                validation_test_bool=True)
        # obtain logged bandit feedback generated by behavior policy for validation data
        validation_bandit_feedback, _, action_dist_validation_behavior = dataset.obtain_batch_bandit_feedback(random_state=i,
                                        base_classifier_b=base_classifier_b,
                                        validation_test_bool=True)
    
        # obtain action choice probabilities for validation data
        action_dist_validation, _ = dataset.obtain_action_dist_by_eval_policy(
                                        base_classifier_e=base_classifier_e,
                                        alpha_e=alpha_e, 
                                        validation_test_bool=True
                                    )

        # candidate taus
        # taus = np.quantile(kl_contexts(action_dist_validation[:,:,0], action_dist_validation_behavior),
        #                         [0.01, 0.05, 0.1, .2, .3, .4, .5, .6, .7, .8, .9, 1])                      
        taus = np.quantile(kl_contexts(action_dist_validation[:,:,0], action_dist_validation_behavior),
                                np.geomspace(0.01,1.0,num=30))
        if dataset_name == "optdigits" or dataset_name == "satimage" or dataset_name == "letter" or\
                dataset_name == "page_blocks" or dataset_name == "pendigits":
            taus = np.quantile(kl_contexts(action_dist_validation[:,:,0], action_dist_validation_behavior),
                                np.geomspace(0.01,1.0,num=15))        
        taus = np.append(taus, 0.99*np.quantile(kl_contexts(action_dist_validation[:,:,0], action_dist_validation_behavior),
                                0))
        taus = np.append(taus, 1.01*np.quantile(kl_contexts(action_dist_validation[:,:,0], action_dist_validation_behavior),
                                1))
        ope_estimators = [DRIBTTuning(taus=list(taus), sigma=sigma, estimator_name="dr-ibt")]
        ope_estimators.append(dribt(tau=0, sigma=sigma, 
                                estimator_name="dr-ibt (tau=0)"))
        for tau in taus:
            ope_estimators.append(dribt(tau=tau, sigma=sigma, 
                                estimator_name="dr-ibt (tau="+str(tau)+",sigma="+str(sigma)+")"))
        # import pdb; pdb.set_trace()

        # estimate the mean reward function of the evaluation set of multi-class classification data with ML model

        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
        )
        estimated_rewards_by_reg_model = regression_model.fit_predict(
            context=validation_bandit_feedback["context"],
            action=validation_bandit_feedback["action"],
            reward=validation_bandit_feedback["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )

        # MRDR model and estimate
        regression_model_mrdr = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
            fitting_method='mrdr',
        )
        estimated_rewards_by_reg_model_mrdr = regression_model_mrdr.fit_predict(
            action_dist=action_dist_validation,
            context=validation_bandit_feedback["context"],
            action=validation_bandit_feedback["action"],
            reward=validation_bandit_feedback["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )
        # import pdb; pdb.set_trace()

        # evaluate estimators' performances using squared error
        ope = OffPolicyEvaluation(
            bandit_feedback=validation_bandit_feedback,
            ope_estimators=ope_estimators,
        )
        relative_ee_i = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            action_dist=action_dist_validation,
            behavior_action_dist=action_dist_validation_behavior,
            # estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            metric="se", # earlier, it was relative_ee
            estimated_rewards_by_reg_model={
                "mrdr": estimated_rewards_by_reg_model_mrdr,
                "common": estimated_rewards_by_reg_model,
            }
        )
        # import pdb; pdb.set_trace()

        return relative_ee_i


    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    print("Experiment date and time =", dt_string)

    oracle_name = 'dr-ibt (oracle)'
    for jj, num_data in enumerate(num_data_list):
        print("=" * 45)
        print(y.shape[0], num_data)
        # import pdb; pdb.set_trace()

        processed = Parallel(
            n_jobs=n_jobs,
            verbose=0,
        )([delayed(process_final)(i, num_data, best_sigma[jj]) for i in np.arange(n_runs)])

        relative_ee_dict = {oracle_name: dict(), 'dr-ibt': dict(), 'dr-ibt (tau=0)': dict()}
        for i, relative_ee_i in enumerate(processed):
            # import pdb; pdb.set_trace()
            # taus = relative_ee_i[1]
            # relative_ee_i = relative_ee_i[0]
            relative_ee_dict[oracle_name][i] = np.inf
            for (
                estimator_name,
                relative_ee_,
            ) in relative_ee_i.items():
                # import pdb; pdb.set_trace()
                if estimator_name == 'dr-ibt':
                    relative_ee_dict[estimator_name][i] = relative_ee_
                if estimator_name == 'dr-ibt (tau=0)':
                    relative_ee_dict[estimator_name][i] = relative_ee_
                if estimator_name != 'dr-ibt' and estimator_name != 'dr-ibt (tau=0)'\
                                and relative_ee_ <= relative_ee_dict[oracle_name][i]: 
                    relative_ee_dict[oracle_name][i] = relative_ee_
        relative_ee_df = DataFrame(relative_ee_dict).describe().T.round(6)

        # print("=" * 45)
        print(f"random_state={random_state}")
        print(relative_ee_df[["mean", "std"]])
        print("=" * 45)

        # import pdb; pdb.set_trace()

        if args.save_logs:
            # save results of the evaluation of off-policy estimators in './logs' directory.
            log_path = Path(f"./logs/{dataset_name}")
            log_path.mkdir(exist_ok=True, parents=True)
            csv_filename = 'se_num_data_'+str(num_data)+\
                            '_random_state_'+str(random_state)+\
                            '_stoc_reward_'+str(stoc_reward)+\
                            '.csv'
                            # '_alpha_b_'+str(alpha_b)+\
            relative_ee_df.to_csv(log_path / csv_filename)


    ############# Other estimators
    ope_estimators = [
                        DirectMethod(),
                        # InverseProbabilityWeighting(),
                        # SelfNormalizedInverseProbabilityWeighting(),
                        DoublyRobust(),
                        DoublyRobust(estimator_name="mrdr"), 
                        # SelfNormalizedDoublyRobust(),
                    ]
    def process_other_est(i: int, num_data):
        # from the original data generate validation sets: BOOTSTRAP SAMPLING
        # sklearn.train_test_split accepts range (0,1); hence -0.01 below
        validation_size = num_data / y.shape[0]
        if validation_size >= 1.0: validation_size-=0.01
        dataset.split_train_eval(eval_size=validation_size, random_state=i, 
                                validation_test_bool=True)
        # obtain logged bandit feedback generated by behavior policy for validation data
        validation_bandit_feedback, _, action_dist_validation_behavior = dataset.obtain_batch_bandit_feedback(random_state=i,
                                        base_classifier_b=base_classifier_b,
                                        validation_test_bool=True)
    
        # obtain action choice probabilities for validation data
        action_dist_validation, _ = dataset.obtain_action_dist_by_eval_policy(
                                        base_classifier_e=base_classifier_e,
                                        alpha_e=alpha_e, 
                                        validation_test_bool=True
                                    )

        # estimate the mean reward function of the evaluation set of multi-class classification data with ML model

        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
        )
        estimated_rewards_by_reg_model = regression_model.fit_predict(
            context=validation_bandit_feedback["context"],
            action=validation_bandit_feedback["action"],
            reward=validation_bandit_feedback["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )

        # MRDR model and estimate
        regression_model_mrdr = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
            fitting_method='mrdr',
        )
        estimated_rewards_by_reg_model_mrdr = regression_model_mrdr.fit_predict(
            action_dist=action_dist_validation,
            context=validation_bandit_feedback["context"],
            action=validation_bandit_feedback["action"],
            reward=validation_bandit_feedback["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )
        # import pdb; pdb.set_trace()

        # evaluate estimators' performances using squared error
        ope = OffPolicyEvaluation(
            bandit_feedback=validation_bandit_feedback,
            ope_estimators=ope_estimators,
        )
        relative_ee_i = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            action_dist=action_dist_validation,
            behavior_action_dist=action_dist_validation_behavior,
            # estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            metric="se", # earlier, it was relative_ee
            estimated_rewards_by_reg_model={
                "mrdr": estimated_rewards_by_reg_model_mrdr,
                "common": estimated_rewards_by_reg_model,
            }
        )

        return relative_ee_i

    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    print("Experiment date and time =", dt_string)

    for num_data in num_data_list:
        print("=" * 45)
        print(y.shape[0], num_data)

        
        if y.shape[0] == num_data:
            processed = Parallel(
                n_jobs=30,
                verbose=0,
            )([delayed(process_other_est)(i, num_data) for i in np.arange(n_runs)])
        else:
            processed = Parallel(
                n_jobs=n_jobs,
                verbose=0,
            )([delayed(process_other_est)(i, num_data) for i in np.arange(n_runs)])

        relative_ee_dict = {est.estimator_name: dict() for est in ope_estimators}
        for i, relative_ee_i in enumerate(processed):
            for (
                estimator_name,
                relative_ee_,
            ) in relative_ee_i.items():
                relative_ee_dict[estimator_name][i] = relative_ee_
        relative_ee_df = DataFrame(relative_ee_dict).describe().T.round(6)

        print("=" * 45)
        print(f"random_state={random_state}")
        print("-" * 45)
        print(relative_ee_df[["mean", "std"]])
        print("=" * 45)

        if args.save_logs:
            # save results of the evaluation of off-policy estimators in './logs' directory.
            log_path = Path(f"./logs/{dataset_name}")
            log_path.mkdir(exist_ok=True, parents=True)
            # csv_filename = 'se_num_data_list_'+str(num_data)+'_dt_'+dt_string+'.csv'
            # relative_ee_df.to_csv(log_path / csv_filename)
            csv_filename = 'se_num_data_'+str(num_data)+\
                            '_random_state_'+str(random_state)+\
                            '_stoc_reward_'+str(stoc_reward)+\
                            '.csv'
                            # '_alpha_b_'+str(alpha_b)+\
            relative_ee_df.to_csv(log_path / csv_filename, mode='a', header=False)

    ###############################            DRos
    ############ FIND ORACLE DRos and DRos-tuned here:

    def process_dros(i: int, num_data):
        # from the original data generate validation sets: BOOTSTRAP SAMPLING
        # sklearn.train_test_split accepts range (0,1); hence -0.01 below
        validation_size = num_data / y.shape[0]
        if validation_size >= 1.0: validation_size-=0.01
        dataset.split_train_eval(eval_size=validation_size, random_state=i, 
                                validation_test_bool=True)
        # obtain logged bandit feedback generated by behavior policy for validation data
        validation_bandit_feedback, _, action_dist_validation_behavior = dataset.obtain_batch_bandit_feedback(random_state=i,
                                        base_classifier_b=base_classifier_b,
                                        validation_test_bool=True)
    
        # obtain action choice probabilities for validation data
        action_dist_validation, _ = dataset.obtain_action_dist_by_eval_policy(
                                        base_classifier_e=base_classifier_e,
                                        alpha_e=alpha_e, 
                                        validation_test_bool=True
                                    )

        # candidate taus
        taus = np.quantile(action_dist_validation[:,validation_bandit_feedback["action"],0]/validation_bandit_feedback["pscore"],
                                [0.05, .95])
        if dataset_name == "optdigits" or dataset_name == "satimage" or dataset_name == "letter" or\
                    dataset_name == "page_blocks" or dataset_name == "pendigits":
            taus = np.geomspace(0.01*(taus[0]**2),100*(taus[1]**2),num=15)
        else:
            taus = np.geomspace(0.01*(taus[0]**2),100*(taus[1]**2),num=30)
        ope_estimators = [DoublyRobustWithShrinkageTuning(lambdas=list(taus), estimator_name="dr-os")]
        for tau in taus:
            ope_estimators.append(DoublyRobustWithShrinkage(lambda_=tau, estimator_name="dr-os (lambda="+str(tau)+")"))
        # import pdb; pdb.set_trace()

        # estimate the mean reward function of the evaluation set of multi-class classification data with ML model

        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
        )
        estimated_rewards_by_reg_model = regression_model.fit_predict(
            context=validation_bandit_feedback["context"],
            action=validation_bandit_feedback["action"],
            reward=validation_bandit_feedback["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )

        # MRDR model and estimate
        regression_model_mrdr = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
            fitting_method='mrdr',
        )
        estimated_rewards_by_reg_model_mrdr = regression_model_mrdr.fit_predict(
            action_dist=action_dist_validation,
            context=validation_bandit_feedback["context"],
            action=validation_bandit_feedback["action"],
            reward=validation_bandit_feedback["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )
        # import pdb; pdb.set_trace()

        # evaluate estimators' performances using squared error
        ope = OffPolicyEvaluation(
            bandit_feedback=validation_bandit_feedback,
            ope_estimators=ope_estimators,
        )
        relative_ee_i = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            action_dist=action_dist_validation,
            behavior_action_dist=action_dist_validation_behavior,
            # estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            metric="se", # earlier, it was relative_ee
            estimated_rewards_by_reg_model={
                "mrdr": estimated_rewards_by_reg_model_mrdr,
                "common": estimated_rewards_by_reg_model,
            }
        )
        # import pdb; pdb.set_trace()

        return relative_ee_i

    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    print("Experiment date and time =", dt_string)

    oracle_name = 'dr-os (oracle)'
    for jj, num_data in enumerate(num_data_list):
        print("=" * 45)
        print(y.shape[0], num_data)
        # import pdb; pdb.set_trace()

        
        processed = Parallel(
                n_jobs=n_jobs,
                verbose=0,
            )([delayed(process_dros)(i, num_data) for i in np.arange(n_runs)])
        

        relative_ee_dict = {oracle_name: dict(), 'dr-os': dict()}
        for i, relative_ee_i in enumerate(processed):
            # import pdb; pdb.set_trace()
            # taus = relative_ee_i[1]
            # relative_ee_i = relative_ee_i[0]
            relative_ee_dict[oracle_name][i] = np.inf
            for (
                estimator_name,
                relative_ee_,
            ) in relative_ee_i.items():
                # import pdb; pdb.set_trace()
                if estimator_name == 'dr-os':
                    relative_ee_dict[estimator_name][i] = relative_ee_
                if estimator_name != 'dr-os'\
                                and relative_ee_ <= relative_ee_dict[oracle_name][i]: 
                    relative_ee_dict[oracle_name][i] = relative_ee_
        relative_ee_df = DataFrame(relative_ee_dict).describe().T.round(6)

        # print("=" * 45)
        print(f"random_state={random_state}")
        print(relative_ee_df[["mean", "std"]])
        print("=" * 45)

        # import pdb; pdb.set_trace()

        if args.save_logs:
            # save results of the evaluation of off-policy estimators in './logs' directory.
            log_path = Path(f"./logs/{dataset_name}")
            log_path.mkdir(exist_ok=True, parents=True)
            csv_filename = 'se_num_data_'+str(num_data)+\
                            '_random_state_'+str(random_state)+\
                            '_stoc_reward_'+str(stoc_reward)+\
                            '.csv'
                            # '_alpha_b_'+str(alpha_b)+\
            relative_ee_df.to_csv(log_path / csv_filename, mode='a', header=False)

    ############ FIND ORACLE switchdr and switchdr-tuned here:
    def powspace(start, stop, power, num):
        start = np.power(start, 1/float(power))
        stop = np.power(stop, 1/float(power))
        return np.power( np.linspace(start, stop, num=num), power)

    def process_switchdr(i: int, num_data):
        # from the original data generate validation sets: BOOTSTRAP SAMPLING
        # sklearn.train_test_split accepts range (0,1); hence -0.01 below
        validation_size = num_data / y.shape[0]
        if validation_size >= 1.0: validation_size-=0.01
        dataset.split_train_eval(eval_size=validation_size, random_state=i, 
                                validation_test_bool=True)
        # obtain logged bandit feedback generated by behavior policy for validation data
        validation_bandit_feedback, _, action_dist_validation_behavior = dataset.obtain_batch_bandit_feedback(random_state=i,
                                        base_classifier_b=base_classifier_b,
                                        validation_test_bool=True)
    
        # obtain action choice probabilities for validation data
        action_dist_validation, _ = dataset.obtain_action_dist_by_eval_policy(
                                        base_classifier_e=base_classifier_e,
                                        alpha_e=alpha_e, 
                                        validation_test_bool=True
                                    )

        # candidate taus
        # taus = np.quantile(action_dist_validation[:,validation_bandit_feedback["action"],0]/validation_bandit_feedback["pscore"],
        #                         [0.05, .95])
        taus = np.quantile(action_dist_validation[:,validation_bandit_feedback["action"],0]/validation_bandit_feedback["pscore"],
                                [0, 1])
        # taus = np.geomspace(taus[0],taus[1],num=25)
        # print(taus)
        # taus = powspace(taus[1],taus[0],power=5,num=25)
        if dataset_name == "optdigits" or dataset_name == "satimage" or dataset_name == "letter" or\
                    dataset_name == "page_blocks" or dataset_name == "pendigits":
            taus = powspace(taus[1],taus[0],power=5,num=15)
        else:
            taus = powspace(taus[1],taus[0],power=5,num=25)
        ope_estimators = [SwitchDoublyRobustTuning(taus=list(taus), estimator_name="switch-dr")]
        for tau in taus:
            ope_estimators.append(SwitchDoublyRobust(tau=tau, estimator_name="switch-dr (tau="+str(tau)+")"))
        # import pdb; pdb.set_trace()

        # estimate the mean reward function of the evaluation set of multi-class classification data with ML model

        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
        )
        estimated_rewards_by_reg_model = regression_model.fit_predict(
            context=validation_bandit_feedback["context"],
            action=validation_bandit_feedback["action"],
            reward=validation_bandit_feedback["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )

        # MRDR model and estimate
        regression_model_mrdr = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
            fitting_method='mrdr',
        )
        estimated_rewards_by_reg_model_mrdr = regression_model_mrdr.fit_predict(
            action_dist=action_dist_validation,
            context=validation_bandit_feedback["context"],
            action=validation_bandit_feedback["action"],
            reward=validation_bandit_feedback["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )
        # import pdb; pdb.set_trace()

        # evaluate estimators' performances using squared error
        ope = OffPolicyEvaluation(
            bandit_feedback=validation_bandit_feedback,
            ope_estimators=ope_estimators,
        )
        relative_ee_i = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            action_dist=action_dist_validation,
            behavior_action_dist=action_dist_validation_behavior,
            # estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            metric="se", # earlier, it was relative_ee
            estimated_rewards_by_reg_model={
                "mrdr": estimated_rewards_by_reg_model_mrdr,
                "common": estimated_rewards_by_reg_model,
            }
        )
        # import pdb; pdb.set_trace()

        return relative_ee_i

    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    print("Experiment date and time =", dt_string)

    oracle_name = 'switch-dr (oracle)'
    for jj, num_data in enumerate(num_data_list):
        print("=" * 45)
        print(y.shape[0], num_data)
        processed = Parallel(
                n_jobs=n_jobs,
                verbose=0,
            )([delayed(process_switchdr)(i, num_data) for i in np.arange(n_runs)])
        

        relative_ee_dict = {oracle_name: dict(), 'switch-dr': dict()}
        for i, relative_ee_i in enumerate(processed):
            # import pdb; pdb.set_trace()
            # taus = relative_ee_i[1]
            # relative_ee_i = relative_ee_i[0]
            relative_ee_dict[oracle_name][i] = np.inf
            for (
                estimator_name,
                relative_ee_,
            ) in relative_ee_i.items():
                # import pdb; pdb.set_trace()
                if estimator_name == 'switch-dr':
                    relative_ee_dict[estimator_name][i] = relative_ee_
                if estimator_name != 'switch-dr'\
                                and relative_ee_ <= relative_ee_dict[oracle_name][i]: 
                    relative_ee_dict[oracle_name][i] = relative_ee_
        relative_ee_df = DataFrame(relative_ee_dict).describe().T.round(6)

        # print("=" * 45)
        print(f"random_state={random_state}")
        print(relative_ee_df[["mean", "std"]])
        print("=" * 45)

        # import pdb; pdb.set_trace()

        if args.save_logs:
            # save results of the evaluation of off-policy estimators in './logs' directory.
            log_path = Path(f"./logs/{dataset_name}")
            log_path.mkdir(exist_ok=True, parents=True)
            csv_filename = 'se_num_data_'+str(num_data)+\
                            '_random_state_'+str(random_state)+\
                            '_stoc_reward_'+str(stoc_reward)+\
                            '.csv'
                            # '_alpha_b_'+str(alpha_b)+\
            relative_ee_df.to_csv(log_path / csv_filename, mode='a', header=False)