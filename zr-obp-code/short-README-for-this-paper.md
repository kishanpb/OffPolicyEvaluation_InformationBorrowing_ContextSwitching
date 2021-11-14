# Quick installation:

Create a virtual environment. For example
```
conda create -n ope python=3.7
```

In the virtual environment (do `source activate ope`), install this package:
```
cd zr-obp-code
python setup.py install
```

# To execute one experiment with one seed:
```
python -m examples.multiclass.all_est_evaluate    --n_runs 500    --dataset_name letter  --eval_size 0.3    --base_model_for_behavior_policy logistic_regression    --base_model_for_evaluation_policy logistic_regression    --base_model_for_reg_model logistic_regression    --n_jobs -1    --save_logs True --stoc_reward True  --random_state 12345
```
where
- `$n_runs` specifies the number of simulation runs in the experiment to estimate standard deviations of the performance of OPE estimators.
- `$dataset_name` specifies the name of the multi-class classification dataset and should be one of the 10 UCI datasets.
- `$eval_size` specifies the proportion of the dataset to calculate the ground truth.
- `$base_model_for_behavior_policy` specifies the base ML model for defining behavior policy and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$base_model_for_evaluation_policy` specifies the base ML model for defining evaluation policy and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$base_model_for_reg_model` specifies the base ML model for defining regression model and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$n_jobs` is the maximum number of concurrently running jobs.
- `save_logs` is a flag to save the logs in the `logs/dataset-name` path.
- `stoc_reward` is a flag that enables stochastic reward when `True`.
- `random_state` is the seed value used for all randomness in the code (numpy.random, sklearn, etc).

# Large scale experiment:
Use the bash script `run.sh` to execute more seeds by changing the dataset names and seeds if necessary. 

## For more details, please go through the README.md provided by the OBP library.