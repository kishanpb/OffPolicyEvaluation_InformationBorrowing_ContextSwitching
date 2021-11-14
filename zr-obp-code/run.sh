#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
arr=(300 400 500 800 900 1000 1200 1500 1600 1800 1900 2100 2200 2300 2500 2600 2700 2800 3100 3300 3400 3600 3700 3800 3900 4000 4300 4400 4600 4700 4900)
for i in ${arr[@]}
  do
     echo "Running deterministic reward - $i seed "
     python -m examples.multiclass.all_est_evaluate    --n_runs 500    --dataset_name satimage   --eval_size 0.3    --base_model_for_behavior_policy logistic_regression    --base_model_for_evaluation_policy logistic_regression    --base_model_for_reg_model logistic_regression    --n_jobs -1    --save_logs True   --random_state $i 

     echo " Running stochastic reward - $i seed "
     python -m examples.multiclass.all_est_evaluate    --n_runs 500    --dataset_name satimage   --eval_size 0.3    --base_model_for_behavior_policy logistic_regression    --base_model_for_evaluation_policy logistic_regression    --base_model_for_reg_model logistic_regression    --n_jobs -1    --save_logs True  --stoc_reward True  --random_state $i 

 done