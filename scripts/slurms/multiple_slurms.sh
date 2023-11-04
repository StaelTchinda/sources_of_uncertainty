#!/bin/bash

# Define arrays of values for each parameter
data_values=("cifar10")
model_values=("resnet20" "resnet32" "resnet44")
bayesian_values=("mc_dropout")
stage_values=("val")

# Loop through each combination of parameter values and run the sbatch command
for data in "${data_values[@]}"
do
    for model in "${model_values[@]}"
    do
        for bayesian in "${bayesian_values[@]}"
        do
            for stage in "${stage_values[@]}"
            do
                sbatch scripts/slurms/bayesian_eval.sbatch --data "$data" --model "$model" --bayesian "$bayesian" --stage "$stage"
            done
        done
    done
done
