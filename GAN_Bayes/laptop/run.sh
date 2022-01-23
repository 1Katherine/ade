#!/bin/bash
# --helo
if [[ $1 = "--help" ]] || [[ $1 = "-h" ]]
then
    echo "'--sampleType', type=str, default = all,help=
    all: for all samole,
    firsttwogroup: The first two groups of random samples and GAN samples are used as initial samples,
    interval: interval sampling,
    best: 10 samples with the least execution time"
    echo -e "\n"
    echo "'--ganrsGroup', type=int, default = 0, help=A set of random samples and the number of GAN samples.
    For example, two random samples are followed by two GAN samples, so ganrsGroup is equal to 4"
    echo -e "\n"
    echo "'--niters', type=int, default = 15, help=The number of iterations of the Bayesian optimization algorithm"
    echo -e "\n"
fi

interationsNumber=20
groupNumber=4
# type = all / firsttwogroup / interval / best
initType=best

echo $initType
echo $interationsNumber
python ganrs_Bayesian_Optimization_regitser.py --sampleType=$initType --ganrsGroup=$groupNumber --niters=$interationsNumber
