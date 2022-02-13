#!/bin/bash
#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount<1)); then
echo "no arg(benchmark-size)";
exit;
fi

all_initpoints=50
n_initsamples=6
n_interations=$(($all_initpoints - $n_initsamples))

path=$(pwd)
echo $path
# $1 = wordcount-100G
echo "=============== start $1 ===============" >> $path/direct_lhs_Bayesian_domain.log
echo $(date) >> $path/direct_lhs_Bayesian_domain.log
echo "=============== start $1 ===============" >> $path/direct_lhs_Bayesian_domain.log

startTime=$(date "+%m-%d-%H-%M")
mv  $path/config/$1 $path/config/$1-$startTime
mv $path/logs.json $path/config/$1-$startTime
mv $path/generationConf.csv $path/config/$1-$startTime
mv $path/target.png $path/config/$1-$startTime

mkdir -p $path/config/wordcount-100G

python3 $path/LHS_Bayesian_Optimization_domain.py --benchmark=$1 --ninit=$n_initsamples --ninit=$n_interations

finishTime=$(date "+%m-%d-%H-%M")
mv  $path/config/$1 $path/config/$1-$finishTime
mv $path/logs.json $path/config/$1-$finishTime
mv $path/generationConf.csv $path/config/$1-$finishTime
mv $path/target.png $path/config/$1-$finishTime

echo "=============== finish $1 ===============" >> $path/direct_lhs_Bayesian_domain.log
echo $(date) >> $path/direct_lhs_Bayesian_domain.log
echo "=============== finish $1 ===============" >> $path/direct_lhs_Bayesian_domain.log
mv $path/direct_lhs_Bayesian_domain.log $path/direct_lhs_Bayesian_domain-$finishTime.log
