#!/bin/bash
#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount==0)); then
echo "no arg(benchmark-size)";
exit;
fi
all_initpoints=50
n_initsamples=6
n_interations=$(($all_initpoints - $n_initsamples))

path=$(pwd)
echo $path
# $1 = wordcount-100G
echo "=============== start $1 ===============" >> $path/direct_Bayesian.log
echo $(date) >> $path/direct_Bayesian.log
echo "=============== start $1 ===============" >> $path/direct_Bayesian.log

startTime=$(date "+%m-%d-%H-%M")
mv  $path/config/$1 $path/config/$1-$startTime
mv $path/logs.json $path/config/$1-$startTime
mv $path/generationConf.csv $path/config/$1-$startTime
mv $path/target.png $path/config/$1-$startTime

mkdir -p $path/config/$1

python3 $path/rs_direct_Bayesian_newei.py --benchmark=$1 --ninit=$n_initsamples --niters=$n_interations

finishTime=$(date "+%m-%d-%H-%M")
mv  $path/config/$1 $path/config/$1-$finishTime
mv $path/logs.json $path/config/$1-$finishTime
mv $path/generationConf.csv $path/config/$1-$finishTime
mv $path/target.png $path/config/$1-$finishTime

echo "=============== finish $1 ===============" >> $path/direct_Bayesian.log
echo $(date) >> $path/direct_Bayesian.log
echo "=============== finish $1 ===============" >> $path/direct_Bayesian.log
mv $path/direct_Bayesian.log $path/direct_Bayesian-$finishTime.log