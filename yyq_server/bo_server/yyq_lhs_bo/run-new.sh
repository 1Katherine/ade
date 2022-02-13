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
echo "=============== start $1 ===============" >> $path/direct_ganrs_Bayesian.log
echo $(date) >> $path/direct_ganrs_Bayesian.log
echo "=============== start $1 ===============" >> $path/direct_ganrs_Bayesian.log

startTime=$(date "+%m-%d-%H-%M")
mv  $path/config/$1 $path/config/$1-$startTime
mv $path/logs.json $path/config/$1-$startTime
mv $path/generationConf.csv $path/config/$1-$startTime
mv $path/ganrs_target.png $path/config/$1-$startTime

mkdir -p $path/config/wordcount-100G

python3 $path/ganrs_Bayesian_Optimization_server.py  --sampleType=$initType --ganrsGroup=$groupNumber --n=$firstn --niters=$interationsNumber  --initFile=$file

finishTime=$(date "+%m-%d-%H-%M")
mv  $path/config/$1 $path/config/$1-$finishTime
mv $path/logs.json $path/config/$1-$finishTime
mv $path/generationConf.csv $path/config/$1-$finishTime
mv $path/ganrs_target.png $path/config/$1-$finishTime

echo "=============== finish $1 ===============" >> $path/direct_ganrs_Bayesian.log
echo $(date) >> $path/direct_ganrs_Bayesian.log
echo "=============== finish $1 ===============" >> d$path/direct_ganrs_Bayesian.log
mv $path/direct_ganrs_Bayesian.log $path/direct_ganrs_Bayesian-$finishTime.log
