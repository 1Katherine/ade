#!/bin/bash
#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount<1)); then
echo "no arg(benchmark-size)";
exit;
fi
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

all_initpoints=50
groupNumber=6
# type = all / firstngroup / interval / best
firstn=1
initType=firstngroup
file=/usr/local/home/yyq/bo/ganrs_bo/wordcount-100G-GAN-42.csv
if [ $initType == "firstngroup" ]
then
  interationsNumber=$(($all_initpoints - $firstn * $groupNumber))
else
  interationsNumber=20
fi

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
echo "=============== finish $1 ===============" >> $path/direct_ganrs_Bayesian.log
mv $path/direct_ganrs_Bayesian.log $path/direct_ganrs_Bayesian-$finishTime.log
