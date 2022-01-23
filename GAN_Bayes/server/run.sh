#!/bin/bash
#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount==0)); then
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

interationsNumber=20
groupNumber=4
# type = all / firsttwogroup / interval / best
initType=firsttwogroup
file=/usr/local/home/yyq/bo/ganrs_bo/wordcount-100G-GAN-30.csv


# $1 = wordcount-100G
echo "=============== start $1 ===============" >> /usr/local/home/yyq/bo/ganrs_bo/direct_ganrs_Bayesian.log
echo $(date) >> /usr/local/home/yyq/bo/ganrs_bo/direct_ganrs_Bayesian.log
echo "=============== start $1 ===============" >> /usr/local/home/yyq/bo/ganrs_bo/direct_ganrs_Bayesian.log

startTime=$(date "+%m-%d-%H-%M")
mv /usr/local/home/yyq/bo/ganrs_bo/config/$1 /usr/local/home/yyq/bo/ganrs_bo/config/$1-$startTime
mv /usr/local/home/yyq/bo/ganrs_bo/logs.json /usr/local/home/yyq/bo/ganrs_bo/config/$1-$startTime
mv /usr/local/home/yyq/bo/ganrs_bo/generationConf.csv /usr/local/home/yyq/bo/ganrs_bo/config/$1-$startTime
mv /usr/local/home/yyq/bo/ganrs_bo/ganrs_target.png /usr/local/home/yyq/bo/ganrs_bo/config/$1-$startTime

mkdir -p /usr/local/home/yyq/bo/ganrs_bo/config/wordcount-100G

python3 /usr/local/home/yyq/bo/ganrs_bo/ganrs_Bayesian_Optimization_server.py  --sampleType=$initType --ganrsGroup=$groupNumber --niters=$interationsNumber --initFile=$file

finishTime=$(date "+%m-%d-%H-%M")
mv /usr/local/home/yyq/bo/ganrs_bo/config/$1 /usr/local/home/yyq/bo/ganrs_bo/config/$1-$finishTime
mv /usr/local/home/yyq/bo/ganrs_bo/logs.json /usr/local/home/yyq/bo/ganrs_bo/config/$1-$finishTime
mv /usr/local/home/yyq/bo/ganrs_bo/generationConf.csv /usr/local/home/yyq/bo/ganrs_bo/config/$1-$finishTime
mv /usr/local/home/yyq/bo/ganrs_bo/ganrs_target.png /usr/local/home/yyq/bo/ganrs_bo/config/$1-$startTime

echo "=============== finish $1 ===============" >> /usr/local/home/yyq/bo/ganrs_bo/direct_ganrs_Bayesian.log
echo $(date) >> /usr/local/home/yyq/bo/ganrs_bo/direct_ganrs_Bayesian.log
echo "=============== finish $1 ===============" >> /usr/local/home/yyq/bo/ganrs_bo/direct_ganrs_Bayesian.log
