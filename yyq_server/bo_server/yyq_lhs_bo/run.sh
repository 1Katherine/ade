#!/bin/bash
#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount==0)); then
echo "no arg(benchmark-size)";
exit;
fi


# $1 = wordcount-100G
echo "=============== start $1 ===============" >> /usr/local/home/yyq/bo/lhs_bo/direct_lhs_Bayesian.log
echo $(date) >> /usr/local/home/yyq/bo/lhs_bo/direct_lhs_Bayesian.log
echo "=============== start $1 ===============" >> /usr/local/home/yyq/bo/lhs_bo/direct_lhs_Bayesian.log

startTime=$(date "+%m-%d-%H-%M")
mv /usr/local/home/yyq/bo/lhs_bo/config/$1 /usr/local/home/yyq/bo/lhs_bo/config/$1-$startTime
mv /usr/local/home/yyq/bo/lhs_bo/logs.json /usr/local/home/yyq/bo/lhs_bo/config/$1-$startTime
mv /usr/local/home/yyq/bo/lhs_bo/generationConf.csv /usr/local/home/yyq/bo/lhs_bo/config/$1-$startTime
mv /usr/local/home/yyq/bo/lhs_bo/lhs_target.png /usr/local/home/yyq/bo/lhs_bo/config/$1-$startTime

mkdir -p /usr/local/home/yyq/bo/lhs_bo/config/$1
python3 /usr/local/home/yyq/bo/lhs_bo/LHS_Bayesian_Optimization.py -b=$1

finishTime=$(date "+%m-%d-%H-%M")
mv /usr/local/home/yyq/bo/lhs_bo/config/$1 /usr/local/home/yyq/bo/lhs_bo/config/$1-$finishTime
mv /usr/local/home/yyq/bo/lhs_bo/logs.json /usr/local/home/yyq/bo/lhs_bo/config/$1-$finishTime
mv /usr/local/home/yyq/bo/lhs_bo/generationConf.csv /usr/local/home/yyq/bo/lhs_bo/config/$1-$finishTime
mv /usr/local/home/yyq/bo/lhs_bo/lhs_target.png /usr/local/home/yyq/bo/lhs_bo/config/$1-$finishTime

echo "=============== finish $1 ===============" >> /usr/local/home/yyq/bo/lhs_bo/direct_lhs_Bayesian.log
echo $(date) >> /usr/local/home/yyq/bo/lhs_bo/direct_lhs_Bayesian.log
echo "=============== finish $1 ===============" >> /usr/local/home/yyq/bo/lhs_bo/direct_lhs_Bayesian.log
