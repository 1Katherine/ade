#!/bin/bash
#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount==0)); then
echo "no arg(benchmark-size)";
exit;
fi


# $1 = wordcount-100G
echo "=============== start $1 ===============" >> direct_wlhs_Bayesian.log
echo $(date) >> direct_wlhs_Bayesian.log
echo "=============== start $1 ===============" >> direct_wlhs_Bayesian.log

startTime=$(date "+%m-%d-%H-%M")
mv /usr/local/home/yyq/bo/wlhs_bo/config/$1 /usr/local/home/yyq/bo/wlhs_bo/config/$1-$startTime
mv /usr/local/home/yyq/bo/wlhs_bo/logs.json /usr/local/home/yyq/bo/wlhs_bo/config/$1-$startTime
mv /usr/local/home/yyq/bo/wlhs_bo/generationConf.csv /usr/local/home/yyq/bo/wlhs_bo/config/$1-$startTime
mv /usr/local/home/yyq/bo/wlhs_bo/wlhs_target* /usr/local/home/yyq/bo/wlhs_bo/config/$1-$startTime

mkdir -p /usr/local/home/yyq/bo/wlhs_bo/config/wordcount-100G
python3 wLHS_Bayesian_Optimization.py

finishTime=$(date "+%m-%d-%H-%M")
mv /usr/local/home/yyq/bo/wlhs_bo/config/$1 /usr/local/home/yyq/bo/wlhs_bo/config/$1-$finishTime
mv /usr/local/home/yyq/bo/wlhs_bo/logs.json /usr/local/home/yyq/bo/wlhs_bo/config/$1-$finishTime
mv /usr/local/home/yyq/bo/wlhs_bo/generationConf.csv /usr/local/home/yyq/bo/wlhs_bo/config/$1-$finishTime
mv /usr/local/home/yyq/bo/wlhs_bo/wlhs_target* /usr/local/home/yyq/bo/wlhs_bo/config/$1-$startTime

echo "=============== finish $1 ===============" >> direct_wlhs_Bayesian.log
echo $(date) >> direct_wlhs_Bayesian.log
echo "=============== finish $1 ===============" >> direct_wlhs_Bayesian.log
