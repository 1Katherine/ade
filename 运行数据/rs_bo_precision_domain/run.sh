#!/bin/bash
#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount==0)); then
echo "no arg(benchmark-size)";
exit;
fi


# $1 = wordcount-100G
echo "=============== start $1 ===============" >> direct_rs_precision_bayesian.log
echo $(date) >> direct_rs_precision_bayesian.log
echo "=============== start $1 ===============" >> direct_rs_precision_bayesian.log

startTime=$(date "+%m-%d-%H-%M")
mv /usr/local/home/yyq/bo/rs_bo_precision/config/$1 /usr/local/home/yyq/bo/rs_bo_precision/config/$1-$startTime
mv /usr/local/home/yyq/bo/rs_bo_precision/logs.json /usr/local/home/yyq/bo/rs_bo_precision/config/$1-$startTime
mv /usr/local/home/yyq/bo/rs_bo_precision/generationConf.csv /usr/local/home/yyq/bo/rs_bo_precision/config/$1-$startTime
mv /usr/local/home/yyq/bo/rs_bo_precision/rs_precision_target.png /usr/local/home/yyq/bo/rs_bo_precision/config/$1-$startTime

mkdir -p /usr/local/home/yyq/bo/rs_bo_precision/config/wordcount-100G
python3 RS_Precision_Bayesian_Optimization.py

finishTime=$(date "+%m-%d-%H-%M")
mv /usr/local/home/yyq/bo/rs_bo_precision/config/$1 /usr/local/home/yyq/bo/rs_bo_precision/config/$1-$finishTime
mv /usr/local/home/yyq/bo/rs_bo_precision/logs.json /usr/local/home/yyq/bo/rs_bo_precision/config/$1-$finishTime
mv /usr/local/home/yyq/bo/rs_bo_precision/generationConf.csv /usr/local/home/yyq/bo/rs_bo_precision/config/$1-$finishTime
mv /usr/local/home/yyq/bo/rs_bo_precision/rs_precision_target.png /usr/local/home/yyq/bo/rs_bo_precision/config/$1-$startTime

echo "=============== finish $1 ===============" >> direct_rs_precision_bayesian.log
echo $(date) >> direct_rs_precision_bayesian.log
echo "=============== finish $1 ===============" >> direct_rs_precision_bayesian.log
