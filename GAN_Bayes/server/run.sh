#!/bin/bash
#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount==0)); then
echo "no arg(benchmark-size)";
exit;
fi


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
# 所有样本 ganrsGroup为一组rs和gan生成的样本，比如生成3个rs样本后生成3个gan样本，循环反复则ganrsGroup=6
python3 /usr/local/home/yyq/bo/ganrs_bo/ganrs_Bayesian_Optimization_server.py  --ganrsGroup=6 --sampleType=0
# 前两组样本（前ganrsGroup*2个样本）
#python3 /usr/local/home/yyq/bo/ganrs_bo/ganrs_Bayesian_Optimization_server.py  --ganrsGroup=6 --sampleType=1
# 间隔采样（一组样本选择一个rs生成的一个gan生成的）
#python3 /usr/local/home/yyq/bo/ganrs_bo/ganrs_Bayesian_Optimization_server.py  --ganrsGroup=6 --sampleType=2
# 选择执行时间最短的前几个样本作为初始样本
#python3 /usr/local/home/yyq/bo/ganrs_bo/ganrs_Bayesian_Optimization_server.py  --ganrsGroup=6 --sampleType=3

finishTime=$(date "+%m-%d-%H-%M")
mv /usr/local/home/yyq/bo/ganrs_bo/config/$1 /usr/local/home/yyq/bo/ganrs_bo/config/$1-$finishTime
mv /usr/local/home/yyq/bo/ganrs_bo/logs.json /usr/local/home/yyq/bo/ganrs_bo/config/$1-$finishTime
mv /usr/local/home/yyq/bo/ganrs_bo/generationConf.csv /usr/local/home/yyq/bo/ganrs_bo/config/$1-$finishTime
mv /usr/local/home/yyq/bo/ganrs_bo/ganrs_target.png /usr/local/home/yyq/bo/ganrs_bo/config/$1-$startTime

echo "=============== finish $1 ===============" >> /usr/local/home/yyq/bo/ganrs_bo/direct_ganrs_Bayesian.log
echo $(date) >> /usr/local/home/yyq/bo/ganrs_bo/direct_ganrs_Bayesian.log
echo "=============== finish $1 ===============" >> /usr/local/home/yyq/bo/ganrs_bo/direct_ganrs_Bayesian.log
