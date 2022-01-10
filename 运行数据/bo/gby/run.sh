#!/bin/bash
#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount==0)); then
echo "no arg(benchmark-size)";
exit;
fi



echo "=============== start $1 ===============" >> direct_Bayesian.log
echo $(date) >> direct_Bayesian.log
echo "=============== start $1 ===============" >> direct_Bayesian.log

startTime=$(date "+%m-%d-%H-%M")
mv /usr/local/home/gby/result/$1 /usr/local/home/gby/result/$1-$startTime
mv /usr/local/home/gby/logs.json /usr/local/home/gby/result/$1-$startTime
mv /usr/local/home/gby/generationConf.csv /usr/local/home/gby/result/$1-$startTime

mkdir -p /usr/local/home/gby/result/wordcount-100G
python3 direct_Bayesian.py

finishTime=$(date "+%m-%d-%H-%M")
mv /usr/local/home/gby/result/$1 /usr/local/home/gby/result/$1-$finishTime
mv /usr/local/home/gby/logs.json /usr/local/home/gby/result/$1-$finishTime
mv /usr/local/home/gby/generationConf.csv /usr/local/home/gby/result/$1-$finishTime

echo "=============== finish $1 ===============" >> direct_Bayesian.log
echo $(date) >> direct_Bayesian.log
echo "=============== finish $1 ===============" >> direct_Bayesian.log
