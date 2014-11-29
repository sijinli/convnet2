#!/usr/bin/bash
exp_name=Embedding_ASM_act_12_exp_4_ACCV_fc_j0
# exp_name=ASM_act_12_exp_4
JT=t004
EP=200
BSIZE=1024
macid=13
# DP=croppeddhmlperelskeljt
DP=memfeat
# # For exp 4
# TrainRange=0-76047
# TestRange=76048-105367

# TrainRange=0-76047
# TestRange=0-76047case 

TrainRange=0-76047
TestRange=76048-105367


#################
run_mac=c8k${macid}

/home/grads/sijinli2/pkg/anaconda/bin/python  convnet.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_${exp_name} --save-path=/opt/visal/tmp/for_sijin/Data/saved/Test/${run_mac} --train-range=${TrainRange} --test-range=${TestRange} --layer-def=/home/grads/sijinli2/Projects/DHMLPE/doc/netdef2/dhmlpe-layer-def-${JT}.cfg --layer-params=/home/grads/sijinli2/Projects/DHMLPE/doc/netdef2/dhmlpe-layer-params-${JT}.cfg --data-provider=${DP} --test-freq=200 --epoch=${EP} --mini=128 --batch-size=${BSIZE} --gpu 0 --crop-border=-1

