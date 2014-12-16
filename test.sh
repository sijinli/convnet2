#!/usr/bin/bash
#exp_name=Embedding_ASM_act_14_exp_2_ACCV_fc_j0
exp_name=SP_t004_act_12_p1
JT=t012
EP=2000
BSIZE=1024
macid=13
DP=spsimple
# DP=memfeat
# # For exp 4
TrainRange=0-76047
TestRange=76048-105367

# TrainRange=0-76047
# TestRange=0-76047case 

# TrainRange=0-76047
# TestRange=76048-105367

# TrainRange=0-132743
# TestRange=132744-162007


#################
run_mac=c8k${macid}

/home/grads/sijinli2/pkg/anaconda/bin/python  convnet.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_${exp_name} --save-path=/opt/visal/tmp/for_sijin/Data/saved/Test/${run_mac} --train-range=${TrainRange} --test-range=${TestRange} --layer-def=/home/grads/sijinli2/Projects/DHMLPE/doc/netdef2/dhmlpe-layer-def-${JT}.cfg --layer-params=/home/grads/sijinli2/Projects/DHMLPE/doc/netdef2/dhmlpe-layer-params-${JT}.cfg --data-provider=${DP} --test-freq=50 --epoch=${EP} --mini=128 --batch-size=${BSIZE} --gpu 0 --crop-border=-1

