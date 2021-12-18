#! /bin/bash
for dataset in PIE OfficeCaltech
do
    for beta in 0.1 1 10 100
    do
        /home/ying-peng/anaconda3/envs/pj/bin/python active_learning.py --beta $beta --dataset $dataset
    done
done