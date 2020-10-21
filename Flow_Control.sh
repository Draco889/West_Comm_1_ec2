#!/bin/bash

DISPLAY=:1 && echo "step0" >> /home/markkhusidman/Desktop/Brandywine/crontest2.txt
source /home/markkhusidman/.bashrc
PATH=/home/markkhusidman/anaconda3/envs/West_Comm_1/bin:/home/markkhusidman/anaconda3/condabin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/arkkhusidman/.local/bin:/home/markkhusidman/bin
source /home/markkhusidman/anaconda3/etc/profile.d/conda.sh && echo "step1" >> /home/markkhusidman/Desktop/Brandywine/crontest2.txt
conda activate /home/markkhusidman/anaconda3/envs/West_Comm_1 && echo "step2" >> /home/markkhusidman/Desktop/Brandywine/crontest2.txt
export DISPLAY=:1
echo $PATH >> /home/markkhusidman/Desktop/Brandywine/crontest2.txt
echo `which python3` >> /home/markkhusidman/Desktop/Brandywine/crontest2.txt
echo `env` >> /home/markkhusidman/Desktop/Brandywine/crontest2.txt
echo `python3 --version` >> /home/markkhusidman/Desktop/Brandywine/crontest2.txt
echo `python3 /home/markkhusidman/Desktop/West_Comm_1/Flow_Control.py 2>&1` >> /home/markkhusidman/Desktop/Brandywine/crontest2.txt
echo "step3" >> /home/markkhusidman/Desktop/Brandywine/crontest2.txt
