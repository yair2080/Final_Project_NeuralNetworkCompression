#!/bin/bash
export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

unzip NeuralNetworkCompression.zip
#installing L3C
cd NeuralNetworkCompression/models/L3C
conda create --name l3c_env_test python=3.7 pip --yes 
conda activate l3c_env_test 
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge 
pip install -r pip_requirements.txt 
cd src/torchac 
COMPILE_CUDA=force python setup.py install 
#installing SReC
cd -
cd ..
cd SReC
conda create --name srec_env_test python=3.7 pip --yes 
conda activate srec_env_test 
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge 
pip install -r requirements.txt
cd .. 
cd L3C/src/torchac
conda activate srec_env_test  
cd src/torchac 
COMPILE_CUDA=force  python setup.py install 

exit
 
