#!/bin/bash
cd ./networks/correlation_package
~/anaconda3/envs/anomaly-pytorch/bin/python3 setup.py install --user
cd ../resample2d_package 
~/anaconda3/envs/anomaly-pytorch/bin/python3 setup.py install --user
cd ../channelnorm_package 
~/anaconda3/envs/anomaly-pytorch/bin/python3 setup.py install --user
cd ..
