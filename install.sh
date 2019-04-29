#!/bin/bash
cd ./networks/correlation_package
/usr/local/Cellar/python3/3.6.3/bin/python3 setup.py install --user
cd ../resample2d_package 
/usr/local/Cellar/python3/3.6.3/bin/python3 setup.py install --user
cd ../channelnorm_package 
/usr/local/Cellar/python3/3.6.3/bin/python3 setup.py install --user
cd ..
