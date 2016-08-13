#!/bin/bash

sudo apt-get update 
sudo apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
sudo apt-get install -y python-pip python-dev
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
#export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL
sudo apt-get install -y openmpi-bin openmpi-doc libopenmpi-dev
sudo pip install mpi4py
git config --global user.email "tdhst231@mail.rmu.edu"
git config --global user.name "Trae Hurley"
cd
cd .ssh 
#sudo su -root
ssh-keygen -t dsa -f id_dsa -N ""
#cd
#cd summer-research-2016
#chmod -R guo+rw ./



