#!/bin/bash

git clone https://github.com/facebookresearch/cc_net.git

cd cc_net

mkdir -p data

sudo apt-get update

sudo apt-get install libeigen3-dev -y

sudo apt-get install libboost-all-dev -y

make install

make lang=tr dl_lm

pip install cc_net[getpy]

cd ..
