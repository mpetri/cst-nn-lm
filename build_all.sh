#!/bin/sh

git submodule update --init
mkdir build
cd build
cmake ..
make -j 8
