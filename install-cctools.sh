#!/bin/sh
set -ex
#wget http://ccl.cse.nd.edu/software/files/cctools-6.0.10-source.tar.gz
#tar -xzvf cctools-6.0.10-source.tar.gz
wget https://github.com/nhazekam/cctools/archive/release/6.0.zip
unzip 6.0.zip
#cd cctools-6.0.10-source && ./configure --with-python-path=/usr --with-swig-path=/usr --with-perl-path=/usr && make && sudo make install
cd cctools-release-6.0 && ./configure --with-python-path=/usr --with-swig-path=/usr --with-perl-path=/usr && make && sudo make install

