#!/bin/sh
set -ex
wget http://ccl.cse.nd.edu/software/files/cctools-6.0.6-source.tar.gz
tar -xzvf cctools-6.0.6-source.tar.gz
cd cctools-6.0.6-source && ./configure --with-python-path=/usr --with-swig-path=/usr --with-perl-path=/usr && make && sudo make install

