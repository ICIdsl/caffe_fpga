These are the steps to build and run caffe for FPGA.

Step )
Setup anaconda from file provided

Step ) 
Fix Makefile:
Setup ANACONDA_HOME directory in Makefile

Step )
make clean
make all -j8
make distribute -j8

Step )
./data/cifar10/get_cifar10.sh
source <...>/xilinx/xrt/setup.sh
paste xclbin in caffe_fpga/
./data/cifar10/create_cifar10.sh

Step )
mkdir -p checkpoints/bvlc_alexnet/fpga
