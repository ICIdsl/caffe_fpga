These are the steps to build and run caffe for FPGA.

Step )
Setup anaconda from file provided

Step ) 
Fix Makefile:
Setup ANACONDA_HOME directory in Makefile, otherwise nothign else works

Step )
make clean
make all -j8
make distribute -j8

Step )
./data/get_cifar10.sh
source <...>/xilinx/xrt/
paste xclbin in caffe_fpga/
./data/create_cifar10.sh

Step )
mkdir -p checkpoints/bvlc_alexnet/fpga
