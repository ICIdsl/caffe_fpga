These are the steps to build and run caffe for FPGA.

Step 1)
Setup anaconda from file provided

Step 2) 
Fix Makefile:
Setup ANACONDA_HOME directory in Makefile

Step 3)
make clean
make all -j8
make distribute -j8

Step 4)
./data/cifar10/get_cifar10.sh
source <...>/xilinx/xrt/setup.sh
paste xclbin in caffe_fpga/
./data/cifar10/create_cifar10.sh

Step 5)
mkdir -p checkpoints/bvlc_alexnet/fpga

Step 6)
./build/tools/caffe train --solver=solvers/solver.prototxt
