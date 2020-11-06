# Caffe Barista 
## Overview
This is an open-source implementation of the work presented in the paper "Caffe Barista: Brewing Caffe with FPGAs in the Training Loop" that was published as a short paper in FPL2020. 
If you reference this work in a publication, we would appreciate you using the following citation: 
```
@misc{vink2020caffe,
      title={Caffe Barista: Brewing Caffe with FPGAs in the Training Loop}, 
      author={Diederik Adriaan Vink and Aditya Rajagopal and Stylianos I. Venieris and Christos-Savvas Bouganis},
      year={2020},
      eprint={2006.13829},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```
Checkout https://www.imperial.ac.uk/intelligent-digital-systems to see other publications by the Intelligent Digital Systems Lab at Imperial College London.

The implementation involes a systolic array with all batching and im2col operations being left on the CPU and handled by Caffe.
The tool works exactly like standard Caffe, except for the fact that it now also accepts FPGA as a parameter for the solver_mode.
This was done to ensure exisiting networks could be run using this tool with minimal changes required.
This version of caffe can be downloaded and directly run, requiring only to follow the steps in the Installation Procedure section.

This tool has only been tested on Xilinx FPGAs, and all OpenCL calls follow the Xilinx OpenCL extension.
It also requires the user to have a functioning installation of Anaconda3 or Miniconda3.

## Implementation Details
All files defined here have an equivalent header file in the *include/caffe/...* directories.

#### src/caffe/fpga/mm_fpga.cpp
Includes the *Kernel* function referenced in *src/caffe/util/match_functions.cpp*.
This contains all of the OpenCL calls to setup and execute the FPGA kernel.

#### src/caffe/fpga/mm_utils.cpp
Contains the functions needed to create the tiles for execution.
The size of these tiles varies in accordance to *mm_fpga.hpp*.

#### src/caffe/layers/conv_layer.cpp
Replicated the *Forward_cpu* and *Backward_cpu* to call the relevant FPGA functions rather than the CPU equivalents.

#### src/caffe/layers/base_conv_layer.cpp
Just like for *conv_layer.cpp* we replicated the *forward_cpu_gemm*, *backward_cpu_gemm* and *weight_cpu_gemm* fucntions but replaced all *caffe_cpu_gemm* functions into *caffe_fpga_gemm*.
*forward_cpu_bias* and *backward_cpu_bias* were replicated to create equivalents so they can be called along the fpga execution track, but are still executed on the CPU.

#### src/caffe/util/math_functions.cpp
Implements the *caffe_fpga_gemm* functions to call the Kernel function which is defined in the files located in *src/caffe/fpga* directory.
This also comes with built in verification and profiling systems.

## Installation Procedure
These are the steps to build and run caffe for FPGA.

#### Step 1)
Setup anaconda from caffe_fpga.yml.
Update the "prefix" section of the yml to be appropriate for the installation machine.

#### Step 2) 
Setup ANACONDA_HOME directory in Makefile.config.

#### Step 3)
make clean

make all -j8

make distribute -j8

#### Step 4)
./data/cifar10/get_cifar10.sh

source <...>/xilinx/xrt/setup.sh

paste xclbin from https://imperialcollegelondon.box.com/s/ubjxfx739htpxroqv44ed82qa2oi2mtc (or your own) in caffe_fpga/

./data/cifar10/create_cifar10.sh

#### Step 5)
mkdir -p checkpoints/bvlc_alexnet/fpga

#### Step 6)
./build/tools/caffe train --solver=solvers/solver.prototxt
