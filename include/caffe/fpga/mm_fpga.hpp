#ifndef MM_FPGA_H_
#define MM_FPGA_H_

#include "caffe/caffe.hpp"
#include "caffe/fpga/mm_utils.hpp"
#include "xcl2.hpp"
#include "math.h"

#include "caffe/fpga/AsyncProfiler.hpp"

#include <vector>

#define TILE_ROW 16         // row size
#define TILE_COL 16 	    // col size
#define TILE_COMMON 64      // common dimension size

// #define TILE_ROW 4         // row size
// #define TILE_COL 4 	    // col size
// #define TILE_COMMON 64    // common dimension size

#define INTER_SIZE 11

// #define SIMULATE_BATCHING
// #define PROFILING
// #define PROFILING_TIME

void Kernel(
      int transA,
      const float *aVecIn, 
      int transB,
      const float *bVecIn, 
      float *cVec, 
      int aRow, 
      int aCol, 
      int bRow,
      int bCol,
      float ABscaling,
      float Cscaling,
      double *fpga_times
);

void Kernel_double_buff(
      int transA,
      const float *aVecIn, 
      int transB,
      const float *bVecIn, 
      float *cVec, 
      int aRow, 
      int aCol, 
      int bRow,
      int bCol,
      float ABscaling,
      float Cscaling,
      double *fpga_times
);

void Kernel_double_ddr(
      int transA,
      const float *aVecIn, 
      int transB,
      const float *bVecIn, 
      float *cVec, 
      int aRow, 
      int aCol, 
      int bRow,
      int bCol,
      float ABscaling,
      float Cscaling,
      double *fpga_times
);

void Kernel_tiling(
    int transA,
    const float *aVecIn, 
    int transB,
    const float *bVecIn, 
    float *cVecOut, 
    int aRow, 
    int aCol, 
    int bRow,
    int bCol,
    float ABscaling,
    float Cscaling,
    double *fpga_times  
);

void Kernel_profiling(
    int transA,
    const float *aVecIn, 
    int transB,
    const float *bVecIn, 
    float *cVecOut, 
    int aRow, 
    int aCol, 
    int bRow,
    int bCol,
    float ABscaling,
    float Cscaling
);
#endif
