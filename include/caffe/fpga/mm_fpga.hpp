#ifndef MM_FPGA_H_
#define MM_FPGA_H_

#include "caffe/caffe.hpp"
#include "caffe/fpga/mm_utils.hpp"
#include "xcl2.hpp"
#include "math.h"

#include "caffe/fpga/AsyncProfiler.hpp"

#include <vector>

// #define TILE_ROW 16         // row size
// #define TILE_COL 16 	    // col size
// #define TILE_COMMON 4000     // common dimension size

#define TILE_ROW 16         // row size
#define TILE_COL 16 	    // col size
#define TILE_COMMON 500     // common dimension size

#define INTER_SIZE 17

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
      float Cscaling
      // bool print
      // std::vector<AsyncProfiler*> &profilers
      );
#endif
