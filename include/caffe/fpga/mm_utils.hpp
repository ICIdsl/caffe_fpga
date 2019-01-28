#ifndef MM_UTILS_H_
#define MM_UTILS_H_

#include <vector>
#include "xcl2.hpp"
#include "math.h"

bool is_a_ge_zero_and_a_lt_b(int a, int b);

void TransformToFlattenTiledLayout(
    const float *inputMat, 
    std::vector<float, aligned_allocator<float>> &tiledFlatMat, 
    // const int *inputMat, 
    // std::vector<int, aligned_allocator<int>> &tiledFlatMat, 
    int* params, 
    int ROWS, 
    int COLS, 
    int tR, 
    int tC, 
    bool transposeTiles, 
    bool transposeMat
);

void TransformToMatrixLayoutFunc(
		std::vector<float, aligned_allocator<float>> &tiledFlatMat,
		float *outputMat,
		// std::vector<int, aligned_allocator<int>> &tiledFlatMat,
		// float *outputMat,
		// int *outputMat,
		int TR,
		int TC,
        int ROWS, 
        int COLS,
        bool transposed
);

#endif
