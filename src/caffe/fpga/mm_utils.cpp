#include "caffe/fpga/mm_utils.hpp"

bool is_a_ge_zero_and_a_lt_b(int a, int b) 
{
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

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
)
{
    int tileIter1, tileIter2, iter1, iter2, rowIdx, colIdx; 
    int rowTilesTotal = ceil((float) ROWS / (float) tR);
	int colTilesTotal = ceil((float) COLS / (float) tC);

    int iterBound1 = tR; 
    int iterBound2 = tC; 
    int *row = &iter1; 
    int *col = &iter2; 
    if (transposeMat)
    {
        int tmp = tR; 
        tR = tC; 
        tC = tmp; 

        tmp = COLS; 
        COLS = ROWS; 
        ROWS = tmp; 
        
        tmp = colTilesTotal; 
        colTilesTotal = rowTilesTotal; 
        rowTilesTotal = tmp; 

        row = &iter2; 
        col = &iter1; 

        transposeTiles = !transposeTiles; 
    }
    
    int tileBound1 = rowTilesTotal; 
    int tileBound2 = colTilesTotal; 
    int *tileRowIndex = &tileIter1; 
    int *tileColIndex = &tileIter2; 
    if (transposeTiles)
    {
        tileBound1 = colTilesTotal; 
        tileBound2 = rowTilesTotal; 

        tileRowIndex = &tileIter2; 
        tileColIndex = &tileIter1; 
    }

    params[0] = tileBound1; 
    params[1] = tileBound2; 

    int flatIndex = 0;
    float val; 
    // int val; 
    for (tileIter1 = 0; tileIter1 < tileBound1; tileIter1 ++)
    {
        for (tileIter2 = 0; tileIter2 < tileBound2; tileIter2++)
        {
            for (iter1 = 0; iter1 < iterBound1; iter1++)
            {
                for (iter2 = 0; iter2 < iterBound2; iter2++)
                {
                    rowIdx = *(row) + *(tileRowIndex) * tR; 
                    colIdx = *(col) + *(tileColIndex) * tC; 
                    if (rowIdx >= ROWS || colIdx >= COLS)
                    {
                        val = 0;      
                    }
                    else 
                    {
                        val = inputMat[rowIdx*COLS + colIdx]; 
                    }
                    // tiledFlatMat[flatIndex].push_back(val); 
                    tiledFlatMat.push_back(val); 
                    flatIndex++;
                }
            }
            // flatIndex ++; 
        }
    }
}

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
)
{
	int rowTilesTotal = ceil((float) ROWS / (float) TR);
	int colTilesTotal = ceil((float) COLS / (float) TC);

    int iter1, iter2, tileIter1, tileIter2, rowIdx, colIdx; 

    int iterBound1 = TR; 
    int iterBound2 = TC; 
    int *row = &iter1; 
    int *col = &iter2; 

    int tileBound1 = rowTilesTotal; 
    int tileBound2 = colTilesTotal; 
    int *tileRowIndex = &tileIter1; 
    int *tileColIndex = &tileIter2; 
    if (transposed)
    {
        tileBound1 = colTilesTotal; 
        tileBound2 = rowTilesTotal; 

        tileRowIndex = &tileIter2; 
        tileColIndex = &tileIter1; 
    }

	int flatIndex = 0;
	// int tileIndex = 0;
    // int locIndex = 0; 
    for (tileIter1 = 0; tileIter1 < tileBound1; tileIter1 ++)
    {
        for (tileIter2 = 0; tileIter2 < tileBound2; tileIter2 ++)
        {
            // locIndex = 0; 
            for (iter1 = 0; iter1 < iterBound1; iter1++)
            {
                for (iter2 = 0; iter2 <iterBound2; iter2++)
                {
                    rowIdx = *(row) + *(tileRowIndex) * TR; 
                    colIdx = *(col) + *(tileColIndex) * TC; 
                    if (rowIdx < ROWS && colIdx < COLS)
                    {
                        outputMat[rowIdx * COLS + colIdx] = tiledFlatMat[flatIndex]; 
                        // outputMat[rowIdx * COLS + colIdx] = tiledFlatMat[tileIndex][locIndex]; 
                    }
                    // locIndex++; 
                    flatIndex ++; 
                }
            }
            // tileIndex++;
        }
    }
}
