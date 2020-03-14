#include "caffe/fpga/mm_fpga.hpp"
#include <fstream>
#include <iostream>

std::vector<cl::Device> g_devices = xcl::get_xil_devices();
cl::Device device = g_devices[0];
cl_int err;
cl::Context context(device, NULL, NULL, NULL, &err);
std::string deviceName = device.getInfo<CL_DEVICE_NAME>(&err);

unsigned fileBufSize; 
char* fileBuf = xcl::read_binary_file("mesh_processor_16x16x64.xclbin", fileBufSize);
cl::Program::Binaries bins{{fileBuf,fileBufSize}};

cl::Program program(context, std::vector<cl::Device>{device}, bins, NULL, &err);
cl::Kernel krnl_mesh_proc(program, "matmul", &err);

void Kernel(
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
)
{
    #ifdef PROFILING_TIME
    caffe::Timer tiling; 
    double tilingTime = 0.0; 
    tiling.Start(); 
    #endif
    
    std::vector<float, aligned_allocator<float> > aVec; 
    int aParams[2]; 
	TransformToFlattenTiledLayout(
        aVecIn, 
        aVec, 
        aParams,
        aRow, 
        aCol, 
        TILE_ROW, 
        TILE_COMMON,
        false, 
        transA
    );

    
    std::vector<float, aligned_allocator<float> > bVec;
    int bParams[2]; 
	TransformToFlattenTiledLayout(
        bVecIn, 
        bVec, 
        bParams,
        bRow, 
        bCol, 
        TILE_COMMON,
        TILE_COL,
        true, 
        transB
    );

    std::vector<float, aligned_allocator<float> > cVec;
    int cRow = aParams[0] * TILE_ROW; 
    int cCol = bParams[0] * TILE_COL; 
    for (int i=0; i<cRow * cCol; i++)
    {
        cVec.push_back(0); 
    }
    
    std::vector<int, aligned_allocator<int> > params(4); 
    params[0] = aParams[0]; 
    params[1] = bParams[0]; 
    params[2] = aParams[1]; 
    params[3] = 0; 

    #ifdef PROFILING_TIME
    tilingTime += tiling.MicroSeconds(); 
    #endif

	OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

    cl::Event aWrite, bWrite, paramWrite, cWrite, kernelExec, aRead, bRead, paramsRead, cRead; 

    OCL_CHECK(err, cl::Buffer aVecLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*aVec.size(), aVec.data(), &err));
    OCL_CHECK(err, cl::Buffer bVecLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*bVec.size(), bVec.data(), &err));
    OCL_CHECK(err, cl::Buffer paramsLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int)*params.size(), params.data(), &err));
    OCL_CHECK(err, cl::Buffer cVecLoco(context, CL_MEM_USE_HOST_PTR, sizeof(float)*cVec.size(), cVec.data(), &err));
                
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({aVecLoco}, 0, NULL, &aWrite)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bVecLoco}, 0, NULL, &bWrite)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({paramsLoco}, 0, NULL, &paramWrite)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({cVecLoco}, 0, NULL, &cWrite)); 
    
    q.finish(); 
    
	OCL_CHECK(err, err = krnl_mesh_proc.setArg(0, aVecLoco));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(1, bVecLoco));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(2, paramsLoco));
	OCL_CHECK(err, err = krnl_mesh_proc.setArg(3, cVecLoco));
    
    std::vector<cl::Event> kernel_wait_events = {aWrite, bWrite, paramWrite, cWrite}; 
	OCL_CHECK(err, err = q.enqueueTask(krnl_mesh_proc, &kernel_wait_events, &kernelExec));
    
    std::vector<cl::Event> read_wait_events = {kernelExec}; 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({aVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &aRead)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &bRead)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({paramsLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &paramsRead)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({cVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &cRead)); 
	
    q.finish(); 

    #ifdef PROFILING_TIME
    int teSize = 9;
    cl::Event transfer_events[teSize]; 
    transfer_events[0] = aWrite; 
    transfer_events[1] = bWrite; 
    transfer_events[2] = cWrite; 
    transfer_events[3] = paramWrite; 
    transfer_events[4] = kernelExec; 
    transfer_events[5] = aRead; 
    transfer_events[6] = bRead; 
    transfer_events[7] = paramsRead; 
    transfer_events[8] = cRead; 

    cl_ulong time_start, time_end; 
    std::vector<cl_ulong> event_times; 
    for (unsigned i=0; i<teSize; i++)
    {
	    OCL_CHECK(err, err = transfer_events[i].wait());
	    OCL_CHECK(err, err = transfer_events[i].getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &time_start));
        OCL_CHECK(err, err = transfer_events[i].getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &time_end));
        
        event_times.push_back(time_end - time_start); 
    }

    fpga_times[1] = (event_times[0] + event_times[1] + event_times[2] + event_times[3]) / 1000000.0;  
    fpga_times[2] = (event_times[4]) / 1000000.0; 
    fpga_times[3] = (event_times[5] + event_times[6] + event_times[7] + event_times[8]) / 1000000.0;  
    
    tiling.Start();     
    #endif
    
    TransformToMatrixLayoutFunc(
        cVec, 
        cVecOut, 
        TILE_ROW, 
        TILE_COL, 
        aRow,
        bCol, 
        false
    );
    
    # ifdef PROFILING_TIME
    tilingTime += tiling.MicroSeconds(); 
    fpga_times[0] = tilingTime / 1000.0; 
    #endif
}
