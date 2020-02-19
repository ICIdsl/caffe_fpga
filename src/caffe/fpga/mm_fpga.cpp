#include "caffe/fpga/mm_fpga.hpp"
#include <fstream>
#include <iostream>

std::vector<cl::Device> g_devices = xcl::get_xil_devices();
cl::Device device = g_devices[0];
cl_int err;
cl::Context context(device, NULL, NULL, NULL, &err);
std::string deviceName = device.getInfo<CL_DEVICE_NAME>(&err);

std::string binaryFile = xcl::find_binary_file(deviceName, "mesh_processor_16x16x64");
// std::string binaryFile = xcl::find_binary_file(deviceName, "double_buff_16x16x64");
// std::string binaryFile = xcl::find_binary_file(deviceName, "tiling_fpga_16x16x64");
// std::string binaryFile = xcl::find_binary_file(deviceName, "double_ddr_16x16x64");

cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
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
    
    // LOG(INFO) << "TransformFlatten";
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

    
    // LOG(INFO) << "TransformFlatten";
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

    // LOG(INFO) << "Create Queue";
	OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

    cl::Event aWrite, bWrite, paramWrite, cWrite, kernelExec, aRead, bRead, paramsRead, cRead; 

    // LOG(INFO) << "Create Buffers";
    OCL_CHECK(err, cl::Buffer aVecLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*aVec.size(), aVec.data(), &err));
    OCL_CHECK(err, cl::Buffer bVecLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*bVec.size(), bVec.data(), &err));
    OCL_CHECK(err, cl::Buffer paramsLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int)*params.size(), params.data(), &err));
    OCL_CHECK(err, cl::Buffer cVecLoco(context, CL_MEM_USE_HOST_PTR, sizeof(float)*cVec.size(), cVec.data(), &err));
                
    // LOG(INFO) << "Migrate Memory Objects";
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({aVecLoco}, 0, NULL, &aWrite)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bVecLoco}, 0, NULL, &bWrite)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({paramsLoco}, 0, NULL, &paramWrite)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({cVecLoco}, 0, NULL, &cWrite)); 
    
    q.finish(); 
    
    // LOG(INFO) << "Set Arguments";
	OCL_CHECK(err, err = krnl_mesh_proc.setArg(0, aVecLoco));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(1, bVecLoco));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(2, paramsLoco));
	OCL_CHECK(err, err = krnl_mesh_proc.setArg(3, cVecLoco));
    
    // LOG(INFO) << "Enqueue Task";
    std::vector<cl::Event> kernel_wait_events = {aWrite, bWrite, paramWrite, cWrite}; 
	OCL_CHECK(err, err = q.enqueueTask(krnl_mesh_proc, &kernel_wait_events, &kernelExec));
    
    // LOG(INFO) << "Migrate Memory Objects back";
    std::vector<cl::Event> read_wait_events = {kernelExec}; 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({aVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &aRead)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &bRead)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({paramsLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &paramsRead)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({cVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &cRead)); 
	
    q.finish(); 
    // LOG(INFO) << "Finished";

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

void Kernel_double_buff(
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
    
    // LOG(INFO) << "TransformFlatten";
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

    
    // LOG(INFO) << "TransformFlatten";
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

    // LOG(INFO) << "Create Queue";
	OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

    cl::Event aWrite, bWrite, paramWrite, cWrite, kernelExec, aRead, bRead, paramsRead, cRead; 

    // LOG(INFO) << "Create Buffers";
    OCL_CHECK(err, cl::Buffer aVecLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*aVec.size(), aVec.data(), &err));
    OCL_CHECK(err, cl::Buffer bVecLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*bVec.size(), bVec.data(), &err));
    OCL_CHECK(err, cl::Buffer cVecLoco(context, CL_MEM_USE_HOST_PTR, sizeof(float)*cVec.size(), cVec.data(), &err));
                
    // LOG(INFO) << "Migrate Memory Objects";
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({aVecLoco}, 0, NULL, &aWrite)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bVecLoco}, 0, NULL, &bWrite)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({cVecLoco}, 0, NULL, &cWrite)); 
    
    q.finish(); 
    
    // LOG(INFO) << "Set Arguments";
	OCL_CHECK(err, err = krnl_mesh_proc.setArg(0, aVecLoco));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(1, bVecLoco));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(2, params[0]));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(3, params[1]));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(4, params[2]));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(5, params[3]));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(6, cVecLoco));
    
    // LOG(INFO) << "Enqueue Task";
    std::vector<cl::Event> kernel_wait_events = {aWrite, bWrite, cWrite}; 
	OCL_CHECK(err, err = q.enqueueTask(krnl_mesh_proc, &kernel_wait_events, &kernelExec));
    
    // LOG(INFO) << "Migrate Memory Objects back";
    std::vector<cl::Event> read_wait_events = {kernelExec}; 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({aVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &aRead)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &bRead)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({cVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &cRead)); 
	
    q.finish(); 
    // LOG(INFO) << "Finished";

    #ifdef PROFILING_TIME
    int teSize = 7;
    cl::Event transfer_events[teSize]; 
    transfer_events[0] = aWrite; 
    transfer_events[1] = bWrite; 
    transfer_events[2] = cWrite; 
    transfer_events[3] = kernelExec; 
    transfer_events[4] = aRead; 
    transfer_events[5] = bRead; 
    transfer_events[6] = cRead; 

    cl_ulong time_start, time_end; 
    std::vector<cl_ulong> event_times; 
    for (unsigned i=0; i<teSize; i++)
    {
	    OCL_CHECK(err, err = transfer_events[i].wait());
	    OCL_CHECK(err, err = transfer_events[i].getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &time_start));
        OCL_CHECK(err, err = transfer_events[i].getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &time_end));
        
        event_times.push_back(time_end - time_start); 
    }

    fpga_times[1] = (event_times[0] + event_times[1] + event_times[2]) / 1000000.0;  
    fpga_times[2] = (event_times[3]) / 1000000.0; 
    fpga_times[3] = (event_times[4] + event_times[5] + event_times[6]) / 1000000.0;
    
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

void Kernel_double_ddr(
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
    
    // LOG(INFO) << "TransformFlatten";
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

    
    // LOG(INFO) << "TransformFlatten";
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

    // LOG(INFO) << "Create Queue";
	OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

    cl::Event aWrite, bWrite, paramWrite, cWrite, kernelExec, aRead, bRead, paramsRead, cRead; 

    cl_mem_ext_ptr_t inExt1, inExt2, outExt; 
    inExt1.flags = XCL_MEM_DDR_BANK0;
    inExt2.flags = XCL_MEM_DDR_BANK1;
    outExt.flags = XCL_MEM_DDR_BANK2;

    inExt1.obj = aVec.data();
    inExt2.obj = bVec.data();
    outExt.obj = cVec.data();

    inExt1.param = 0;
    inExt2.param = 0;
    outExt.param = 0;

    // LOG(INFO) << "Create Buffers";
    OCL_CHECK(err, cl::Buffer aVecLoco(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR, sizeof(float)*aVec.size(), &inExt1, &err));
    OCL_CHECK(err, cl::Buffer bVecLoco(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR, sizeof(float)*bVec.size(), &inExt2, &err));
    OCL_CHECK(err, cl::Buffer cVecLoco(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR, sizeof(float)*cVec.size(), &outExt, &err));
                
    // LOG(INFO) << "Migrate Memory Objects";
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({aVecLoco}, 0, NULL, &aWrite)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bVecLoco}, 0, NULL, &bWrite)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({cVecLoco}, 0, NULL, &cWrite)); 
    
    q.finish(); 
    
    // LOG(INFO) << "Set Arguments";
	OCL_CHECK(err, err = krnl_mesh_proc.setArg(0, aVecLoco));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(1, bVecLoco));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(2, params[0]));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(3, params[1]));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(4, params[2]));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(5, params[3]));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(6, cVecLoco));
    
    // LOG(INFO) << "Enqueue Task";
    std::vector<cl::Event> kernel_wait_events = {aWrite, bWrite, cWrite}; 
	OCL_CHECK(err, err = q.enqueueTask(krnl_mesh_proc, &kernel_wait_events, &kernelExec));
    
    // LOG(INFO) << "Migrate Memory Objects back";
    std::vector<cl::Event> read_wait_events = {kernelExec}; 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({aVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &aRead)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &bRead)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({cVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &cRead)); 
	
    q.finish(); 
    // LOG(INFO) << "Finished";

    #ifdef PROFILING_TIME
    int teSize = 7;
    cl::Event transfer_events[teSize]; 
    transfer_events[0] = aWrite; 
    transfer_events[1] = bWrite; 
    transfer_events[2] = cWrite; 
    transfer_events[3] = kernelExec; 
    transfer_events[4] = aRead; 
    transfer_events[5] = bRead; 
    transfer_events[6] = cRead; 

    cl_ulong time_start, time_end; 
    std::vector<cl_ulong> event_times; 
    for (unsigned i=0; i<teSize; i++)
    {
	    OCL_CHECK(err, err = transfer_events[i].wait());
	    OCL_CHECK(err, err = transfer_events[i].getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &time_start));
        OCL_CHECK(err, err = transfer_events[i].getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &time_end));
        
        event_times.push_back(time_end - time_start); 
    }

    fpga_times[1] = (event_times[0] + event_times[1] + event_times[2]) / 1000000.0;  
    fpga_times[2] = (event_times[3]) / 1000000.0; 
    fpga_times[3] = (event_times[4] + event_times[5] + event_times[6]) / 1000000.0;
    
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
)
{
    // LOG(INFO) << "TransformFlatten";
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
    
    // LOG(INFO) << "TransformFlatten";
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
    
    #ifdef PROFILING_TIME
    caffe::Timer tiling; 
    double tilingTime = 0.0; 
    tiling.Start(); 
    #endif
    
    
    std::vector<float, aligned_allocator<float> > aVecInNew; 
    for(int i=0; i<aRow*aCol; i++)
    {
        aVecInNew.push_back(aVecIn[i]);
    }

    std::vector<float, aligned_allocator<float> > bVecInNew; 
    for(int i=0; i<bRow*bCol; i++)
    {
        bVecInNew.push_back(bVecIn[i]);
    }

    std::vector<float, aligned_allocator<float> > cVec;
    int cRow = aRow; 
    int cCol = bCol; 
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

    // LOG(INFO) << "Create Queue";
	OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

    cl::Event aWrite, bWrite, paramWrite, cWrite, kernelExec, aRead, bRead, paramsRead, cRead; 

    // LOG(INFO) << "Create Buffers";
    OCL_CHECK(err, cl::Buffer aVecLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*aRow*aCol, aVecInNew.data(), &err));
    OCL_CHECK(err, cl::Buffer bVecLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*bRow*bCol, bVecInNew.data(), &err));
    OCL_CHECK(err, cl::Buffer cVecLoco(context, CL_MEM_USE_HOST_PTR, sizeof(float)*cRow*cCol, cVec.data(), &err));
                
    // LOG(INFO) << "Migrate Memory Objects";
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({aVecLoco}, 0, NULL, &aWrite)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bVecLoco}, 0, NULL, &bWrite)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({cVecLoco}, 0, NULL, &cWrite)); 
    
    q.finish(); 

    // LOG(INFO) << "Set Arguments";
	OCL_CHECK(err, err = krnl_mesh_proc.setArg(0, aVecLoco));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(1, bVecLoco));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(2, params[0]));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(3, params[1]));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(4, params[2]));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(5, params[3]));
	OCL_CHECK(err, err = krnl_mesh_proc.setArg(6, cVecLoco));
    
    // LOG(INFO) << "Enqueue Task";
    std::vector<cl::Event> kernel_wait_events = {aWrite, bWrite, cWrite}; 
	OCL_CHECK(err, err = q.enqueueTask(krnl_mesh_proc, &kernel_wait_events, &kernelExec));
    
    // LOG(INFO) << "Migrate Memory Objects back";
    std::vector<cl::Event> read_wait_events = {kernelExec}; 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({aVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &aRead)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &bRead)); 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({cVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &cRead)); 
	
    q.finish(); 
    // LOG(INFO) << "Finished";

    #ifdef PROFILING_TIME
    int teSize = 7;
    cl::Event transfer_events[teSize]; 
    transfer_events[0] = aWrite; 
    transfer_events[1] = bWrite; 
    transfer_events[2] = cWrite; 
    transfer_events[3] = kernelExec; 
    transfer_events[4] = aRead; 
    transfer_events[5] = bRead; 
    transfer_events[6] = cRead; 

    cl_ulong time_start, time_end; 
    std::vector<cl_ulong> event_times; 
    for (unsigned i=0; i<teSize; i++)
    {
	    OCL_CHECK(err, err = transfer_events[i].wait());
	    OCL_CHECK(err, err = transfer_events[i].getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &time_start));
        OCL_CHECK(err, err = transfer_events[i].getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &time_end));
        
        event_times.push_back(time_end - time_start); 
    }

    fpga_times[1] = (event_times[0] + event_times[1] + event_times[2]) / 1000000.0;  
    fpga_times[2] = (event_times[3]) / 1000000.0; 
    fpga_times[3] = (event_times[4] + event_times[5] + event_times[6]) / 1000000.0;  
    
    tiling.Start();     
    #endif
    
    for(int i=0; i<cRow*cCol; i++)
    {
        cVecOut[i] = cVec[i];
    }
    
    # ifdef PROFILING_TIME
    tilingTime += tiling.MicroSeconds(); 
    fpga_times[0] = tilingTime / 1000.0; 
    #endif
}

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
)
{
    std::cout << "Running batching simulation" << std::endl; 
    std::ofstream profiling_log; 
    profiling_log.open("/home/centos/src/project_data/caffe/huawei_proj/profile/log_sim.csv", std::ofstream::app);

    unsigned num_mult = 14; 
    int m_sim[14] = {64,192,384,256,256,256,2304,256,3456,384,1728,192,1600,64};
    int n_sim[14] = {64,16,4,4,4,2304,4,3456,4,1728,4,1600,16,363};
    int k_sim[14] = {363,1600,1728,3456,2304,4,256,4,256,4,384,16,192,64};
     
    double fpga_times[4];      
    caffe::Timer tiling; 

    // for(unsigned ix=0; ix<num_mult; ix++)
    // {
        unsigned ix = 1; 
        std::cout << ix << std::endl; 
        int batch_size = 128;
        int a_size = m_sim[ix] * k_sim[ix] * batch_size; 
        int b_size = n_sim[ix] * k_sim[ix] * batch_size; 

        std::cout << a_size << "," << b_size << std::endl; 

        float *aVecSim = new float[a_size];
        float *bVecSim = new float[b_size];
        //  int *aVecSim = new int[a_size];
        //  int *bVecSim = new int[b_size];
    
        profiling_log << TILE_ROW << "," << TILE_COL << "," << TILE_COMMON << "," << m_sim[ix] << "," << n_sim[ix] << "," << k_sim[ix]*batch_size << "," << "NULL," ;  
        
        // int tmp; 
        float tmp; 
        for (int i=0; i<a_size; i++)
        {
            tmp = (rand() / (float)RAND_MAX * 20) + -10;    
            aVecSim[i] = tmp; 
        }
        for (int i=0; i<b_size; i++)
        {
            tmp = (rand() / (float)RAND_MAX * 20) + -10;    
            bVecSim[i] = tmp; 
        }
        
        double tilingTime = 0.0; 
        // std::cout << "here 1" << std::endl; 
        tiling.Start(); 
    
        std::vector<float, aligned_allocator<float> > aVec; 
        // std::vector<int, aligned_allocator<int> > aVec; 
        int aParams[2]; 
	    TransformToFlattenTiledLayout(
            aVecSim, 
            aVec, 
            aParams,
            m_sim[ix], 
            k_sim[ix]*batch_size, 
            TILE_ROW, 
            TILE_COMMON,
            false, 
            transA
        );
        delete []aVecSim; 
        
        std::vector<float, aligned_allocator<float> > bVec;
        // std::vector<int, aligned_allocator<int> > bVec;
        int bParams[2]; 
	    TransformToFlattenTiledLayout(
            bVecSim, 
            bVec, 
            bParams,
            k_sim[ix] * batch_size, 
            n_sim[ix], 
            TILE_COMMON,
            TILE_COL,
            true, 
            transB
        );
        delete []bVecSim; 
        
        std::vector<float, aligned_allocator<float> > cVec;
        // std::vector<int, aligned_allocator<int> > cVec;
        for (int i=0; i<m_sim[ix] * n_sim[ix]; i++)
        {
            cVec.push_back(0); 
        }

        std::vector<int, aligned_allocator<int> > params(4); 
        params[0] = aParams[0]; 
        params[1] = bParams[0]; 
        params[2] = aParams[1]; 
        params[3] = 0; 

        tilingTime += tiling.MicroSeconds(); 
        // std::cout << "here 1.5" << std::endl; 
        
        #ifdef DEBUG
        std::cout << "Debugging on" << std::endl; 
        std::vector<float> cVecTmp; 
        for (int i=0; i<cRow; i++)
        {
            for (int j=0; j<cCol; j++)
            {
                cVecTmp.push_back(0); 
            }
        }
        for (int tile1 = 0; tile1 < params[0]; tile1++)
        {
            for (int tile2 = 0; tile2 < params[1]; tile2++)
            {
                int cTile = tile1 * params[1] + tile2; 

                for (int tile3 = 0; tile3 < params[2]; tile3++)
                {
                    int aTile = tile1 * params[2] + tile3; 
                    int bTile = tile2 * params[2] + tile3; 

                    for (int i=0; i<TILE_ROW; i++)
                    {
                        for (int j=0; j<TILE_COL; j++)
                        {
                            for (int k=0; k<TILE_COMMON; k++)
                            {
                                cVecTmp.at(cTile * TILE_ROW * TILE_COL + i*TILE_COL + j) += aVec.at(aTile * TILE_ROW * TILE_COMMON + i*TILE_COMMON + k) * bVec.at(bTile * TILE_COL * TILE_COMMON + k*TILE_COL + j);
                            }
                        }
                    }
                }
            }
        }
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
        fpga_times[3] = (event_times[5]) / 1000000.0;
        
        #ifdef DEBUG
        for (int i=0; i<aRow; i++)
        {
            for (int j=0; j<bCol; j++)
            {
                if (cVec[i*bCol + j] - cVecTmp[i*bCol + j] > 0.0001) 
                {
                    std::cout << "tiling issue?" << std::endl; 
                    std::cout << i << "," << j << std::endl; 
                    std::cout << cVec[i*bCol + j] - cVecTmp[i*bCol + j] << std::endl; 
                }
            }
        }
        #endif
        
        // std::cout << "here 2" << std::endl; 
        tiling.Start();     
        TransformToMatrixLayoutFunc(
            cVec, 
            cVecOut, 
            TILE_ROW, 
            TILE_COL, 
            m_sim[ix],
            n_sim[ix], 
            false
        );
        tilingTime += tiling.MicroSeconds(); 
        // std::cout << "here 3" << std::endl; 
        fpga_times[0] = tilingTime / 1000.0; 
        // std::cout << "here 4" << std::endl; 
        
        double total_time = 0.0; 
        for (unsigned i=0; i<4; i++)
        {
            total_time += fpga_times[i]; 
        }
        // std::cout << "here 5" << std::endl; 
        
        profiling_log << fpga_times[0] << "," << fpga_times[1] << "," << fpga_times[2] << "," << fpga_times[3] << "," << total_time << std::endl; 
        // std::cout << "here 6" << std::endl; 
    // }
    
    profiling_log.close(); 
}
