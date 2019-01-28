#include "caffe/fpga/mm_fpga.hpp"

std::vector<cl::Device> g_devices = xcl::get_xil_devices();
cl::Device device = g_devices[0];
cl_int err;
cl::Context context(device, NULL, NULL, NULL, &err);
// cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
std::string deviceName = device.getInfo<CL_DEVICE_NAME>(&err);
std::string binaryFile = xcl::find_binary_file(deviceName, "mesh_processor");
cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
cl::Program program(context, std::vector<cl::Device>{device}, bins, NULL, &err);
// cl::Kernel krnl_mesh_proc(program, "mesh_processor", &err);
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
      float Cscaling
      )
{
    
    int ix = 1; 
    int m_sim[6] = {64,192,384,256,256,128};
    int n_sim[6] = {64,16,4,4,4,10};
    int k_sim[6] = {363,1600,1728,3456,2304,256};
    int batch_size = 128;
    int a_size = m_sim[ix] * k_sim[ix] * batch_size; 
    int b_size = n_sim[ix] * k_sim[ix] * batch_size; 

    std::cout << m_sim[ix] << std::endl; 
    std::cout << n_sim[ix] << std::endl; 
    std::cout << k_sim[ix] << std::endl; 
    std::cout << a_size << std::endl; 
    std::cout << b_size << std::endl; 
    
    // std::cout << "0" << std::endl; 
    
    float *aVecSim = new float[a_size];
    float *bVecSim = new float[b_size];
    //  int *aVecSim = new int[a_size];
    //  int *bVecSim = new int[b_size];
    
    // std::cout << "0.5" << std::endl; 

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

    caffe::Timer tiling; 
    double tilingTime = 0.0; 

    // std::cout << "1" << std::endl; 

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
    // std::cout << "2" << std::endl; 

    // std::vector<float, aligned_allocator<float> > aVec; 
    // int aParams[2]; 
	// TransformToFlattenTiledLayout(
    //     aVecIn, 
    //     aVec, 
    //     aParams,
    //     aRow, 
    //     aCol, 
    //     TILE_ROW, 
    //     TILE_COMMON,
    //     false, 
    //     transA
    // );

    // std::vector<float, aligned_allocator<float> > bVec;
    // int bParams[2]; 
	// TransformToFlattenTiledLayout(
    //     // bVecIn, 
    //     bVecSim, 
    //     bVec, 
    //     bParams,
    //     bRow, 
    //     bCol, 
    //     TILE_COMMON,
    //     TILE_COL,
    //     true, 
    //     transB
    // );
    
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
    // std::cout << "3" << std::endl; 
    
    std::vector<int, aligned_allocator<int> > params(4); 
    params[0] = aParams[0]; 
    params[1] = bParams[0]; 
    params[2] = aParams[1]; 
    params[3] = 0; 

    // std::vector<float, aligned_allocator<float> > cVec;
    // for (int i=0; i<aRow * bCol; i++)
    // {
    //     cVec.push_back(0); 
    // }
    std::vector<float, aligned_allocator<float> > cVec;
    // std::vector<int, aligned_allocator<int> > cVec;
    for (int i=0; i<m_sim[ix] * n_sim[ix]; i++)
    {
        cVec.push_back(0); 
    }
    // std::cout << "4" << std::endl; 
    
    tilingTime += tiling.MicroSeconds(); 

    // std::cout << "Tiling time = " << tilingTime << std::endl; 

	// OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
	OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    
    cl::Event aWrite, bWrite, paramWrite, cWrite, kernelExec; 

    
    //OCL_CHECK(err, cl::Buffer aVecLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*aVec.size(), aVec.data(), &err));
    //OCL_CHECK(err, cl::Buffer bVecLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*bVec.size(), bVec.data(), &err));
    //OCL_CHECK(err, cl::Buffer paramsLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int)*params.size(), params.data(), &err));
    //OCL_CHECK(err, cl::Buffer cVecLoco(context, CL_MEM_USE_HOST_PTR, sizeof(float)*cVec.size(), cVec.data(), &err));
    
    // OCL_CHECK(err, cl::Buffer aVecLoco(context, CL_MEM_READ_ONLY, sizeof(float)*aVec.size(), NULL, &err));
    // OCL_CHECK(err, cl::Buffer bVecLoco(context, CL_MEM_READ_ONLY, sizeof(float)*bVec.size(), NULL, &err));
    // OCL_CHECK(err, cl::Buffer paramsLoco(context, CL_MEM_READ_ONLY, sizeof(int)*params.size(), NULL, &err));
    // OCL_CHECK(err, cl::Buffer cVecLoco(context, CL_MEM_READ_WRITE, sizeof(float)*cVec.size(), NULL, &err));
    
    OCL_CHECK(err, cl::Buffer aVecLoco(context, CL_MEM_READ_ONLY, sizeof(int)*aVec.size(), NULL, &err));
    OCL_CHECK(err, cl::Buffer bVecLoco(context, CL_MEM_READ_ONLY, sizeof(int)*bVec.size(), NULL, &err));
    OCL_CHECK(err, cl::Buffer paramsLoco(context, CL_MEM_READ_ONLY, sizeof(int)*params.size(), NULL, &err));
    OCL_CHECK(err, cl::Buffer cVecLoco(context, CL_MEM_READ_WRITE, sizeof(int)*cVec.size(), NULL, &err));
                
    //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({aVecLoco}, 0, NULL, &aWrite)); 
    //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bVecLoco}, 0, NULL, &bWrite)); 
    //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({paramsLoco}, 0, NULL, &paramWrite)); 
    //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({cVecLoco}, 0, NULL, &cWrite)); 
    // OCL_CHECK(err, err = clEnqueueWriteBuffer(q(), aVecLoco(), CL_TRUE, 0, sizeof(float)*aVec.size(), aVec.data(), 0, NULL, NULL));
    // OCL_CHECK(err, err = clEnqueueWriteBuffer(q(), bVecLoco(), CL_TRUE, 0, sizeof(float)*bVec.size(), bVec.data(), 0, NULL, NULL));
    // OCL_CHECK(err, err = clEnqueueWriteBuffer(q(), paramsLoco(), CL_TRUE, 0, sizeof(int)*params.size(), params.data(), 0, NULL, NULL));
    // OCL_CHECK(err, err = clEnqueueWriteBuffer(q(), cVecLoco(), CL_TRUE, 0, sizeof(float)*cVec.size(), cVec.data(), 0, NULL, NULL));
    OCL_CHECK(err, err = clEnqueueWriteBuffer(q(), aVecLoco(), CL_TRUE, 0, sizeof(int)*aVec.size(), aVec.data(), 0, NULL, NULL));
    OCL_CHECK(err, err = clEnqueueWriteBuffer(q(), bVecLoco(), CL_TRUE, 0, sizeof(int)*bVec.size(), bVec.data(), 0, NULL, NULL));
    OCL_CHECK(err, err = clEnqueueWriteBuffer(q(), paramsLoco(), CL_TRUE, 0, sizeof(int)*params.size(), params.data(), 0, NULL, NULL));
    OCL_CHECK(err, err = clEnqueueWriteBuffer(q(), cVecLoco(), CL_TRUE, 0, sizeof(int)*cVec.size(), cVec.data(), 0, NULL, NULL));
    

	OCL_CHECK(err, err = krnl_mesh_proc.setArg(0, aVecLoco));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(1, bVecLoco));
    OCL_CHECK(err, err = krnl_mesh_proc.setArg(2, paramsLoco));
	OCL_CHECK(err, err = krnl_mesh_proc.setArg(3, cVecLoco));
    
    caffe::Timer fpgaFwdTimer;
	double fpgaFwdTime = 0.0;
    fpgaFwdTimer.Start(); 

    //std::vector<cl::Event> kernel_wait_events = {aWrite, bWrite, paramWrite, cWrite}; 
	//OCL_CHECK(err, err = q.enqueueTask(krnl_mesh_proc, &kernel_wait_events, &kernelExec));
    OCL_CHECK(err, err = q.enqueueTask(krnl_mesh_proc));
    q.flush();
    
    fpgaFwdTime += fpgaFwdTimer.MicroSeconds();
    std::cout << "Compute time = " << fpgaFwdTime << std::endl; 
    
    // std::vector<cl::Event> read_wait_events = {kernelExec}; 
    // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({cVecLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, NULL)); 
    OCL_CHECK(err, err = clEnqueueReadBuffer(q(), cVecLoco(), CL_TRUE, 0, sizeof(float)*cVec.size(), cVec.data(), 0, NULL, NULL));
	
    q.finish(); 
    // std::cout << "6" << std::endl; 

    // TransformToMatrixLayoutFunc(
    //     cVec, 
    //     cVecOut, 
    //     TILE_ROW, 
    //     TILE_COL, 
    //     aRow,
    //     bCol, 
    //     false
    // );

    TransformToMatrixLayoutFunc(
        cVec, 
        cVecOut, 
        TILE_ROW, 
        TILE_COL, 
        m_sim[ix],
        n_sim[ix], 
        false
    );
    // std::cout << "7" << std::endl; 

    delete []aVecSim; 
    delete []bVecSim; 
}
