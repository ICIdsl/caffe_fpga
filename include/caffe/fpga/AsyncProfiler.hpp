#ifndef ASYNC_PROFILER_H 
#define ASYNC_PROFILER_H

#include "xcl2.hpp"
#include <vector>

class AsyncProfiler
{
private :
	cl_ulong totalExecTime, avgExecTime; 
	std::vector<cl_ulong> startTimes, endTimes; 
	std::vector<cl::Event*> eventsVec; 

public : 
	AsyncProfiler(); 
	~AsyncProfiler(); 
	
	cl::Event* add_event(); 
	cl::Event get_last_event(); 
	void calculate_times(); 
	cl_ulong total(); 
	cl_ulong average(); 	
	void start_times(std::vector<cl_ulong> &startTimes); 
	void end_times(std::vector<cl_ulong> &endTimes); 
	
	void custom_debug(int mode, std::vector<cl_ulong> &times); 
};

#endif //ASYNC_PROFILER_H
