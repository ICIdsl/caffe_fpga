#include "caffe/fpga/AsyncProfiler.hpp"

AsyncProfiler::AsyncProfiler() : totalExecTime(0.0) {}

AsyncProfiler::~AsyncProfiler() 
{
	for (int i=0; i<this->eventsVec.size(); i++)
	{
		delete this->eventsVec[i]; 
	}
}

cl::Event* AsyncProfiler::add_event()
{
	cl::Event *newEvent = new cl::Event; 
	this->eventsVec.push_back(newEvent); 
	return newEvent; 
} 

cl::Event AsyncProfiler::get_last_event() 
{
	return *this->eventsVec.back(); 
}

void AsyncProfiler::calculate_times() 
{
	cl_int err; 
	cl_ulong timeStart, timeEnd; 
    
	for (int i=0; i<this->eventsVec.size(); i++)
	{
		OCL_CHECK(err, err = this->eventsVec[i]->wait());
		
		// OCL_CHECK(err, err = this->eventsVec[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_QUEUED, &timeQueued));
		// OCL_CHECK(err, err = this->eventsVec[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_SUBMIT, &timeSubmit));
		OCL_CHECK(err, err = this->eventsVec[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &timeStart));
		
		OCL_CHECK(err, err = this->eventsVec[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &timeEnd));
		
		
		this->startTimes.push_back(timeStart);	
		
		this->endTimes.push_back(timeEnd);	
		
		this->totalExecTime += timeEnd - timeStart; 
		
	}
	
	this->avgExecTime = this->totalExecTime / this->eventsVec.size(); 
}

cl_ulong AsyncProfiler::total() 
{
	return this->totalExecTime; 
}

cl_ulong AsyncProfiler::average() 
{
	return this->avgExecTime; 
}

void AsyncProfiler::start_times(std::vector<cl_ulong> &startTimes) 
{
	startTimes = this->startTimes; 
}

void AsyncProfiler::end_times(std::vector<cl_ulong> &endTimes) 
{
	endTimes = this->endTimes; 
}

void AsyncProfiler::custom_debug(int mode, std::vector<cl_ulong> &times)
{	
	cl_int err; 
	cl_ulong start, end; 

	int iterEnd = this->eventsVec.size(); 
	for (int i=0; i<iterEnd; i++)
	{
		OCL_CHECK(err, err = this->eventsVec[i]->wait()); 	
		OCL_CHECK(err, err = this->eventsVec[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start));
		OCL_CHECK(err, err = this->eventsVec[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &end));
		
		if (mode == 0) 
		{
			times.push_back(start); 
		}
		else 
		{
			times.push_back(end); 
		}
	}
}
