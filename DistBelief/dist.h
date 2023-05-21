#pragma once
#ifdef SERVER_EXPORTS
#define __Server__ __declspec(dllexport)
#else
#define __Server__ __declspec(dllimport)

#endif

#define _CRT_SECURE_NO_WARNINGS


#include <iostream>
#include <Agent.h>
#include <CUDA_ACC.h>
#include"G_Net.h"

#include<thread>
#include<mutex>
#include<condition_variable>

using namespace Base_Net;
using namespace Net_CPU;

typedef std::function<bool(Agent*, Mat*, int, int, Agent*)> ServerTrainFunc;
typedef std::function<Net*(HyperParamSearcher&)> ServerNetFunc;
__Server__ bool Default_Train_Func(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col, Agent*Server);
typedef Net*(*NetFunc)(HyperParamSearcher&param);
typedef bool(*Train_Option)(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col, Agent*Server);
template<typename Train_Func>
void Server_Test(int trainNum, int testNum, Train_Func train_func = Agent::Default_Train_Func);
template<typename Train_Func, typename NetFunc>
void Server_StartTrain(NetFunc NetFunc, Train_Func train_func = Default_Train_Func);
//template<typename NetFunc>
//void Server_Init(int workers_Num, NetFunc Net, Agent*InitAgent, HyperParamSearcher&param);

__Server__ void Server_Init(int workers_Num, NetFunc Func, string Param_Path);
__Server__ void Server_Init(int workers_Num, Agent*InitAgent, HyperParamSearcher&param);
__Server__ void Server_InitData(Base_Param**param, int paramNum, int trainNum);
template __Server__ void Server_StartTrain<Train_Option, NetFunc>(NetFunc, Train_Option);
template __Server__ void Server_StartTrain<ServerTrainFunc, ServerNetFunc>(ServerNetFunc, ServerTrainFunc);
template __Server__ void Server_Test<Agent::Train_Option>(int, int, Agent::Train_Option);
//template __Server__ void Agent::Train<std::function<bool(Agent*, Mat*, int, int)>>(Base_Param**, int, int, const char*, std::function<bool(Agent*, Mat*, int, int)>, Agent::DataWrite2Device);

__Server__ void Server_Disponse(bool freeAgent = true);