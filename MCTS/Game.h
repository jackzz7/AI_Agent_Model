#pragma once
#define _CRT_SECURE_NO_WARNINGS
#ifdef GAME_EXPORTS
#define __GAME__ __declspec(dllexport)
#else
#define __GAME__ __declspec(dllimport)

#endif
#include <iostream>
#include <conio.h>
#include<Windows.h>
#include<time.h>
#include<assert.h>

#include<Param.h>
#include<Attention.h>
#include<ConvNet.h>

#include"RL_MCTS.h"

using namespace Attention;

using namespace Base_Net;
using namespace Net_CPU;

using namespace std;


template<class T> constexpr inline const T& _min(const T&a, const T&b);
template<class T> constexpr inline const T& _max(const T&a, const T&b);

template __GAME__ const double& _min(const double&, const double&);
template __GAME__ const double& _max(const double&, const double&);

//Ball Game Options
void BallGame_User_Param(HyperParamSearcher&param);
Net* BallGame_AgentNet(HyperParamSearcher&param);
double BallGame_StartGame(Agent*agent);

//Five Chess
#define pi pair<int,int>
#define X first
#define Y second
Net*FiveChess_RewardNet(HyperParamSearcher&param);
Net*FiveChess_FC_PolicyNet(HyperParamSearcher&param);
Net*FiveChess_SL_PolicyNet(HyperParamSearcher&param);
Net*FiveChess_RL_PolicyNet(HyperParamSearcher&param);
Net*Gomoku_SL_PolicyNet(HyperParamSearcher&param);
Net*Gomoku_ValueNet(HyperParamSearcher&param);
Net*Gomoku_Policy_Value_Net(HyperParamSearcher&param);
Net*Gomoku_Policy_Value_ResNet(HyperParamSearcher&param);
//void FiveChess_User_Param(HyperParamSearcher&param);
double FiveChess_StartGame(Agent**_agent, int AgentNum, double&Win_ratio, int Turn, bool Test = false);
//static bool GomokuSimulation(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col);
//static bool RL_GomokuSimulation(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col);
//trainSet API


struct Gomoku :LSTM_Param {
	Gomoku() {}
	Gomoku(int Game_ID, int Game_Step) :LSTM_Param(1) {
		param[0]._DataIn->Reset(1);
		//board id
		param[0]._DataIn->SetValue(1, 0, Game_ID * 1000.0 + Game_Step, false);
	}
	Gomoku(int Game_ID, int Game_Step, double*OutPut, int MaxStep) :LSTM_Param(1) {
		param[0]._DataIn->Reset(1 + MaxStep);
		//board id
		param[0]._DataIn->SetValue(1, 0, Game_ID * 1000.0 + Game_Step, false);
		//OutPut
		for (int i = 0; i < MaxStep; i++)
			param[0]._DataIn->SetValue(1, 0, OutPut[i], false);
	}
	static void SetMove(LSTM_Param&data,const pi&move, int Max_Step) {
		if (data.param[0]._DataOut->MaxCount != Max_Step)
			data.param[0]._DataOut->Reset(Max_Step);
		//move
		double idx = move.second * 15 + move.first + 1;
		assert(0 < idx&&idx <= 15 * 15);
		data.param[0]._DataOut->SetValue(1, 0, idx, false);
	}
	static Mat _Row;
	static Mat Reward, Randomer;
	static bool GomokuSimulation(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col = -1) {
		Mat&BoardInPut = agent->InPut_Net.front()->GetOutPut(), &BoardOutPut = agent->OutPut_Net.front()->GetOutPut();
		//if (!_Row.IsValid() || _Row.GetCol() != agent->Net_Param["Batch"])
		_Row.Reset(agent->All_In[0].GetRow(), agent->Net_Param["Batch"]), Randomer.Reset(1, agent->Net_Param["Batch"]);

		Mat*OutPutMask = NULL;
		for (int j = 0; j < agent->tot_Net_Num; j++) {
			Net*Mask = NULL;
			//OutPutMask
			if (Mask = agent->Net_Priority[j]->GetMask(Net_Flag_OutPut_Mask)) {
				OutPutMask = &Mask->GetOutPut();
				OutPutMask->_ZeroMemory();
			}
			//Gradient mask
			else if (Mask = agent->Net_Priority[j]->GetMask(Net_Flag_Gradient_Mask)) {
				if (_step + 1 == agent->RNN_Max_Steps) {
					Mask->GetOutPut().f(Device_Func::Assignment, 1.0);
				}
				else Mask->GetOutPut()._ZeroMemory();
			}
		}
		//rotation seed
		//Train Rotation
		if(test_start_col < 0)Randomer.GenerateRandom(8);
		//Test no Rotation
		else Randomer._ZeroMemory();
		//InPut
		if (_step == 0) {
			BoardInPut._ZeroMemory();
			Train_Test(_Row, agent->All_In[0], RandomGenerator, test_start_col);
			//add OutPutMask
			BoardInPut.GomokuSimulation(&agent->All_Out[0], _Row, true, OutPutMask, &Randomer);
		}
		//test
		else {}

		//OutPut
		BoardOutPut._ZeroMemory();
		if (_step + 1 == agent->RNN_Max_Steps) {
			Train_Test(_Row, agent->All_In[0], RandomGenerator, test_start_col);
			//PolicyNet
			//BoardOutPut.GomokuSimulation(agent->All_Out[0], _Row, false);
			//ValueNet
			BoardOutPut.GomokuSimulation(&agent->All_Out[0], _Row, false, NULL, &Randomer);
		}
		return true;
	}
	static bool RL_GomokuSimulation(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col) {
		Mat&BoardInPut = agent->InPut_Net.front()->GetOutPut(), &BoardOutPut = agent->OutPut_Net.front()->GetOutPut();
		//if (!_Row.IsValid() || _Row.GetCol() != agent->Net_Param["Batch"][0])
		_Row.Reset(agent->All_In[0].GetRow(), agent->Net_Param["Batch"]);

		Mat*OutPutMask = NULL;
		for (int j = 0; j < agent->tot_Net_Num; j++) {
			Net*Mask = NULL;
			//OutPutMask
			if (Mask = agent->Net_Priority[j]->GetMask(Net_Flag_OutPut_Mask)) {
				OutPutMask = &Mask->GetOutPut();
				OutPutMask->_ZeroMemory();
			}
			//Gradient mask
			else if (Mask = agent->Net_Priority[j]->GetMask(Net_Flag_Gradient_Mask)) {
				if (_step + 1 == agent->RNN_Max_Steps) {
					Mask->GetOutPut().f(Device_Func::Assignment, 1.0);
				}
				else Mask->GetOutPut()._ZeroMemory();
			}
		}
		//InPut
		if (_step == 0) {
			BoardInPut._ZeroMemory();
			Train_Test(_Row, agent->All_In[0], RandomGenerator, test_start_col);
			//add OutPutMask
			BoardInPut.GomokuSimulation(&agent->All_Out[0], _Row, true, OutPutMask);
		}
		//test
		else {}

		//OutPut
		BoardOutPut._ZeroMemory();
		if (_step + 1 == agent->RNN_Max_Steps) {
			Train_Test(_Row, agent->All_In[0], RandomGenerator, test_start_col);
			BoardOutPut.GomokuSimulation(&agent->All_Out[0], _Row, false, &Reward);
			//Reward
			/*for (auto RL : agent->Reinforce_Net)
				RL->Cal_Status_Reward(_step, _Row, agent->Reward_OutPut_Net->Pre_Net_Head.front());
			agent->Reward_OutPut_Net->GetOutPut() = _Row;*/
			int old_row = _Row.GetRow();
			_Row.ResetRows(1);
			agent->Online_GetReward(_step, _Row);
			_Row.ResetRows(old_row);
		}
		return true;
	}
	static bool MCTS_GomokuSimulation(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col) {
		Mat&BoardInPut = agent->InPut_Net.front()->GetOutPut(), &BoardOutPut = agent->OutPut_Net.front()->Net_Node_Num > 1 ? agent->OutPut_Net.front()->GetOutPut() : agent->OutPut_Net.back()->GetOutPut();
		Mat&ValueOutPut = agent->OutPut_Net.front()->Net_Node_Num == 1 ? agent->OutPut_Net.front()->GetOutPut() : agent->OutPut_Net.back()->GetOutPut();
		//if (!_Row.IsValid() || _Row.GetCol() != agent->Net_Param["Batch"][0])
		_Row.Reset(agent->All_In[0].GetRow(), agent->Net_Param["Batch"]), Randomer.Reset(1, agent->Net_Param["Batch"]);

		Mat*OutPutMask = NULL;
		for (int j = 0; j < agent->tot_Net_Num; j++) {
			Net*Mask = NULL;
			if (Mask = agent->Net_Priority[j]->GetMask(Net_Flag_OutPut_Mask)) {
				OutPutMask = &Mask->GetOutPut();
				OutPutMask->_ZeroMemory();
			}
		}
		//random rotation
		Randomer.GenerateRandom(8);
		//InPut
		BoardInPut._ZeroMemory();
		Train_Test(_Row, agent->All_In[0], RandomGenerator, test_start_col);
		BoardInPut.GomokuSimulation(&agent->All_Out[0], _Row, true, OutPutMask, &Randomer);

		BoardOutPut._ZeroMemory();
		ValueOutPut._ZeroMemory();
		BoardOutPut.GomokuSimulation(&agent->All_Out[0], _Row, false, &Reward, &Randomer, &ValueOutPut);
		return true;
	}
	
	static bool MCTS_Gomoku_Rollout(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col) {
		Mat&BoardInPut = agent->InPut_Net.front()->GetOutPut();
		Mat*OutPutMask = NULL;
		//OutPutMask
		for (int j = 0; j < agent->tot_Net_Num; j++) {
			if (agent->Net_Priority[j]->GetMask(Net_Flag_OutPut_Mask)) {
				OutPutMask = &agent->Net_Priority[j]->GetMask(Net_Flag_OutPut_Mask)->GetOutPut();
				OutPutMask->_ZeroMemory();
				break;
			}
		}
		//InPut
		assert(agent->All_In[0].GetCol() == agent->Net_Param["Batch"]);
		BoardInPut._ZeroMemory();
		BoardInPut.GomokuSimulation(NULL, agent->All_In[0], true, OutPutMask);
		return true;
	}
	static bool MCTS_Direct(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col) {
		Mat&BoardInPut = agent->InPut_Net.front()->GetOutPut();
		assert(agent->InPut_Net.size() == 1);
		//InPut
		assert(agent->All_In[0].GetCol() == agent->Net_Param["Batch"]);
		BoardInPut = agent->All_In[0];
		//BoardInPut._ZeroMemory();
		//BoardInPut.GomokuSimulation(NULL, agent->All_In[0], true, OutPutMask);
		return true;
	}
};


class AgentGenerator {
	HyperParamSearcher param;
	void default_hyperparam_Init() {
		//Random Search param
		//Attention
		param.Random_Uniform("stdev", 0.2, 0.01);
		param.Draw_Geometric("Reward_Baseline_Beta1", 0.001, 0.1, false, true);
		param.Draw_Geometric("Mean_Baseline_Beta1", 0.001, 0.1, false, true);
		//param.Random_Uniform_Step("att_boundary", 1, 400);
		param.Draw_Geometric("Loc_Gradient_decay", 0.01, 1.0);
		//Train hyperparam
		param.Random_Uniform_Step("Batch", 128, 5);
		param.Draw_Geometric("Speed", 0.0001, 0.1);
		param.Draw_Geometric("Speed_decay_time", 5000, 50000, true);
		//param.Draw_Geometric("Max_Epoch", 1000, 2000, true);
		param["Max_Epoch"] = 0.05 * param["Batch"];
		param["EarlyStop"] = 1;
	}
public:
	typedef Net*(*AgentNet)(HyperParamSearcher&);
	typedef double(*Game)(Agent**, int, Agent*, int&);
	typedef void(*Hyperparam)(HyperParamSearcher&);
	AgentGenerator(const string&path = "", const string&best_path="") { 
		param.clear();
		if (path.length() > 0 && best_path.length() > 0)
			param.SetPath(path, best_path);
	}
	//Supervise learning
	void SL_run(int Search_Cnt, AgentNet agentNet_Func, Hyperparam hyperparam_Func, Base_Param** Data,Agent::Train_Option train_func = Agent::Default_Train_Func,Agent::DataWrite2Device Data2Device = Agent::Default_Data2Device) {
		int cnt = Search_Cnt + 1;
		while (cnt-- > 0) {
			default_hyperparam_Init();
			if (cnt <= 0) {
				param.Read_Param();
				param["Max_Epoch"] = 1000000;
				param["EarlyStop"] = 0;
			}
			//User hyperparam
			if (hyperparam_Func)hyperparam_Func(param);
			Agent*agent;
			if (param["Init_Agent"])agent = new Agent("SL_Net", param["Batch"], param["Max_Step"], param["Max_srand_row"], true);
			else agent = new Agent(Agent::RNN, param["Batch"], agentNet_Func(param), param["Max_Step"], true, time(0), param["Max_srand_row"]);
			agent->Online_Init(param);
			param.Compare(agent->Train(Data, param["paramNum"], param["trainNum"], "", NULL, train_func, Data2Device));
			delete agent;
		}
	}
	template<typename Train_Func>
	void RL_run(Agent*agent, int epochs,Hyperparam hyperparam_Func, Base_Param** Data, condition_variable*data_CV,Train_Func train_func = Agent::Default_Train_Func, Agent::DataWrite2Device Data2Device = Agent::Default_Data2Device) {
		param["Max_Epoch"] = epochs;
		param["EarlyStop"] = 0;
		//User hyperparam
		if (hyperparam_Func)hyperparam_Func(param);
		agent->Prase_Param(agent->Net_Param, param);
		agent->Train(Data, param["paramNum"], param["trainNum"], "", data_CV, train_func, Data2Device);
	}
	void MCTS_run(AgentNet agentNet_Func, Hyperparam hyperparam_Func, Judger GameJudger, AgentResponse ResponseFunc, DataSet*player_ds = NULL) {
		param.Read_Param();
		//User hyperparam
		if (hyperparam_Func)
			hyperparam_Func(param);
		const int rollout_Num = 3, rollout_Batch = 8;
		Agent*agent = NULL, **rollout = new Agent*[rollout_Num], **rollout0 = new Agent*[rollout_Num], **rollout1 = new Agent*[rollout_Num];
		//if (param["Init_Agent"])
		int Agent_ID = 0, Maximum_DataSet_Number = 0;
		fstream file; file.open("lastest_Agent_id", ios::in);
		if (file.is_open()) {
			char buf[50]; file.getline(buf, 50);
			sscanf(buf, "%d %d", &Agent_ID, &Maximum_DataSet_Number);
		}file.close();
		char agent_path[50]; sprintf(agent_path, "Agent_#%d", Agent_ID);
		agent = new Agent(agent_path, param["Batch"], param["Max_Step"], param["Max_srand_row"], true);
		//agent = new Agent(Agent::RNN, param["Batch"], agentNet_Func(param), param["Max_Step"], true, time(0), param["Max_srand_row"]);
		agent->Online_Init(param);

		param["Batch"] = rollout_Batch;
		for (int i = 0; i < rollout_Num; i++) {
			rollout[i] = new Agent(agent_path, param["Batch"], param["Max_Step"], param["Max_srand_row"], false);
			//rollout[i] = new Agent(Agent::RNN, param["Batch"], agentNet_Func(param), param["Max_Step"], false, time(0), param["Max_srand_row"]);
			rollout[i]->Online_Init(param);
			//rollout0[i] = new Agent(agent_path, param["Batch"], param["Max_Step"], param["Max_srand_row"], false);
			//rollout0[i]->Online_Init(param);
			rollout1[i] = new Agent("SL_Net", param["Batch"], param["Max_Step"], param["Max_srand_row"], false);
			rollout1[i]->Online_Init(param);
			agent->Data_Assignment(rollout[i]);
		}
		param["Batch"] = agent->Net_Param["Batch"];

		RL_SelfPlay_with_MCTS(GameJudger, agent, rollout_Num, rollout, rollout0, rollout1, ResponseFunc, player_ds, Agent_ID, Maximum_DataSet_Number);
		delete agent;
	}
	//Type=0:generate data,Type=1:Evaluation,Type=3 train agent
	void MCTS_run_extend(int Type, Environment** e, int Generator_Count = 1, bool multiMCTS = false) {
		param.Read_Param();
		int rep_Count = 1;
		int rollout_Num = Type == 0 ? 6 : 3, rollout_Batch = Type == 0 ? 128 : 8;
		Agent* agent = NULL, *** rollout = new Agent ** [4];// , ** rollout1 = new Agent * [rollout_Num];
		int Agent_ID = 0, Maximum_DataSet_Number = 0;
		fstream file; file.open("Ex_lastest_Agent_id", ios::in);
		if (file.is_open()) {
			char buf[50]; file.getline(buf, 50);
			sscanf_s(buf, "%d %d", &Agent_ID, &Maximum_DataSet_Number);
		}file.close();
		char agent_path[50]; sprintf(agent_path, "ExAgent_#%d", Agent_ID);
		param["Max_Step"] = e[0]->getMaxUnrolledStep();
		if (param["Init_Agent"])agent = new Agent(Agent::RNN, param["Batch"], e[0]->JointNet(param), param["Max_Step"], true, time(0), param["Max_srand_row"]);
		else agent = new Agent(agent_path, param["Batch"], param["Max_Step"], param["Max_srand_row"], true);

		agent->Online_Init(param);
		//agent->Write_to_File("FLT_Net");

		if (Type != 2) {
			int Agent_Count = 4;
			for (int t = 0; t < Agent_Count; t++) {
				rollout[t] = new Agent * [rollout_Num];
				param["Max_Step"] = 1;
				for (int i = 0; i < rollout_Num; i++) {
					param["Batch"] = i < rep_Count ? Generator_Count : rollout_Batch;
					Net* net = i < rep_Count ? e[0]->RepresentationNet(param) : e[0]->DynamicsNet(param);
					Net* net1 = i < rep_Count ? e[0]->RepresentationNet(param) : e[0]->DynamicsNet(param);
					rollout[t][i] = new Agent(Agent::RNN, param["Batch"], net, param["Max_Step"], false, time(0), param["Max_srand_row"]);
					rollout[t][i]->Online_Init(param);
				}
				e[0]->JoinetNet_Assignment(rollout[t], agent, rollout_Num);
			}
		}
		if (Type == 1) {
			//Agent* agent1 = new Agent(Agent::RNN, param["Batch"], e[0]->JointNet(param), param["Max_Step"], false, time(0), param["Max_srand_row"]);
			Agent* agent1 = new Agent("SL_Net", param["Batch"], param["Max_Step"], param["Max_srand_row"], false);
			//e[0]->JoinetNet_Assignment(rollout[1], agent1, rollout_Num);
			//e[0]->JoinetNet_Assignment(rollout[0], agent1, rollout_Num);
			//e[0]->JoinetNet_Assignment(rollout[1], agent1, rollout_Num);
			//e[0]->JoinetNet_Assignment(rollout[2], agent1, rollout_Num);
			e[0]->JoinetNet_Assignment(rollout[1], agent1, rollout_Num);
			if (!multiMCTS)
				MCTS_Evaluation(rollout_Num, rollout[0], rollout[1], *e[0], *e[1]);
			else multiMCTS_Evaluation(rollout_Num, rollout, e);
		}
		else if (Type == 0) {
			delete agent;
			if (!multiMCTS)
				RL_SelfPlay_with_MCTS_extend(rollout_Num, rollout[0], Agent_ID, Maximum_DataSet_Number, e, Generator_Count, false);
			else RL_SelfPlay_with_multiMCTS_extend(rollout_Num, rollout[0], Agent_ID, Maximum_DataSet_Number, e, Generator_Count, false);
		}
		else if (Type == 2)MCTS_Train_extend(agent, Agent_ID, Maximum_DataSet_Number, *e[0], Generator_Count, param["paramNum"]);
		else assert(false);
	}
};

struct DataSet {
	LSTM_Param**dataSet = NULL;
	double*Reward = NULL;

	int dataCount = 0;// , file_dataCount = 0;
	int gameCount = 0;// , file_gameCount = 0;
	int MaxCount = 0;

	DataSet() {}
	~DataSet() {
		Disponse();
	}
	void Disponse() {
		for (int i = 0; i < MaxCount; i++)delete dataSet[i], dataSet[i] = NULL;
		delete[] dataSet; dataSet = NULL;
		delete[] Reward; Reward = NULL;
		MaxCount = dataCount = gameCount = 0;
		//file_dataCount = file_gameCount = 0;
	}
	void trainSet_Init(int Max_Count) {
		Disponse();
		dataSet = new LSTM_Param*[Max_Count];
		for (int i = 0; i < Max_Count; i++)dataSet[i] = new LSTM_Param();
		Reward = new double[Max_Count];
		MaxCount = Max_Count;
	}
	int trainSet_dataCount()const { return dataCount; }
	int trainSet_gameCount()const { return gameCount; }
	LSTM_Param& trainSet_Param(int idx) { assert(idx < MaxCount); return *dataSet[idx]; }
	//swap data(move)
	void trainSet_Add_data(const LSTM_Param&data) {
		assert(dataCount < MaxCount);
		LSTM_Param&ds = *dataSet[dataCount];
		ds = data;
		dataCount++;
	}
	void trainSet_Add_Reward(const double&reward) {
		assert(gameCount < MaxCount);
		Reward[gameCount] = reward;
		gameCount++;
	}
	void trainSet_Save_Load(bool Write, int _MaxCount, const char*path = "trainSet_data") {
		fstream file; file.open(path, ios::binary | (Write ? ios::out : ios::in));
		if (file.is_open()) {
			int __MaxCount = MaxCount;
			File_WR(file, &__MaxCount, sizeof(__MaxCount), Write);
			if (!Write)trainSet_Init(__MaxCount);
			assert(__MaxCount <= MaxCount);
			File_WR(file, &dataCount, sizeof(dataCount), Write);
			for (int i = 0; i < dataCount; i++)
				dataSet[i]->WR(file, Write);
			assert(dataCount <= MaxCount);
			File_WR(file, &gameCount, sizeof(gameCount), Write);
			File_WR(file, Reward, sizeof(double)*gameCount, Write);
		}
		else {
			if (!Write)trainSet_Init(_MaxCount); 
			printf("No DataSet found\n");
		}
		//file.flush();
		file.close();
	}
	void Read_From_Memory(const char*data) {
		const char* data_ptr = data;
		if (!data) {
			printf("No DataSet found\n");
		}
		else {
			int __MaxCount = MaxCount;
			ReadFromMemory(data_ptr, &__MaxCount, sizeof(__MaxCount));
			trainSet_Init(__MaxCount);
			assert(__MaxCount <= MaxCount);
			ReadFromMemory(data_ptr, &dataCount, sizeof(dataCount));
			for (int i = 0; i < dataCount; i++)
				dataSet[i]->ReadMem(data_ptr);
			assert(dataCount <= MaxCount);
			ReadFromMemory(data_ptr, &gameCount, sizeof(gameCount));
			ReadFromMemory(data_ptr, Reward, sizeof(double) * gameCount);
		}
	}

	void add_absorbing_Head(Environment&e, int traj_Output_Size, int Value_Num) {
		//add absorbing state
		trainSet_Add_data(Environment::Data(e.getScreen_Size(), traj_Output_Size, 1.0, Value_Num, e));
		trainSet_Add_data(Environment::Data(e.getScreen_Size(), traj_Output_Size, -1.0, Value_Num, e));
		trainSet_Add_data(Environment::Data(e.getScreen_Size(), traj_Output_Size, 0.0, Value_Num, e));
	}
	void complete_absorbing_Head() {
		assert(trainSet_Param(0).Out(0).MaxCount == trainSet_Param(3).Out(0).MaxCount);
		//absorbing
		for (int i = 0; i < trainSet_Param(0).Count; i++) {
			trainSet_Param(0).In(i) = trainSet_Param(3).In(i);
			trainSet_Param(1).In(i) = trainSet_Param(4).In(i);
			trainSet_Param(2).In(i) = trainSet_Param(5).In(i);
		}
	}
	template<typename Train_Func>
	void miniTrain_Start(Agent* agent, condition_variable*data_CV, AgentGenerator::Hyperparam hyperparam_Func, Train_Func train_func) {
		//Gomoku::Reward.Reset(1, gameCount, Reward, true, true, false);
		AgentGenerator RL_Train("Gomoku_param_Rec", "Gomoku_best_param");
		RL_Train.RL_run(agent, 1, hyperparam_Func, (Base_Param**)dataSet, data_CV, train_func);
	}
	void miniTrain_Start(AgentGenerator&ag, AgentGenerator::AgentNet agentNet_Func, AgentGenerator::Hyperparam hyperparam_Func, Agent::Train_Option train_func) {
		//Gomoku::Reward.Reset(1, gameCount, Reward, true, true, false);
		ag.SL_run(0, agentNet_Func, hyperparam_Func, (Base_Param**)dataSet, train_func);
	}
	double Test(Agent*agent, int testNum, Agent::Train_Option train_func,Agent::TestScoreFunc Score) {
		//Gomoku::Reward.Reset(1, gameCount, Reward, true, true, false);
		return agent->Data_Test((Base_Param**)dataSet, testNum, train_func, Agent::Default_Data2Device, Score);
	}
};

const char* ReadFileFromZip(const char* zipfile, const char* filename_in_zip);