#define Net_EXPORTS

#include"G_Net.h"
#include"RNN_LSTM.h"
#include"Attention.h"
#include"ConvNet.h"
#include<vector>
#include<type_traits>

#include<mutex>

using namespace Base_Net;
using namespace _Net_;

map<Net*, bool>Nets;
list<Net*>_Nets;
map<Net*, int>Q;
map<int, vector<Net*>>M;
void dfs(Net*Ins,int deep) {
	if (!Ins)return;
	if (Q.find(Ins) != Q.end() && deep <= Q[Ins])return;
	Q[Ins] = deep;
	for (auto k : Ins->Next_Net_Head) {
		if(k->Reconnect_Net.find(Ins)!=k->Reconnect_Net.end())
			continue;
		dfs(k, deep + 1);
	}
}
void dfs1(Net*Ins) {
	if (!Ins)return;
	if (Nets.find(Ins) != Nets.end())return;
	Nets[Ins] = true;
	_Nets.push_back(Ins);
	for (auto k : Ins->Pre_Net_Head) {
		dfs1(k);
	}
	for (auto k : Ins->Next_Net_Head) {
		dfs1(k);
	}
}
mutex agent_mux;
Agent::Agent(Net_Type Type,int Batch, Net*Start_Net, int RNN_Max_Steps, bool Reset_All_Cuda_States, time_t random_seed, int Max_srand_row, bool ReadFile)
	:RNN_Max_Steps(RNN_Max_Steps) {
	lock_guard<mutex>ds_locker(agent_mux);
	srand(random_seed);
	//Parse Net Struction
	Nets.clear(); _Nets.clear(); Q.clear(); M.clear();
	//search all Net Node
	dfs1(Start_Net);
	
	for (auto k : _Nets) {
		if (k->Ops == InPut && !(k->Net_Flag&Net_Flag_Factor))
			InPut_Net.push_back(k);
	}
	//only one Input with special flag
	Net*In = NULL; for (auto&k : InPut_Net)
		if (!In || k->Net_Node_Num < In->Net_Node_Num || (k->Net_Node_Num == In->Net_Node_Num&&(k->Net_Flag&Net_Flag_RNN_Initial_Step)))In = k;
	Nets.clear(); _Nets.clear();
	//ensure Only one net structure
	dfs1(In);
	//Completed Before All Functioins
	if (Reset_All_Cuda_States) {
		int Max_Node_Num = 1;
		for (auto k : _Nets) {
			if (k->Net_Flag&(Net_Flag_DropOut|Net_Flag_Reinforce))
				Max_Node_Num = max((int)k->Net_Node_Num, Max_Node_Num);
			if (k->Net_Flag&Net_Flag_Reinforce)
				Max_Node_Num = max(Batch, Max_Node_Num);
		}
		//Reset All Cuda States(Cautious)
		//Use Before All Cuda Func
		if (Max_srand_row != -1)
			Max_Node_Num = Max_srand_row;
		Func_Init(Max_Node_Num);
	}
	//Init Net Connect
	NT = Type;
	Net_Param.clear();
	Default_Param();
	All_In = new Mat[RNN_Max_Steps];
	All_Out = new Mat[RNN_Max_Steps];
	Net*_Ins_Net = NULL, *Ins_Net = NULL;
	for (auto k : _Nets) {
		Ins_Net = k;
		int Pre_Node_Num = 0;
		for (auto k : Ins_Net->Pre_Net_Head)
			Pre_Node_Num += k->Net_Node_Num;
		switch (NT)
		{
		case Full_Connected:
			if (Ins_Net->Ops == CustomOps) {
				_Ins_Net = Ins_Net;
			}
			else if (Ins_Net->Ops == Transform)
				_Ins_Net = new G_Net::TrainUnit(Ins_Net->Net_Node_Num, Pre_Node_Num, Batch);
			else _Ins_Net = new G_Net::InPutUnit(Ins_Net->Net_Node_Num, Batch);
			break;
		case RNN:
			if (Ins_Net->Ops == CustomOps) {
				_Ins_Net = Ins_Net;
			}
			else if (Ins_Net->Ops == Transform)
				_Ins_Net = new RNN_LSTM::LSTMUnit(RNN_Max_Steps, Ins_Net->Net_Node_Num, Pre_Node_Num, Batch);
			else if (Ins_Net->Ops == BNTransform) {
				Net*Pre = Ins_Net->Pre_Net_Head.front();
				Conv*cv = dynamic_cast<Conv*>(Pre);
				int BN_Node;
				if (cv)BN_Node = cv->Filters;
				else BN_Node = Pre->Net_Node_Num;
				_Ins_Net = new RNN_LSTM::LSTMUnit(RNN_Max_Steps, BN_Node, 1, Batch, -1, true);
			}
			else _Ins_Net = new RNN_LSTM::LSTMInput(RNN_Max_Steps, Ins_Net->Net_Node_Num, Batch);
			break;
		default:
			assert(false);
			break;
		}
		//Replace
		if (Ins_Net->Ops != CustomOps) {
			*_Ins_Net = *Ins_Net;
			for (auto k : _Ins_Net->Pre_Net_Head) {
				for (auto&g : k->Next_Net_Head) {
					if (g == Ins_Net) {
						g = _Ins_Net;
					}
				}
			}
			for (auto k : _Ins_Net->Next_Net_Head) {
				for (auto&g : k->Pre_Net_Head) {
					if (g == Ins_Net) {
						g = _Ins_Net;
					}
				}
			}
			//Reconnect
			for (auto k : _Ins_Net->Next_Net_Head) {
				if (k->Reconnect_Net.find(Ins_Net) != k->Reconnect_Net.end()) {
					k->Reconnect_Net.erase(Ins_Net);
					k->Reconnect_Net[_Ins_Net] = true;
				}
			}
			delete Ins_Net;
		}
	}
	Nets.clear(); _Nets.clear();
	//search all Net Node
	dfs1(_Ins_Net);
	//set Net Deep
	for (auto k : _Nets) {
		dfs(k, 0);
	}
	for (auto k : _Nets) {
		M[Q[k]].push_back(k);
	}
	//Net Order Array,InPut,OutPut
	tot_Net_Num = Nets.size();
	if (!ReadFile)Net_Priority = new Net*[tot_Net_Num];
	int cnt = 0;
	OutPut_Net.clear(); Reinforce_Net.clear(); InPut_Net.clear();
	for (auto k : M) {
		for (auto g : k.second) {
			Net_Priority[cnt++] = g;
			if (g->Ops == InPut && !(g->Net_Flag&Net_Flag_Factor))
				InPut_Net.push_back(g);
			else if (g->Ops == OutPut) {
				OutPut_Net.push_back(g);
			}
			//Reinforcement learning Net
			if (g->Net_Flag&Net_Flag_Reinforce)
				Reinforce_Net.push_back(g);
		}
	}assert(cnt == tot_Net_Num);
	//find closest OutPut Func
	for (auto&k : OutPut_Net) {
		Ins_Net = k;
		if (k->CostFunction == null_Cost_Func)
			while (!Ins_Net->Pre_Net_Head.empty()) {
				Net*p = Ins_Net->Pre_Net_Head.front();
				//reinforce
				if (p->Net_Flag&Net_Flag_Reinforce) {
					k->CostFunction = null_Cost_Func;
					break;
				}
				else if (p->Func) {
					if (p->Func == softmax)
						k->CostFunction = CrossEntropy;
					else k->CostFunction = MeanSquaredError;
					break;
				}
				Ins_Net = p;
			}
	}
}
//RNN Based Net
template<typename Train_Func>
double Agent::Train(Base_Param**param, int paramNum, int trainNum, const char*Train_Param, condition_variable*data_CV,Train_Func train_func,DataWrite2Device data_proc_func) {
	Init_Param(Train_Param);
	data_proc_func(param, paramNum, this, false);
	//tell prepare data
	if (data_CV) {
		//lock_guard<mutex>locker(data_mux);
		printf("data written\n");
		data_CV->notify_one();
	}
	int Batch = Net_Param["Batch"];
	int _trainNum = trainNum / Batch * Batch, _testNum = (All_In[0].GetCol() - _trainNum) / Batch * Batch;
	int tot_Train_Output_Num = _trainNum, tot_Test_Output_Num = _testNum;
	int tot_Train_Positive_Num = 0, tot_Test_Positive_Num = 0;

	Mat*Reward = new Mat[RNN_Max_Steps], _Reward(1, Batch);
	Learning_rate Speeder(Net_Param["Speed"], Net_Param["Speed_decay_time"]);
	Mat RandomGenerator(1, Batch);
	double Last_Test = 0, Best_Test = 0;
	ui BatchCountPerEpoch = max((int)(_trainNum / Batch * Net_Param["ExtraTrainFactorPerEpoch"]), 1);
	BatchTrainCount = 0;
	clock_t Start = clock();
	double*Test_Best_Table = new double[(int)Net_Param["Max_Epoch"] + 10]{ 0 };
	bool RewardFromSL = Net_Param["Reward_From_SL"];
	bool Stochastic = Net_Param["Stochastic"];
	double tot_Loss = 0, Last_Loss = 0, min_Loss = 1e18;

	//temp
	fstream file; file.open("loss_data.txt", ios::out|ios::app);

	while (BatchTrainCount / BatchCountPerEpoch < Net_Param["Max_Epoch"]) {
		tot_Loss = 0;
		//epoch
		while (true) {
			//Cell State, Hidden State zero memory
			RNN_Init();
			lstm_Step = -1;
			double Loss = 0.0, L2_Loss = 0, MSE_Loss = 0;
			//train example random order
			if (Stochastic) {
				RandomGenerator.GenerateRandom(_trainNum);
			}
			for (int i = 0; i < RNN_Max_Steps; i++) {
				if (!train_func(this, &RandomGenerator, i, Stochastic ? -1 : (BatchTrainCount*Batch)))
					break;
				//Step DropOut
				DropOut(true);
				//Feed forward&Cal Loss
				Loss += Forward((BatchTrainCount % BatchCountPerEpoch + G_Net::TrainUnit::BN_avg_miniBatch_Num) >= BatchCountPerEpoch ? 2 : true, i, true, true, &L2_Loss, &MSE_Loss);
				assert(!isnan(Loss));
				//get OutPut cor Reward
				//if (RewardFromSL)
					//OutPut_Net.front()->Test(NULL, &Reward[i]);
				lstm_Step++;
			}
			//cal total reward
			/*if (RewardFromSL) {
				_Reward._ZeroMemory();
				for (int i = lstm_Step; i > -1; i--) {
					_Reward += Reward[i];
					Online_GetReward(i, _Reward);
				}
			}*/
			if (lstm_Step > -1) {
				//Backward
				for (int i = lstm_Step; i > -1; i--)
					Backward(i, lstm_Step + 1);
				//Update Weigh
				if (!Net_Param["EpochUpdate"] || (BatchTrainCount + 1) % BatchCountPerEpoch == 0)
					Update();
				//Log OutPut
				printf("Loss:%f L2_Loss:%.02lf Count:%d epoch:%d last:%.02f best:%.02f Speed:%f Time:%d min(s)\n", Loss, L2_Loss, BatchTrainCount, BatchTrainCount / BatchCountPerEpoch, Last_Test, Best_Test, Net_Param["Speed"], (clock() - Start) / 1000 / 60);

				if (BatchTrainCount % (BatchCountPerEpoch/10) == 0) {
					//temp
					char tmp[100];
					sprintf(tmp, "%.02f %.02f", Loss / Batch, MSE_Loss / Batch);
					string str = tmp;
					str += "\n";
					file.write(str.c_str(), str.length());
				}
				
				tot_Loss += Loss;
				//Speed decay
				Speeder.Step_Decay(++BatchTrainCount / BatchCountPerEpoch, Net_Param["Speed"]);
				if (BatchTrainCount % BatchCountPerEpoch == 0) {
					break;
				}
			}
		}
		
		min_Loss = min(min_Loss, tot_Loss);
		printf("tot_Loss:%0.2lf Last_Loss:%0.2lf Min_Loss:%0.2lf\n", tot_Loss, Last_Loss, min_Loss);
		Last_Loss = tot_Loss;
		Last_Test = Test(tot_Test_Output_Num, tot_Test_Positive_Num, _trainNum, _testNum / Batch* Net_Param["ExtraTestFactorPerEpoch"], train_func);
		Best_Test = max(Best_Test, Last_Test);
		ui epoch = BatchTrainCount / BatchCountPerEpoch;
		Test_Best_Table[epoch] = max(Test_Best_Table[epoch - 1], Best_Test);
		//early Stop
		if (Last_Test <= Test_Best_Table[epoch / 2]&&Net_Param["EarlyStop"]) {
			break;
		}
		if (epoch % 20 == 0) {
			double res = Test(tot_Train_Output_Num, tot_Test_Positive_Num, 0, _trainNum / Batch, train_func);
			printf("Train Cor:%.02f%%\n", res);
		}

		if (_kbhit()) {
			int get = _getch();
			if (get == (int)'c')
				break;
			else if (get == (int)'w') {
				Write_to_File("SL_Net");
			}
		}
	}
	file.close();
	delete[] Test_Best_Table;
	delete[] Reward;
	return Best_Test;
}
void Agent::DropOut(bool IsTrain, bool Used) {
	if (IsTrain)
		for (int i = 0; i < tot_Net_Num; i++)
			if (Net_Priority[i]->Net_Flag&Net_Flag_Factor) {
				Net*nxt = Net_Priority[i]->Next_Net_Head.front();
				if (nxt->Net_Flag&Net_Flag_DropOut)
					if (!Used)
						Net_Priority[i]->GetOutPut().f(Assignment, 1.0);
					else if (nxt->Net_Flag&Net_Flag_InPut_DropOut)
						Net_Priority[i]->GetOutPut().f(DropOut_Bernoulli, Net_Param["InPut_DropOut"]);
					else if (nxt->Net_Flag&Net_Flag_Hidden_DropOut)
						Net_Priority[i]->GetOutPut().f(DropOut_Bernoulli, Net_Param["Hidden_DropOut"]);
					else if (nxt->Net_Flag&Net_Flag_OutPut_DropOut)
						Net_Priority[i]->GetOutPut().f(DropOut_Bernoulli, Net_Param["OutPut_DropOut"]);
			}
}
double Agent::Forward(int IsTrain, int Step, bool backward, bool cal_loss, double*L2_Loss, double* MSE_Loss) {
	double Sum = 0;
	for (int i = 0; i < tot_Net_Num; i++) {
		if ((Net_Priority[i]->Net_Flag&Net_Flag_RNN_Initial_Step) && Step > 0)continue;
		if ((Net_Priority[i]->Net_Flag&Net_Flag_RNN_non_Initial_Step) && Step == 0)continue;

		double loss = Net_Priority[i]->Forward(Step, IsTrain, cal_loss);
		Sum += loss;
		assert(!isnan(Sum));
		//summary L2 loss
		if (L2_Loss&&dynamic_cast<G_Net::TrainUnit*>(Net_Priority[i])) {
			auto net = ((G_Net::TrainUnit*)Net_Priority[i]);
			*L2_Loss += (net->C_Weigh._f(Pow2).Sum().ReadFromDevice(true)[0] + net->C_b._f(Pow2).Sum().ReadFromDevice(true)[0]);
		}
		//temp
		//summary MSE loss
		if (Net_Priority[i]->Ops == OpsType::OutPut && Net_Priority[i]->CostFunction == CostFunc::MeanSquaredError) {
			*MSE_Loss += loss;
		}


		if (backward)
			Net_Priority[i]->LSTM_Save(Step);
	}
	if (L2_Loss)*L2_Loss *= Net_Param["L2_Factor"], Sum += *L2_Loss;
	return Sum;
}
//Cal Reinforce Reward in Train
void Agent::Reinforce_Reward(int Step, Net*Reward_Net, double*Step_Reward, int Ins_Max_Step, int Type) {
	for (auto RL : Reinforce_Net)
		RL->Cal_TD_Reward(Step, Reward_Net, Step_Reward, Ins_Max_Step, Type);
}
void Agent::Backward(int Step,int RNN_Max_Steps) {
	for (int i = tot_Net_Num - 1; i > -1; i--) {
		if (Net_Priority[i]->Ops == InPut || Net_Priority[i]->Ops == OutPut)continue;
		if ((Net_Priority[i]->Net_Flag&Net_Flag_RNN_Initial_Step) && Step > 0)continue;
		if ((Net_Priority[i]->Net_Flag&Net_Flag_RNN_non_Initial_Step) && Step == 0)continue;

		Net_Priority[i]->Backward(Step, (RNN_Max_Steps == -1) ? this->RNN_Max_Steps : RNN_Max_Steps);
	}
}
void Agent::Update(Agent*Server) {
	Net_Param["Beta1_Pow"] *= Net_Param["Beta1"];
	Net_Param["Beta2_Pow"] *= Net_Param["Beta2"];
	for (int i = 0; i < tot_Net_Num; i++) {
		Net_Priority[i]->Update(Net_Param, Server ? Server->Net_Priority[i] : NULL);
	}
}
void Agent::getGradient(Agent*Worker, double Scale) {
	for (int i = 0; i < tot_Net_Num; i++) {
		if (i < Worker->tot_Net_Num) {
			G_Net::TrainUnit*dst = NULL, *src = NULL;
			if ((dst = dynamic_cast<G_Net::TrainUnit*>(Net_Priority[i])) && (src = dynamic_cast<G_Net::TrainUnit*>(Worker->Net_Priority[i]))) {
				dst->C_Weigh += src->C_BiasWeigh * Scale;
				dst->C_b += src->C_Biasb * Scale;
				src->C_BiasWeigh._ZeroMemory();
				src->C_Biasb._ZeroMemory();
			}
			else assert(!dst && !src);
		}
	}
}
template<typename Train_Func>
double Agent::Test(int OutPut_Num, int Positive_Num,int trainNum, int Batchs,Train_Func func, TestScoreFunc Score) {
	if (OutPut_Num <= 0)return printf("no Test data\n"), -1;
	int Batch = Net_Param["Batch"];
	int _MoveCnt = Net_Param["ExtraTestFactorPerEpoch"];
	double cor = 0, tot = OutPut_Num / Batch * Batch, posi = 0, prec = 0, Loss = 0;
	vector<double>part_Loss; part_Loss.resize(OutPut_Net.size(), 0.0);
	for (int t = 0; t < Batchs; t++) {
		RNN_Init();
		for (int i = 0; i < RNN_Max_Steps; i++) {
			if (!func(this, NULL, i, trainNum + t * Batch / _MoveCnt)) {
				t = (t / _MoveCnt + 1)*_MoveCnt - 1;
				break;
			}
			Loss += Forward(false, i, false);
			int idx = 0;
			for (auto&k : OutPut_Net) {
				k->Test(&cor, NULL, &part_Loss[idx++]);
			}
		}
	}
	printf("Test:%0.2lf%% Loss:%0.2lf Part:", 100.0*cor / tot, Loss);
	for (auto&k : part_Loss)printf(" %0.2lf %0.4lf", k, k*2.0 / OutPut_Num / 4.0);
	printf("\n");
	if (Score)return Score(100.0*cor / tot, part_Loss, OutPut_Num);
	else return 100.0*cor / tot;
}

template __Net__ double Agent::Train<Agent::Train_Option>(Base_Param**, int, int, const char*, condition_variable*,Agent::Train_Option, Agent::DataWrite2Device);
template __Net__ double Agent::Train<std::function<bool(Agent*, Mat*, int, int)>>(Base_Param**, int, int, const char*, condition_variable*,std::function<bool(Agent*, Mat*, int, int)>, Agent::DataWrite2Device);
template __Net__ double Agent::Test<Agent::Train_Option>(int, int, int, int, Agent::Train_Option, TestScoreFunc);

double Agent::Data_Test(Base_Param**param, int paramNum, Train_Option train_func, DataWrite2Device data_proc_func,TestScoreFunc Score) {
	data_proc_func(param, paramNum, this, false);
	int Batch = Net_Param["Batch"];
	int _testNum = paramNum / Batch * Batch;
	int tot_Test_Output_Num = _testNum;

	return Test(tot_Test_Output_Num, 0, 0, _testNum / Batch, train_func, Score);
}
bool Agent::Default_Train_Func(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col) {
	Train_Test(agent->InPut_Net.front()->GetOutPut(), agent->All_In[_step], RandomGenerator, test_start_col);
	Train_Test(agent->OutPut_Net.front()->GetOutPut(), agent->All_Out[_step], RandomGenerator, test_start_col);
	for (int j = 0; j < agent->tot_Net_Num; j++)
		//Gradient mask
		if (agent->Net_Priority[j]->Net_Flag&Net_Flag_Gradient_Mask) {
			if (_step + 1 == agent->RNN_Max_Steps) {
				agent->Net_Priority[j]->Pre_Net_Head.back()->GetOutPut().f(Device_Func::Assignment, 1.0);
			}
			else agent->Net_Priority[j]->Pre_Net_Head.back()->GetOutPut()._ZeroMemory();
		}
	return true;
}
void Agent::Default_Data2Device(Base_Param**data, int paramNum, Agent*agent,bool W_Optimize) {
	int InPutNum = data[0]->In(0).MaxCount;
	int OutPutNum = data[0]->Out(0).MaxCount;
	int Data_Max_Steps = data[0]->LSTM_Count();
	assert(Data_Max_Steps == agent->RNN_Max_Steps);
	for (int i = 0; i < Data_Max_Steps; i++) {
		int In_Num = 0, Out_Num = 0;
		ui In_Max_Row = 0, Out_Max_Row = 0;
		for (int l = 0; l < paramNum; l++) if (i < data[l]->LSTM_Count() && data[l]->In(i).Count>0)In_Num++; else break;
		for (int l = 0; l < paramNum; l++) if (i < data[l]->LSTM_Count() && data[l]->Out(i).Count>0)Out_Num++; else break;
		if (In_Num > 0)agent->All_In[i].Reset(InPutNum, In_Num, false, W_Optimize, false);
		if (Out_Num > 0)agent->All_Out[i].Reset(OutPutNum, Out_Num, false, W_Optimize, false);
		for (int l = 0; l < In_Num; l++) {
			In_Max_Row = max(In_Max_Row, data[l]->In(i).Count);
			for (int k = 0; k < InPutNum; k++)
				if (i < data[l]->LSTM_Count() && k < data[l]->In(i).Count)
					agent->All_In[i][k*In_Num + l] = data[l]->In(i)[k];
				else agent->All_In[i][k*In_Num + l] = 0;
		}
		for (int l = 0; l < Out_Num; l++) {
			Out_Max_Row = max(Out_Max_Row, data[l]->Out(i).Count);
			for (int k = 0; k < OutPutNum; k++)
				if (i < data[l]->LSTM_Count()&&k < data[l]->Out(i).Count)
					agent->All_Out[i][k*Out_Num + l] = data[l]->Out(i)[k];
				else agent->All_Out[i][k*Out_Num + l] = 0;
		}
		if (In_Num > 0)agent->All_In[i].WriteToDevice(W_Optimize ? In_Max_Row : -1);
		if (Out_Num > 0)agent->All_Out[i].WriteToDevice(W_Optimize ? Out_Max_Row : -1);
	}
}
class IBase {
public:
	virtual Net*Create() = 0;
	virtual const std::type_info& type_of_T() = 0;
};
template<class T>
class Custom:public IBase {
public:
	T*Create() {return new T();}
	const std::type_info& type_of_T() { return typeid(T); }
};
IBase*NetStack[] = { new Custom<Attention::REINFORCE>(),new Custom<Attention::Glimpse>(),new Custom<ConvNet>(),new Custom<POOL>() };
//Data WR
Net* Agent::Net_WR(fstream&file, int Batch, int RNN_Max_Steps, bool Write) {
	File_WR(file, (char*)&NT, sizeof(Net_Type), Write);
	if (!Write&&file.gcount() != sizeof(Net_Type)) {
		return NULL;
	}
	//Write Net Structure
	File_WR(file, (char*)&tot_Net_Num, sizeof(int), Write);
	//temp Net
	if (!Write) {
		Net_Priority = new Net*[tot_Net_Num];
		for (int i = 0; i < tot_Net_Num; i++)
			Net_Priority[i] = new Net();
	}
	//Custom Net Write&Read
	int sz = 0; vector<int>ID; ID.clear();
	for (int i = 0; i < tot_Net_Num; i++)
		if (Net_Priority[i]->Ops == CustomOps)sz++, ID.push_back(i);
	File_WR(file, (char*)&sz, sizeof(int), Write);
	for (int i = 0; i < sz; i++) {
		int idx = (i < ID.size()) ? ID[i] : -1;
		File_WR(file, (char*)&idx, sizeof(int), Write);
		//Type
		int TypeID = -1;
		int sz = sizeof(NetStack) / sizeof(IBase*);
		for (int j = 0; j < sz; j++)
			if (typeid(*Net_Priority[idx]) == NetStack[j]->type_of_T()) { TypeID = j; break; }
		File_WR(file, (char*)&TypeID, sizeof(int), Write);
		if (!Write) {
			assert(TypeID > -1);
			delete Net_Priority[idx];
			Net_Priority[idx] = NetStack[TypeID]->Create();
		}
		//recall custom net initalize
		Net_Priority[idx]->Data_WR(file, Net_Priority, tot_Net_Num, Write, Batch, RNN_Max_Steps);
	}
	//Base Net Info
	for (int i = 0; i < tot_Net_Num; i++) {
		Net_Priority[i]->Net::Data_WR(file, Net_Priority, tot_Net_Num, Write);
	}
	return Net_Priority[0];
}
void Agent::NetData_WR(fstream&file, bool Write) {
	for (int i = 0; i < tot_Net_Num; i++) {
		if (dynamic_cast<G_Net::TrainUnit*>(Net_Priority[i]))
			((G_Net::TrainUnit*)Net_Priority[i])->G_Net::TrainUnit::Data_WR(file, Net_Priority, tot_Net_Num, Write, 0, RNN_Max_Steps);
	}
}
void Agent::Data_Assignment(Agent*dst_Agent) {
	for (int i = 0; i < tot_Net_Num; i++) {
		if (i < dst_Agent->tot_Net_Num)
			if (dynamic_cast<G_Net::TrainUnit*>(Net_Priority[i]) && dynamic_cast<G_Net::TrainUnit*>(dst_Agent->Net_Priority[i]))
				Net_Priority[i]->Data_Assignment(dst_Agent->Net_Priority[i], RNN_Max_Steps);
			else assert(!dynamic_cast<G_Net::TrainUnit*>(Net_Priority[i]) && !dynamic_cast<G_Net::TrainUnit*>(dst_Agent->Net_Priority[i]));
	}
}

bool Base_Net::cmp(Net*&a, Net*&b) {
	return a == b;
}
template<class T>
bool Base_Net::cmp(T&it, Net*&b) {
	return it.first == b;
}
template<class T>
void Base_Net::Insert(list<T>&ls, const T&val) {
	ls.push_back(val);
}
template<class T, class T1>
void Base_Net::Insert(map<T, T1>&mp, const T&val) {
	mp[val] = true;
}
template<class T>
void Base_Net::iter_WR(T&iter, fstream&file, Net**Net_Priority, int tot_Net_Num, bool W) {
	int sz = iter.size();
	File_WR(file, (char*)&sz, sizeof(int), W);
	if (W)
		for (auto k : iter) {
			bool Write = false;
			for (int i = 0; i < tot_Net_Num; i++)
				if (cmp(k, Net_Priority[i]))
				{
					file.write((char*)&i, sizeof(int));
					Write = true;
					break;
				}
			if (!Write) {
				printf("Net Data Write Error\n"); assert(false);
			}
		}
	else
		for (int i = 0; i < sz; i++) {
			int idx = -1;
			file.read((char*)&idx, sizeof(int));
			assert(idx > -1);
			Insert(iter, Net_Priority[idx]);
		}
}
template void Base_Net::iter_WR < list < Net* >>(list < Net* >& iter, fstream& file, Net** Net_Priority, int tot_Net_Num, bool W);
template void Base_Net::iter_WR < map<Net*, bool>>(map<Net*, bool>& iter, fstream& file, Net** Net_Priority, int tot_Net_Num, bool W);
void Base_Net::Param_WR(fstream&file, NetParam&Net_Param, HyperParamSearcher&param, bool Write) {
	int sz = Net_Param.size();
	File_WR(file, (char*)&sz, sizeof(int), Write);
	auto it = Net_Param.begin();
	for (int i = 0; i < sz; i++) {
		char data[100] = { 0 }; double val = 0;
		if (it != Net_Param.end()) {
			strcpy(data, it->first.c_str());
			//if (it->second.IsValid())
			val = it->second;
			//else val = -1;
			it++;
		}
		int len = strlen(data);
		File_WR(file, (char*)&len, sizeof(int), Write);
		File_WR(file, (char*)data, sizeof(char)*len, Write);
		File_WR(file, (char*)&val, sizeof(double), Write);
		param[data] = val;
	}
}
void Base_Net::Train_Test(Mat&Step_Data, Mat&Data, Mat*RandomGenerator, int test_start_col) {
	//train
	if (test_start_col<0)
		Step_Data.RandMatrix(Data, *RandomGenerator);
	//test
	else Step_Data.Append(Data, 0, test_start_col);
}