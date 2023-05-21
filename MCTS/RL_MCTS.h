#pragma once
#ifdef MCTS_EXPORTS
#define __MCTS__ __declspec(dllexport)
#else
#define __MCTS__ __declspec(dllimport)

#endif

#define _CRT_SECURE_NO_WARNINGS


#include<iostream>
#include<map>
#include<list>
#include<queue>
#include<assert.h>
#include<set>
#include<random>

#include<thread>
#include<mutex>
#include<condition_variable>

#include<G_Net.h>




#define pi std::pair<int,int>
#define X first
#define Y second
#define ull unsigned long long
#define ll long long
#define us unsigned short
#define ui unsigned int
#define uc unsigned char

using namespace Base_Net;
using namespace std::placeholders;
using namespace _Net_;

typedef struct Position;
typedef struct DataSet;
struct __MCTS__ Environment {
	int step;
	int legal_moves_Count;
	static int HiddenStateSize;
	Environment(int legal, int HiddenSize) :legal_moves_Count(legal),step(0) {
		HiddenStateSize = HiddenSize; 
	}
	virtual ~Environment() {}
	//InPut Encode
	virtual void RepresentEncode(Param* Data, int fillZero) = 0;
	virtual void DynamicEncode(const list<Net*>& InPut_Nets, Param* Data, int idx, Mat* Hidden_State, const ui Action, int fillZero) = 0;
	//virtual void PredictionEncode(Param*Data, const double*Hidden_State, int fillZero) = 0;

	virtual void RepresentDecode(Agent* agent) = 0;
	virtual void DynamicDecode(Agent* agent, int step) = 0;

	virtual void Get_NextState_Reward_And_Policy_Value(const list<Net*>& OutPut_Nets, Param* Data, int idx, Mat* Next_State, double* Reward, double* Policy, double* Value) {}
	virtual void Get_Initial_State_And_Policy_Value(const list<Net*>& OutPut_Nets, Param* Data, int idx, Mat* Next_State, double* Policy, double* Value) = 0;

	virtual void End_Reset(Agent* agent) {
		for (auto& k : agent->OutPut_Net)
			if (k->CostFunction == MeanSquaredError && k->Net_Node_Num == getHiddenStateSize()) {
				k->Pre_Net_Head.front()->GetOutPut().Reset(getHiddenStateSize(), k->Pre_Net_Head.front()->GetOutPut().GetRow());
				break;
			}
	}
	//virtual void Get_NextState_And_Reward(const list<Net*>&OutPut_Nets, Param*Data, double*Next_State, double*Reward) = 0;
	//virtual void Get_Initial_State(const list<Net*>&OutPut_Nets, Param*Data, double*Next_State) = 0;
	//virtual void Get_Policy_And_Value(const list<Net*>&OutPut_Nets, Param*Data, double*Policy, double*Value) = 0;

	//environment API
	typedef std::function < vector<int>(const int&,const int&, vector<int>)> Policy_Func;
	typedef std::function < void(const int&,const int&)> State_Func;
	typedef std::function < void(int*)> GameResult_Func;
	virtual void Environment_Loop(vector<Policy_Func>PolicyFunc, GameResult_Func GameResult_Func, vector<State_Func> StateFuncs, int* Match_result=NULL,DataSet*reward_ds=NULL) {}
	virtual vector<pair<vector<double>, bool>> getRoundsReward() { return {}; }
	virtual vector<vector<double>> getAgentPolicy(int ID) { return {}; }
	//virtual void Environment_Match_Loop(vector<Policy_Func> PolicyFunc, int* result) {}
	virtual void Reset(mt19937* rng = NULL) {}
	virtual void Reset(double* Scr) {}
	virtual void Act(ui Action, double* Reward) {}
	virtual int OpponentsAct(int* Others_Action) { return -1; }
	virtual inline int GetNextPlayerID(int action, int InsPlayerID) { return -1; }
	virtual void GetInsActionMask(bool* ActIsValid) {}
	virtual int GetInsActionMask(bool* ActIsValid, int CheckPlayerID, bool AddHistory) { return -1; }
	//void GetObservation(double*Scr, ui&ScrW, ui&ScrH);
	virtual bool GetGameState(int* result) { return true; }
	virtual void GetGameScreen(double* Scr) = 0;
	virtual void DirectRepresent(double* ScrData) {};

	//extra environment API
	virtual void GetExtraPreActionMask(bool* ActIsValid, int CheckPlayerID, int Type) {}
	virtual void PreAct(ui Action, int Type, int shift) {}
	virtual int getInsCheckPlayerID() { return -1; }
	virtual bool getSpecialActionMask(ui move, ui action_idx,double Prior) { return true; }
	virtual void StochasticEnv() {}
	virtual void ModifyStochasticActionMask(double* Action_Prior, bool* ActionMask) {}

	//Agent Network
	virtual Net* RepresentationNet(HyperParamSearcher& param) = 0;
	virtual Net* DynamicsNet(HyperParamSearcher& param) = 0;
	//virtual Net*PredictionNet(HyperParamSearcher&param) = 0;
	virtual Net* JointNet(HyperParamSearcher& param) = 0;
	//virtual std::function<bool(Agent*, Mat*, int, int/*, Agent**/)> getTrainFun() = 0;
	//virtual std::function<Net*(HyperParamSearcher&)> getJointNetFun() = 0;
	virtual bool Train_In_Out_Process(Agent* agent, Mat* RandomGenerator, int _step, int test_start_col) = 0;
	inline std::function<bool(Agent*, Mat*, int, int/*, Agent**/)> getTrainFun() {
		return std::bind(&Environment::Train_In_Out_Process, this, _1, _2, _3, _4/*, _5*/);
	}
	inline std::function<Net* (HyperParamSearcher&)> getJointNetFun() {
		return std::bind(&Environment::JointNet, this, _1);
	}

	virtual const double getRewardDiscount() = 0;
	virtual const ui getMaxUnrolledStep() = 0;
	//all valid possible action moves
	virtual const ui getRealGameActionSpace() = 0;
	//idx-value pair compressed maximum action move count
	virtual const ui getSimplifyActionSpace() {
		return getRealGameActionSpace();
	}
	virtual const ui getScreen_Size() = 0;
	virtual const ui getHiddenStateSize() = 0;
	virtual const ui getAct(ui move) {
		return move;
	}
	virtual const ui getSimAct(ui move) {
		return move;
	}
	void find_and_swap(Net* tar, Net** stk) {
		for (int i = 0; i < 2; i++) {
			if (*tar == stk[i]) { swap(stk[i], stk[-1]); return; }
		}assert(false);
	}
	void JoinetNet_Assignment(Agent** dst, Agent* JoinetNet, int AgentNum) {
		Net** stk = new Net * [JoinetNet->tot_Net_Num]{ NULL };
		for (int t = 0; t < AgentNum; t++) {
			int cnt = 0, k = 0;
			for (int i = 0; i < JoinetNet->tot_Net_Num; i++)
				if (dynamic_cast<G_Net::TrainUnit*>(JoinetNet->Net_Priority[i]))
					if ((t != 0 && (JoinetNet->Net_Priority[i]->Net_Flag & Net_Flag_RNN_non_Initial_Step)) ||
						(t == 0 && (JoinetNet->Net_Priority[i]->Net_Flag & Net_Flag_RNN_Initial_Step)) ||
						(JoinetNet->Net_Priority[i]->Net_Flag < Net_Flag_RNN_Initial_Step)
						)
						stk[cnt++] = JoinetNet->Net_Priority[i];

			for (int i = 0; i < dst[t]->tot_Net_Num; i++) {
				if (dynamic_cast<G_Net::TrainUnit*>(dst[t]->Net_Priority[i])) {
					//simple swap Net
					if (!(*dst[t]->Net_Priority[i] == stk[k]))find_and_swap(dst[t]->Net_Priority[i], &stk[k + 1]);//swap(stk[k], stk[k + 1]);
					assert(*dst[t]->Net_Priority[i] == stk[k]);
					//for (int j = 0; j < AgentNum; j++)
					stk[k]->Data_Assignment(dst[t]->Net_Priority[i], 1);
					k++;
				}
			}
			assert(k == cnt);
		}delete[]stk;
	}

	//train Data structure
	struct Data :LSTM_Param {
		Data() {}
		Data(Environment* e) :LSTM_Param(e->getMaxUnrolledStep()) {}
		//absorbing states
		Data(int ScrSize, int Action_Space, double Value, int Value_Num, Environment& e) :LSTM_Param(e.getMaxUnrolledStep()) {
			param[0]._DataOut->Reset(ScrSize + Action_Space + 2 + Value_Num);
			for (int i = 0; i < ScrSize; i++)
				param[0]._DataOut->SetValue(1, 0, 0, false);
			//random action move
			param[0]._DataOut->SetValue(1, 0, -1, false);
			//always zero Value
			for (int i = 0; i < Value_Num; i++)
				param[0]._DataOut->SetValue(1, 0, Value, false);
			//always zero Reward
			param[0]._DataOut->SetValue(1, 0, 0, false);
			//Uniform policy
			param[0]._DataOut->SetValue(1, 0, -1, false);
		}
		int f(int i, double Value) {
			if (i % 2 == 0) {
				if (Value == 1)return 0;
				else if (Value == -1)return 1;
				else return 2;
			}
			else {
				if (Value == 1)return 1;
				else if (Value == -1)return 0;
				else return 2;
			}
		}
		//two player train Policy Data
		Data(int ID, int MaxStep, int ScrSize, double* Scr, int Action_Space, double* OutPut, int out_cnt, const ui& move, const double& Value, const double& Reward, Environment& e) :LSTM_Param(e.getMaxUnrolledStep()) {
			for (int i = 0; i < e.getMaxUnrolledStep(); i++) {
				param[i]._DataIn->Reset(1);
				int idx = (ID + i) > (MaxStep - 1) ? f(i, Value) : (ID + i);
				param[i]._DataIn->SetValue(1, 0, idx, false);
			}

			param[0]._DataOut->Reset(ScrSize + Action_Space + 3);
			if (out_cnt > Action_Space)printf("\nPolicy Count exceed\n"), assert(false);
			for (int i = 0; i < ScrSize; i++)
				param[0]._DataOut->SetValue(1, 0, Scr[i], false);
			//action move
			param[0]._DataOut->SetValue(1, 0, move, false);
			//Value
			param[0]._DataOut->SetValue(1, 0, Value, false);
			//Reward
			param[0]._DataOut->SetValue(1, 0, Reward, false);
			//Policy
			for (int i = 0; i < out_cnt; i++)
				param[0]._DataOut->SetValue(1, 0, OutPut[i], false);
		}
		//mulitlp Players train data
		Data(int ID, int*PlayerOrder, int StartID, int MaxStep, int ScrSize, double* Scr, int Action_Space, double* OutPut, int out_cnt, const ui& move, const double* Value, int Value_Num, const double& Reward, Environment& e) :LSTM_Param(e.getMaxUnrolledStep()) {
			int InsPlayerID = -1;
			for (int i = 0; i < e.getMaxUnrolledStep(); i++) {
				param[i]._DataIn->Reset(1 + Value_Num);
				//int idx = (((ID + i) > (MaxStep - 1)) || (ID - StartID < 5 && i >= 5 - (ID - StartID) % 5)) ? 0 : (ID + i);
				int idx = ((ID + i) > (MaxStep - 1)) ? 0 : (ID + i);
				param[i]._DataIn->SetValue(1, 0, idx, false);
				//players related Values
				if (idx != 0)
					InsPlayerID = PlayerOrder[idx - StartID];
				else InsPlayerID = (InsPlayerID + 1) % Value_Num;
				assert(InsPlayerID != -1);
				for (int j = 0; j < Value_Num; j++) {
					double val = Value[(InsPlayerID + j) % Value_Num];
					param[i]._DataIn->SetValue(1, 0, val, false);
				}
				assert(PlayerOrder[ID - StartID] != -1);
			}

			param[0]._DataOut->Reset(ScrSize + Action_Space + 2);
			if (out_cnt > Action_Space)printf("\nPolicy Count exceed\n"), assert(false);
			for (int i = 0; i < ScrSize; i++)
				param[0]._DataOut->SetValue(1, 0, Scr[i], false);
			//action move
			param[0]._DataOut->SetValue(1, 0, move, false);
			//Reward
			param[0]._DataOut->SetValue(1, 0, Reward, false);
			//Policy
			for (int i = 0; i < out_cnt; i++)
				param[0]._DataOut->SetValue(1, 0, OutPut[i], false);
		}
	};

	static void encoder(list<Net*>OutPut_Nets, Base_Param** param, ui Batch, int lstm_step) {
		ui totOut = 0; for (auto& k : OutPut_Nets) {
			//Hidden State Head
			if (k->CostFunction == MeanSquaredError && k->Net_Node_Num == HiddenStateSize)continue;
			totOut += k->Net_Node_Num;
		}
		for (int j = 0; j < Batch; j++)param[j]->Out(lstm_step).Reset(totOut);
		ui offset = 0;
		for (auto& k : OutPut_Nets) {
			//Hidden State Head
			if (k->CostFunction == MeanSquaredError && k->Net_Node_Num == HiddenStateSize)continue;
			floatType* M = k->Pre_Net_Head.front()->GetOutPut().ReadFromDevice(true);
			int Node = k->Net_Node_Num;
			for (int i = 0; i < Node; i++) {
				for (int j = 0; j < Batch; j++)
					param[j]->Out(lstm_step)[i + offset] = M[i * Batch + j];
			}
			for (int j = 0; j < Batch; j++)
				param[j]->Out(lstm_step).Count += Node;
			offset += Node;
		}
	}
	virtual void PrintScr() {}
	virtual int parse_action(const string& move) { return -1; }
	virtual string Encode_action(int simAction) { return 0; }
};
//Agent answer
typedef void(*AgentResponse)(int flag, int idx, int rotation, int**position, int step, double*OutPut, double*Value, Agent*agent);
typedef void(*AgentResponseFunc)(int flag, int idx, double*Screen, const float*Hidden_State, const ui Action, int step, double*OutPut, double*Value, double*Next_State, double*Reward, Agent*agent, Environment*e);
//-1=loss 1=win 2=draw 0=continue
typedef int(*Judger)(int**Position, int step, const pi&InsStone);
void agentResponse(int flag, int idx, int rotation, int**position, int step, double*OutPut, double*Value, Agent*agent);
void agentResponse(int flag, int idx, Mat* Hidden_State, const ui Action, double* OutPut, double* Value, Mat* Next_State, double* Reward, int step, Agent* agent, Environment* e, Param** data);
//void agentResponse(int flag, int idx, Mat* Hidden_State, const ui Action, double* OutPut, double* Value, Mat* Next_State, double* Reward, Agent* agent, Environment* e, Param** data);
int GomokuJudger(int**Position, int step, const pi&InsStone);

void RL_SelfPlay_with_MCTS(Judger GameJudger, Agent*train_agent, int rollout_Num, Agent**rollout_agent, Agent**rollout_agent0, Agent**rollout_agent1, AgentResponse ResponseFunc, DataSet*player_ds, int agent_id, int Maximum_DataSet_Number);
__MCTS__ void RL_SelfPlay_with_MCTS_extend(int rollout_Num, Agent** rollout_agent, int agent_id, int Maximum_DataSet_Number, Environment** e, int Generator_Count, bool Reanalyze);
__MCTS__ void MCTS_Evaluation(int rollout_Num, Agent**rollout_agent, Agent**rollout_agent1, Environment&e1, Environment&e2);
__MCTS__ void MCTS_Train_extend(Agent*agent, int agent_id, int Maximum_DataSet_Number, Environment&e, int Generator_Count,int TrainDataNum);

__MCTS__ void RL_SelfPlay_with_multiMCTS_extend(int rollout_Num, Agent** rollout_agent, int agent_id, int Maximum_DataSet_Number, Environment** e, int Generator_Count, bool Reanalyze);
__MCTS__ void multiMCTS_Evaluation(int rollout_Num, Agent*** rollout_agent, Environment** e);
//return 30-bits random unsigned number
//all bits valid
static int rand_i() {
	return (rand() << 15) + rand();
}
void Sampling_DataSet(DataSet&train_ds, int data_Count, int recent_Max_data, int dataSet_Max_ID);
__MCTS__ int pipeOut(const char *fmt, ...);


//__MCTS__ void MCTS_Agent_Init(Agent**agents, int agent_Num, int Thr_Num, int Sims, const char*dir_path, const char*name = "0");
//__MCTS__ pi MCTS_Agent_Response(int**board, pi*moves, int moves_cnt);
//__MCTS__ void MCTS_Agent_New_Game();
//__MCTS__ void MCTS_Agent_End();
struct __declspec(dllexport) Agent_API {
	virtual void MCTS_Agent_Init(const char*agent_Path, const char*agent_param_Path, const char*dir_path, Environment&e) = 0;
	virtual pi MCTS_Agent_Response(pi*moves, int moves_cnt) { return { 0,0 }; }
	virtual string MCTS_Agent_Response(string*moves, int moves_cnt) { return ""; }
	virtual void MCTS_Agent_New_Game() = 0;
	virtual void MCTS_Agent_End() = 0;
	virtual void MCTS_Agent_Run() {}
};
__MCTS__ Agent_API* Get_MCTS_Extend_API();
__MCTS__ Agent_API* Get_multiMCTS_Extend_API();
__MCTS__ extern Param**game_data[Cuda_Max_Stream];

