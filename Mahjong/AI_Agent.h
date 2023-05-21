//train using MCTS generate data
//input:round state output:action policy

#pragma once
#include"simple.h"
#include"Game.h"

using namespace Simple_Net;

struct Riichi_Agent {
	SimpleNet* sim;
	LSTM_Param** data;
	int Batch;

	Riichi_Agent(const char* param_path) {
		sim = new SimpleNet(AgentNetwork, param_path, true);
		new(this)Riichi_Agent(sim);
	}
	Riichi_Agent(const char* agent_path, const char* param_path) {
		sim = new SimpleNet(agent_path, param_path, true);
		new(this)Riichi_Agent(sim);
	}
	Riichi_Agent(SimpleNet* agent) {
		sim = agent;
		Batch = agent->param["Batch"];
		data = new LSTM_Param * [Batch];
		for (int i = 0; i < Batch; i++)data[i] = new LSTM_Param(1);
	}
	~Riichi_Agent() {
		delete sim;
		for (int i = 0; i < Batch; i++)delete data[i];
		delete[] data;
	}

	//recall in Env round
	void DataEncode(int idx, int Screen_Size, double* input_data, int fillZero) {
		assert(this->Screen_Size == Screen_Size);
		data[idx]->In(0).Reset(Screen_Size);
		data[idx]->In(0).Count = Screen_Size;
		data[idx]->Out(0).Reset(0);
		//ensure pass cuda assert
		//if (fillZero)return;
		//encode input
		for (int i = 0; i < Screen_Size; i++)
			data[idx]->In(0)[i] = input_data[i];
	}
	static void cudaEncode(Agent* agent) {
		Mat& BoardInPut = agent->InPut_Net.front()->GetOutPut();
		assert(agent->InPut_Net.size() == 1);
		assert(agent->All_In[0].GetCol() == agent->Net_Param["Batch"]);
		BoardInPut._ZeroMemory();
		BoardInPut.MahjongAgentRepresentDecode(agent->All_In[0], Image_Depth);
	}

	vector<vector<double>> getAction() {
		sim->feed_and_response((Base_Param**)data, cudaEncode);
		//decode output data
		vector<vector<double>> res;
		for (int i = 0; i < Batch; i++) {
			vector<double>policy;
			for (int j = 0; j < ActionSpace; j++)
				policy.push_back(data[i]->Out(0)[j]);
			res.push_back(policy);
		}return res;
	}

	//void combine_ds(int idx, int ds_number);
	void train(const char* ds_path, int epoches, int Maximum_DataSet_Number);

	static const int W = 34;
	static const int H = 1;


	static const int Image_Depth = 4 * (4 + 4 + 4 + 12) + 25 + 10 + 6 + 22 + 4;
	static const int filters = 64;
	static const int block_Num = 64;
	static const int ActionSpace = 34 * (3 + 9);
	static const int Screen_Size = 4 * (25 + 4 * 4 + 2 + 14) + 6 + 9 + 2 + 10 + 10 + 4 + 4 + 1;
	static Net* AgentNetwork(HyperParamSearcher& param) {
		Net* net = (new Net(W * H * Image_Depth, InPut));
		net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, Image_Depth, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		//residual block
		for (int i = 0; i < block_Num; i++) {
			Net* ori_Image = net;
			net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
			net = (new Net(relu))->Add_Forward(BNTransform, net);
			net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
			net = (new Net(null_Func))->Add_Forward(BNTransform, net);
			Net* Cell = (new Net(net->Net_Node_Num))->Add_Pair_Forward(DotPlus, net, ori_Image);
			net = (new Net(net->Net_Node_Num, relu))->Add_Forward(OpsType::function, Cell);
		}
		net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		net = (new ConvNet(3 + 9, 1, 1, 0, param, null_Func, true))->Add_Forward(net, filters, W);
		net = (new Net(ActionSpace, softmax))->Add_Forward(OpsType::function, net);
		net = (new Net(ActionSpace, CrossEntropy))->Add_Forward(OpsType::OutPut, net);
		return net;
	}
	static bool Train_In_Out_Process(Agent* agent, Mat* RandomGenerator, int _step, int test_start_col) {
		Mat& InPut = agent->InPut_Net.front()->GetOutPut();
		Mat& OutPut = agent->OutPut_Net.front()->GetOutPut();
		Mat In_Sample(agent->All_In[0].GetRow(), agent->Net_Param["Batch"]);
		Mat Out_Sample(agent->All_Out[0].GetRow(), agent->Net_Param["Batch"]);

		Train_Test(In_Sample, agent->All_In[_step], RandomGenerator, test_start_col);
		Train_Test(Out_Sample, agent->All_Out[_step], RandomGenerator, test_start_col);
		//InPut
		if (_step == 0) {
			InPut._ZeroMemory();
			OutPut._ZeroMemory();

			InPut.MahjongAgentRepresentDecode(Out_Sample, Image_Depth);
			OutPut.Mahjong_Simplify_Policy_Encode(Out_Sample, Screen_Size + 2);
		}
		else {}
		return true;
	}
};

DataSet* DataSet_Combine(int l, int r, std::function<string(int)>ds_path_func);