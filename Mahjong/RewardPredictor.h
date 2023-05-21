#pragma once
#include"simple.h"
#include"Game.h"

using namespace Simple_Net;

struct Round_Info {
	int round_wind_number;
	vector<int>Scores;
	int riichi_sticks, honba_sticks;
	Round_Info() {}
	Round_Info(int round, int* Score, int riichi, int honba) {
		round_wind_number = round;
		Scores = vector<int>(Score, Score + 4);
		honba_sticks = honba;
		riichi_sticks = riichi;
	}
	static const int input_data_size = 7 * 2;
};
struct RewardData :LSTM_Param {
	RewardData(Round_Info& Start, Round_Info& End,const vector<int>& final_Scores) :LSTM_Param(1) {
		In(0).Reset(Round_Info::input_data_size);
		In(0).Count = In(0).MaxCount;
		Out(0).Reset(4);
		Out(0).Count = Out(0).MaxCount;
		int idx = 0;
		for (int t = 0; t < 2; t++) {
			const Round_Info& it = t == 0 ? Start : End;
			In(0)[idx++] = it.round_wind_number;
			for (int i = 0; i < 4; i++)In(0)[idx++] = it.Scores[i] / 100;
			In(0)[idx++] = it.honba_sticks;
			In(0)[idx++] = it.riichi_sticks;
		}
		
		assert(idx == Round_Info::input_data_size);
		for (int i = 0; i < 4; i++)Out(0)[i] = final_Scores[i];
	}
};

struct RewardAgent {
	SimpleNet* sim;
	LSTM_Param** data;
	int Batch;

	RewardAgent(const char* param_path) {
		sim = new SimpleNet(RewardNetwork, param_path, true);
		new(this)RewardAgent(sim);
	}
	RewardAgent(SimpleNet*agent) {
		sim = agent;
		Batch = agent->param["Batch"];
		data = new LSTM_Param * [Batch];
		for (int i = 0; i < Batch; i++)data[i] = new LSTM_Param(1);
	}
	~RewardAgent() {
		delete sim;
		for (int i = 0; i < Batch; i++)delete data[i];
		delete[] data;
	}

	//recall in Env round
	void DataEncode(int idx, Round_Info& Start, Round_Info& End, int fillZero) {
		data[idx]->In(0).Reset(Round_Info::input_data_size);
		data[idx]->In(0).Count = Round_Info::input_data_size;
		data[idx]->Out(0).Reset(0);
		//ensure pass cuda assert
		if (fillZero)return;
		//encode input
		*data[idx] = RewardData(Start, End, { -1,-1,-1,-1 });
	}
	static void cudaEncode(Agent* agent) {
		Mat& BoardInPut = agent->InPut_Net.front()->GetOutPut();
		assert(agent->InPut_Net.size() == 1);
		assert(agent->All_In[0].GetCol() == agent->Net_Param["Batch"]);
		BoardInPut._ZeroMemory();
		BoardInPut.Mahjong_Reward_RepresentDecode(agent->All_In[0], Image_Depth);
	}

	vector<vector<double>> getReward() {
		sim->feed_and_response((Base_Param**)data,cudaEncode);
		//decode output data
		vector<vector<double>> res;
		for (int i = 0; i < Batch; i++) {
			vector<double>reward;
			for (int j = 0; j < 2*4; j++)
				reward.push_back(data[i]->Out(0)[j]);
			res.push_back(reward);
		}return res;
	}

	void combine_ds(int idx, int ds_number);
	void train(const char* ds_path, int epoches);

	static const int W = 41;
	static const int H = 1;

	static const int Image_Size = 2 * (9 + 4 * 41 + 11 * 2);
	static const int Image_Depth = 2 * (9 + 4 + 11 * 2);
	static const int filters = 32;
	static const int block_Num = 8;
	static Net* RewardNetwork(HyperParamSearcher& param) {

		//fixed data Loss 500
		/*Net* net = (new Net(Image_Size, InPut));
		net = (new Net(filters, null_Func))->Add_Forward(Transform, net);
		net = (new Net(filters, relu))->Add_Forward(BNTransform, net);
		net = (new Net(filters, null_Func))->Add_Forward(Transform, net);
		net = (new Net(filters, relu))->Add_Forward(BNTransform, net);
		net = (new Net(filters, null_Func))->Add_Forward(Transform, net);
		net = (new Net(filters, relu))->Add_Forward(BNTransform, net);
		net = (new Net(4, Base_Net::tanh))->Add_Forward(Transform, net);
		net = (new Net(4, Base_Net::CostFunc::MeanSquaredError))->Add_Forward(OpsType::OutPut, net);*/

		//ResNet
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
		Net* _net = net;
		//4 softmax head
		for (int i = 0; i < 4; i++) {
			net = (new ConvNet(4, 1, 1, 0, param, null_Func))->Add_Forward(_net, filters, W);
			net = (new Net(relu))->Add_Forward(BNTransform, net);
			//win/loss head
			net = (new Net(2, softmax))->Add_Forward(Transform, net);
			net = (new Net(2, CrossEntropy))->Add_Forward(OpsType::OutPut, net);
		}
		/*net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		net = (new Net(128, relu))->Add_Forward(Transform, net);
		net = (new Net(4, Base_Net::tanh))->Add_Forward(Transform, net);
		net = (new Net(4, Base_Net::CostFunc::MeanSquaredError))->Add_Forward(OpsType::OutPut, net);*/
		//Hidden State Head
		//(new Net(W * H * H_filters, MeanSquaredError))->Add_Forward(OpsType::OutPut, net);

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
			//OutPut._ZeroMemory();

			InPut.Mahjong_Reward_RepresentDecode(In_Sample, Image_Depth);

			Mat final_reward = Out_Sample;
			//OutPut.Append_(Out_Sample, 0);
			/*auto* M = final_reward.ReadFromDevice();
			for (int i = 0; i < 4; i++)printf("%lf ", M[i * final_reward.GetCol() + 0]);*/
			final_reward.Mahjong_Values_Encode();
			int _i = 0;
			for (auto it = agent->OutPut_Net.begin(); it != agent->OutPut_Net.end(); it++) {
				(*it)->GetOutPut()._ZeroMemory();
				(*it)->GetOutPut().Mahjong_Reward_softmax_Encode(_i++, final_reward);
			}
			/*M = final_reward.ReadFromDevice();
			for (int i = 0; i < 4; i++)printf("%lf ", M[i * final_reward.GetCol() + 0]);*/
		}
		else {}
		return true;
	}
};

DataSet* DataSet_Combine(int l, int r, std::function<string(int)>ds_path_func);