#pragma once
#include"Game.h"


struct Go :public Environment {
	static const int W = 15;
	static const int H = 15;
	static const int filters = 32;
	static const int H_filters = 32;
	static const int Image_Depth = 4;

	int Board[8][W][H];
	Go() :Environment(W*H / 4 * 3,getHiddenStateSize()) {}
	//int step;
	int last_move;
	void Reset(mt19937* rng = NULL) {
		memset(Board, 0, sizeof(Board));
		step = 0; last_move = -1;
	}
	void Act(ui Action, double*Reward) {
		assert(Board[0][Action%W][Action / W] == 0);
		Board[0][Action%W][Action / W] = step % 2 + 1;
		step++;
		last_move = Action;
		if (Reward) {
			//
		}
	}
	void GetInsActionMask(bool*ActIsValid) {
		memset(ActIsValid, 0, W*H * sizeof(bool));
		for (int i = 0; i < W*H; i++)
			ActIsValid[i] = (Board[0][i%W][i / W] == 0);
	}
	bool GetGameState(int*result) {
		if (Gomoku_Judger(Board[0], step % 2 == 1, { last_move%W,last_move / W })) {
			*result = (step % 2 == 1) ? 1 : -1;
			return true;
		}
		else if (step >= 200) { *result = 0; return true; }
		else return false;
	}
	void GetGameScreen(double*Scr) {
		for (int i = 0; i < H; i++)
			for (int j = 0; j < W; j++)
				Scr[i*W + j] = Board[0][j][i] ? (Board[0][j][i] == (step % 2 + 1) ? 1 : -1) : 0;
	}
	const double getRewardDiscount() {
		return 1.0;
	}
	const ui getMaxUnrolledStep() {
		return 6;
	}
	const ui getRealGameActionSpace() { return W * H; }
	const ui getScreen_Size() { return W * H; }
	const ui getHiddenStateSize() { return W * H*H_filters; }
	const int Stones = 5;
	struct Pos {
		int x, y;
		Pos() { x = y = 0; }
		Pos(int x, int y) :x(x), y(y) {}
	};
	bool Gomoku_Judger(int(*scr)[H], bool First, Pos pos) {
		//row'-'
		int cnt = 0;
		for (int i = pos.x; i < W; i++)
			if (scr[i][pos.y] == !First + 1)cnt++;
			else break;
		for (int i = pos.x - 1; i > -1; i--)
			if (scr[i][pos.y] == !First + 1)cnt++;
			else break;
		if (cnt >= Stones)return true;
		//col'|'
		cnt = 0;
		for (int i = pos.y; i < H; i++)
			if (scr[pos.x][i] == !First + 1)cnt++;
			else break;
		for (int i = pos.y - 1; i > -1; i--)
			if (scr[pos.x][i] == !First + 1)cnt++;
			else break;
		if (cnt >= Stones)return true;
		//'\'
		cnt = 0;
		for (int i = 0; i < H - max(pos.x, pos.y); i++)
			if (scr[pos.x + i][pos.y + i] == !First + 1)cnt++;
			else break;
		for (int i = 1; i <= min(pos.x, pos.y); i++)
			if (scr[pos.x - i][pos.y - i] == !First + 1)cnt++;
			else break;
		if (cnt >= Stones)return true;
		//'/'
		cnt = 0;
		for (int i = 0; i <= min(pos.x, H - pos.y - 1); i++)
			if (scr[pos.x - i][pos.y + i] == !First + 1)cnt++;
			else break;
		for (int i = 1; i < min(W - pos.x, pos.y + 1); i++)
			if (scr[pos.x + i][pos.y - i] == !First + 1)cnt++;
			else break;
		if (cnt >= Stones)return true;
		return false;
	}


	void DynamicEncode(const list<Net*>&InPut_Nets, Param*Data, int idx, Mat*Hidden_State, const ui Action, int fillZero) {
		//Data->_DataIn->Reset(W*H*(H_filters + 1));
		//Data->_DataIn->Count = Data->_DataIn->MaxCount;
		//if (fillZero)return;
		////Spatial action
		//for (int i = 0; i < W*H; i++) {
		//	for (int j = 0; j < H_filters; j++)
		//		(*Data->_DataIn)[i*(H_filters + 1) + j] = Hidden_State[i*H_filters + j];
		//}
		//assert(0 <= Action && Action < W*H);
		//(*Data->_DataIn)[Action*(H_filters + 1) + H_filters + 0] = 1;
		//Data->_DataOut->Reset(0);
	}
	void RepresentEncode(Param*Data, int fillZero) {
		Data->_DataIn->Reset(W*H);
		Data->_DataIn->Count = Data->_DataIn->MaxCount;
		if (fillZero)return;
		double Scr[W*H]{ 0 };GetGameScreen(Scr);
		for (int i = 0; i < W*H; i++)
			(*Data->_DataIn)[i] = Scr[i];
		Data->_DataOut->Reset(0);
	}
	/*void PredictionEncode(Param*Data, const double*Hidden_State, int fillZero) {
		Data->_DataIn->Reset(W*H*H_filters);
		Data->_DataIn->Count = Data->_DataIn->MaxCount;
		if (fillZero)return;
		for (int i = 0; i < W*H; i++) {
			for (int j = 0; j < H_filters; j++)
				(*Data->_DataIn)[i*H_filters + j] = Hidden_State[i*H_filters + j];
		}
		Data->_DataOut->Reset(0);
	}*/
	void RepresentDecode(Agent*agent) {
		Mat&BoardInPut = agent->InPut_Net.front()->GetOutPut();
		assert(agent->InPut_Net.size() == 1);

		//InPut
		assert(agent->All_In[0].GetCol() == agent->Net_Param["Batch"]);
		assert(agent->All_In[0].GetRow() == W * H);
		//BoardInPut = agent->All_In[0];
		BoardInPut._ZeroMemory();
		BoardInPut.GomokuSimulation_Extend(agent->All_In[0], Image_Depth);
	}
	void DynamicDecode(Agent*agent) {
		Gomoku::MCTS_Direct(agent, NULL, 0, 0);
	}

	void Get_NextState_Reward_And_Policy_Value(const list<Net*>&OutPut_Nets, Param*Data, int idx, Mat*Next_State, double*Reward, double*Policy, double*Value) {
		/*int offset = 0; if (OutPut_Nets.front()->Net_Node_Num == 1)offset = 1;
		for (int i = 0; i < Data->_DataOut->Count - 1; i++) {
			Next_State[i] = (*Data->_DataOut)[i + offset];
		}
		*Reward = (*Data->_DataOut)[offset == 1 ? 0 : (Data->_DataOut->Count - 1)];*/
	}
	void Get_Initial_State_And_Policy_Value(const list<Net*>&OutPut_Nets, Param*Data, int idx, Mat*Next_State, double*Policy, double*Value) {
		/*for (int i = 0; i < Data->_DataOut->Count; i++)
			Next_State[i] = (*Data->_DataOut)[i];*/
	}
	/*void Get_Policy_And_Value(const list<Net*>&OutPut_Nets, Param*Data, double*Policy, double*Value) {
		int offset = 0; if (OutPut_Nets.front()->Net_Node_Num == 1)offset = 1;
		for (int i = 0; i < Data->_DataOut->Count - 1; i++) {
			Policy[i] = (*Data->_DataOut)[i + offset];
		}
		*Value = (*Data->_DataOut)[offset == 1 ? 0 : (Data->_DataOut->Count - 1)];
	}*/

	int block_Num = 8;

	Net*RepresentationNet(HyperParamSearcher&param) {
		Net*net = (new Net(W*H*Image_Depth, InPut));
		net = (new ConvNet(filters, 5, 1, (5 - 1) / 2, param, null_Func))->Add_Forward(net, Image_Depth, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		//residual block
		for (int i = 0; i < block_Num; i++) {
			Net*ori_Image = net;
			net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
			net = (new Net(relu))->Add_Forward(BNTransform, net);
			net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
			net = (new Net(null_Func))->Add_Forward(BNTransform, net);
			Net*Cell = (new Net(net->Net_Node_Num))->Add_Pair_Forward(DotPlus, net, ori_Image);
			net = (new Net(net->Net_Node_Num, relu))->Add_Forward(OpsType::function, Cell);
		}
		net = (new ConvNet(H_filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		net = (new Net(W*H*H_filters, MeanSquaredError))->Add_Forward(OpsType::OutPut, net);
		return net;
	}
	Net*DynamicsNet(HyperParamSearcher&param) {
		const int Image_Depth = H_filters + 1;
		Net*net = (new Net(W*H*Image_Depth, InPut));

		net = (new ConvNet(filters, 5, 1, (5 - 1) / 2, param, null_Func))->Add_Forward(net, Image_Depth, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		//residual block
		for (int i = 0; i < block_Num; i++) {
			Net*ori_Image = net;
			net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
			net = (new Net(relu))->Add_Forward(BNTransform, net);
			net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
			net = (new Net(null_Func))->Add_Forward(BNTransform, net);
			Net*Cell = (new Net(net->Net_Node_Num))->Add_Pair_Forward(DotPlus, net, ori_Image);
			net = (new Net(net->Net_Node_Num, relu))->Add_Forward(OpsType::function, Cell);
		}
		Net*_net = net;
		//Hidden_State
		net = (new ConvNet(H_filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		net = (new Net(W*H*H_filters, MeanSquaredError))->Add_Forward(OpsType::OutPut, net);
		//Reward
		net = (new ConvNet(1, 1, 1, 0, param, null_Func))->Add_Forward(_net, filters, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		net = (new Net(64, relu))->Add_Forward(Transform, net);
		net = (new Net(1, Base_Net::tanh))->Add_Forward(Transform, net);
		net = (new Net(1, Base_Net::CostFunc::MeanSquaredError))->Add_Forward(OpsType::OutPut, net);
		return net;
	}
	Net*PredictionNet(HyperParamSearcher&param) {
		const int Image_Depth = H_filters;
		Net*net = (new Net(W*H*Image_Depth, InPut));
		Net*_net = net;
		//Policy
		net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, Image_Depth, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		net = (new ConvNet(1, 1, 1, 0, param, null_Func, true))->Add_Forward(net, filters, W);
		net = (new Net(W*H, softmax))->Add_Forward(OpsType::function, net);
		net = (new Net(W*H, CrossEntropy))->Add_Forward(OpsType::OutPut, net);

		//Value
		net = (new ConvNet(1, 1, 1, 0, param, null_Func))->Add_Forward(_net, Image_Depth, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		net = (new Net(64, relu))->Add_Forward(Transform, net);
		net = (new Net(1, Base_Net::tanh))->Add_Forward(Transform, net);
		net = (new Net(1, Base_Net::CostFunc::MeanSquaredError))->Add_Forward(OpsType::OutPut, net);
		return net;
	}
	Net*JointNet(HyperParamSearcher&param) {
		//RepresentationNet
		Net*net = (new Net(W*H*Image_Depth, InPut, Net_Flag_RNN_Initial_Step));
		net = (new ConvNet(filters, 5, 1, (5 - 1) / 2, param, null_Func, false, Net_Flag_RNN_Initial_Step))->Add_Forward(net, Image_Depth, W);
		net = (new Net(relu, Net_Flag_RNN_Initial_Step))->Add_Forward(BNTransform, net);
		//residual block
		for (int i = 0; i < block_Num; i++) {
			Net*ori_Image = net;
			net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_Initial_Step))->Add_Forward(net, filters, W);
			net = (new Net(relu, Net_Flag_RNN_Initial_Step))->Add_Forward(BNTransform, net);
			net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_Initial_Step))->Add_Forward(net, filters, W);
			net = (new Net(null_Func, Net_Flag_RNN_Initial_Step))->Add_Forward(BNTransform, net);
			Net*Cell = (new Net(net->Net_Node_Num, null_Ops, Net_Flag_RNN_Initial_Step))->Add_Pair_Forward(DotPlus, net, ori_Image);
			net = (new Net(net->Net_Node_Num, relu, Net_Flag_RNN_Initial_Step))->Add_Forward(OpsType::function, Cell);
		}
		net = (new ConvNet(H_filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_Initial_Step))->Add_Forward(net, filters, W);
		Net*RepresentState = net = (new Net(relu, Net_Flag_RNN_Initial_Step))->Add_Forward(BNTransform, net);


		//DynamicsNet
		//dynamics
		Net*actionInPut = (new Net(W*H * 1, InPut, Net_Flag_RNN_non_Initial_Step));
		Net*StateAction = net = (new Net(W*H*(H_filters + 1), W*H, Net_Flag_RNN_non_Initial_Step));
		net = (new ConvNet(filters, 5, 1, (5 - 1) / 2, param, null_Func, false, Net_Flag_RNN_non_Initial_Step))->Add_Forward(net, H_filters + 1, W);
		net = (new Net(relu, Net_Flag_RNN_non_Initial_Step))->Add_Forward(BNTransform, net);
		//residual block
		for (int i = 0; i < block_Num; i++) {
			Net*ori_Image = net;
			net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_non_Initial_Step))->Add_Forward(net, filters, W);
			net = (new Net(relu, Net_Flag_RNN_non_Initial_Step))->Add_Forward(BNTransform, net);
			net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_non_Initial_Step))->Add_Forward(net, filters, W);
			net = (new Net(null_Func, Net_Flag_RNN_non_Initial_Step))->Add_Forward(BNTransform, net);
			Net*Cell = (new Net(net->Net_Node_Num, null_Ops, Net_Flag_RNN_non_Initial_Step))->Add_Pair_Forward(DotPlus, net, ori_Image);
			net = (new Net(net->Net_Node_Num, relu, Net_Flag_RNN_non_Initial_Step))->Add_Forward(OpsType::function, Cell);
		}
		Net*__net = net;
		//Hidden_State
		net = (new ConvNet(H_filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_non_Initial_Step))->Add_Forward(net, filters, W);
		Net*DynamicState = net = (new Net(relu, Net_Flag_RNN_non_Initial_Step))->Add_Forward(BNTransform, net);
		//Reward
		net = (new ConvNet(1, 1, 1, 0, param, null_Func, false, Net_Flag_RNN_non_Initial_Step))->Add_Forward(__net, filters, W);
		net = (new Net(relu, Net_Flag_RNN_non_Initial_Step))->Add_Forward(BNTransform, net);
		net = (new Net(64, relu, Net_Flag_RNN_non_Initial_Step))->Add_Forward(Transform, net);
		net = (new Net(1, Base_Net::tanh, Net_Flag_RNN_non_Initial_Step))->Add_Forward(Transform, net);
		net = (new Net(1, Base_Net::CostFunc::MeanSquaredError, Net_Flag_RNN_non_Initial_Step, getMaxUnrolledStep()))->Add_Forward(OpsType::OutPut, net);


		Net*StateSwitch = (new Net(W*H*H_filters))->Add_Pair_Forward(RNNInPutSwitch, DynamicState, RepresentState);
		StateAction->Add_Pair_Forward(SpatialConcatenate, StateSwitch, actionInPut, Net_Flag_Reconnect);


		//PredictionNet
		//prediction
		net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(StateSwitch, H_filters, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		net = (new ConvNet(1, 1, 1, 0, param, null_Func, true))->Add_Forward(net, filters, W);
		net = (new Net(W*H, softmax))->Add_Forward(OpsType::function, net);
		net = (new Net(W*H, CrossEntropy, 0, getMaxUnrolledStep()))->Add_Forward(OpsType::OutPut, net);
		//Value
		net = (new ConvNet(1, 1, 1, 0, param, null_Func))->Add_Forward(StateSwitch, H_filters, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		net = (new Net(64, relu))->Add_Forward(Transform, net);
		net = (new Net(1, Base_Net::tanh))->Add_Forward(Transform, net);
		net = (new Net(1, Base_Net::CostFunc::MeanSquaredError, 0, getMaxUnrolledStep()))->Add_Forward(OpsType::OutPut, net);

		return net;
	}

	bool Train_In_Out_Process(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col/*, Agent*Server*/) {
		Mat&BoardInPut = (agent->InPut_Net.front()->Net_Flag&Net_Flag_RNN_Initial_Step) ? agent->InPut_Net.front()->GetOutPut() : agent->InPut_Net.back()->GetOutPut();
		Mat&ActionInPut = (agent->InPut_Net.front()->Net_Flag&Net_Flag_RNN_Initial_Step) ? agent->InPut_Net.back()->GetOutPut() : agent->InPut_Net.front()->GetOutPut();

		Mat*BoardOutPut = NULL, *ValueOutPut = NULL, *RewardOutPut = NULL;
		for (auto&k : agent->OutPut_Net) {
			if (k->Net_Flag&Net_Flag_RNN_non_Initial_Step)
				RewardOutPut = &k->GetOutPut();
			else if (k->CostFunction == CostFunc::MeanSquaredError)
				ValueOutPut = &k->GetOutPut();
			else BoardOutPut = &k->GetOutPut();
		}
		//if (!_Row.IsValid() || _Row.GetCol() != agent->Net_Param["Batch"][0])
		//Mat In_Sample(Server->All_In[0].GetRow(), agent->Net_Param["Batch"]);// , Randomer.Reset(1, agent->Net_Param["Batch"]);
		//Mat Out_Sample(Server->All_Out[0].GetRow(), agent->Net_Param["Batch"]);

		//Train_Test(In_Sample, Server->All_In[_step], RandomGenerator, test_start_col);
		//Train_Test(Out_Sample, Server->All_Out[0], &In_Sample, -1);
		//InPut
		//if (_step == 0) {
		//	BoardInPut._ZeroMemory();
		//	BoardOutPut->_ZeroMemory();
		//	ValueOutPut->_ZeroMemory();

		//	BoardInPut.GomokuSimulation_Extend(Out_Sample, Image_Depth);
		//	//BoardInPut.Append_(Out_Sample, 0);

		//	BoardOutPut->Append_(Out_Sample, W*H);
		//	ValueOutPut->Append_(Out_Sample, W*H * 2 + 1);
		//}
		//else {
		//	ActionInPut._ZeroMemory();
		//	BoardOutPut->_ZeroMemory();
		//	ValueOutPut->_ZeroMemory();
		//	RewardOutPut->_ZeroMemory();

		//	ActionInPut.Go_Action_Encode(Out_Sample, W*H * 2, W*H);

		//	BoardOutPut->Append_(Out_Sample, W*H);
		//	ValueOutPut->Append_(Out_Sample, W*H * 2 + 1);
		//	RewardOutPut->Append_(Out_Sample, W*H * 2 + 2);
		//}
		return true;
	}
	std::function<bool(Agent*, Mat*, int, int/*, Agent**/)> getTrainFun() {
		return std::bind(&Go::Train_In_Out_Process, this, _1, _2, _3, _4/*, _5*/);
	}
	std::function<Net*(HyperParamSearcher&)> getJointNetFun() {
		return std::bind(&Go::JointNet, this, _1);
	}
};
