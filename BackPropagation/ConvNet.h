#pragma once
//#ifdef ConvNet_EXPORTS
#define __ConvNet__ __declspec(dllexport)
//#else
//#define __ConvNet__ __declspec(dllimport)

//#endif

#include"RNN_LSTM.h"
#include"Agent.h"

using namespace _Net_;
using namespace Base_Net;
using namespace Net_CPU;

class __ConvNet__ Conv {
public:
	//OutPut Depth
	int Filters;
	//Receptive Field
	int Receptive;
	//Receptive Stride
	int Stride;
	//Image Padding
	int ZeroPadding;
	//InPut W&H&Depth
	int Pre_W, Pre_H, Pre_Depth;
	//OutPut Image W&H
	int W, H;

	int Max_Step;
	int Batch;
	Conv() {}
	Conv(int Filters, int Receptive_Field, int Stride, int Padding, HyperParamSearcher&param) :Filters(Filters),
		Receptive(Receptive_Field), Stride(Stride), ZeroPadding(Padding) {
		Max_Step = param["Max_Step"];
		Batch = param["Batch"];
	}
	void cal_WH(int Pre_Node, int InPut_Image_Depth, int InPut_Image_W) {
		assert(Pre_Node > 0);
		Pre_Depth = InPut_Image_Depth, Pre_W = InPut_Image_W, Pre_H = Pre_Node / (Pre_W*Pre_Depth);
		DEBUG((Pre_W - Receptive + 2 * ZeroPadding) % Stride != 0, "error:Conv W\n");
		DEBUG((Pre_H - Receptive + 2 * ZeroPadding) % Stride != 0, "error:Conv H\n");
		W = (Pre_W - Receptive + 2 * ZeroPadding) / Stride + 1;
		H = (Pre_H - Receptive + 2 * ZeroPadding) / Stride + 1;
	}
	void Data_WR(fstream&file, bool Write, int Batch, int Max_Step) {
		File_WR(file, (char*)&Filters, sizeof(Filters), Write);
		File_WR(file, (char*)&Receptive, sizeof(Receptive), Write);
		File_WR(file, (char*)&Stride, sizeof(Stride), Write);
		File_WR(file, (char*)&ZeroPadding, sizeof(ZeroPadding), Write);
		File_WR(file, (char*)&Pre_W, sizeof(Pre_W), Write);
		File_WR(file, (char*)&Pre_H, sizeof(Pre_H), Write);
		File_WR(file, (char*)&Pre_Depth, sizeof(Pre_Depth), Write);
		if (!Write) {
			this->Max_Step = Max_Step;
			this->Batch = Batch;
		}
	}
};

class __ConvNet__ ConvNet :public RNN_LSTM::LSTMUnit,public Conv{
	//Location InPut Matrix
	Mat*im2col;
	//extra bias
	bool Different_Bias;

	ConvNet(int Pre_Node, int InPut_Image_Depth, int InPut_Image_W, Activation_Func Function, NetFlag Net_flag) :LSTMUnit(Max_Step, Filters, Receptive*Receptive*InPut_Image_Depth, Batch, (cal_WH(Pre_Node, InPut_Image_Depth, InPut_Image_W),(Different_Bias?W*H*Filters:-1))) {
		Net_Node_Num = W * H * Filters;
		Func = Function;
		C_OutPut.Reset(Net_Node_Num, Batch);
		im2col = new Mat[Max_Step];
		Net_Flag |= Net_flag;
	}
public:
	ConvNet() {}
	ConvNet(int Filters, int Receptive_Field, int Stride, int Padding, HyperParamSearcher&param, Activation_Func Function = relu, bool different_bias = false, NetFlag Net_flag = Net_Flag_null) :Conv(Filters, Receptive_Field, Stride, Padding, param) {
		assert(Stride > 0 && Filters > 0 && Receptive_Field > 0);
		Func = Function;
		Different_Bias = different_bias;
		Net_Flag |= Net_flag;
	}
	~ConvNet() {
		delete[] im2col;
	}
	inline bool checkBN() {
		return Next_Net_Head.front()->Ops == BNTransform;
	}
	double Forward(int idx, int IsTrain, bool cal_loss) {
		//Image depth first,width second
		im2col[idx] = Pre_Net_Head.front()->GetOutPut().Conv_im2col(Pre_Depth, Pre_W, W, H, Receptive, ZeroPadding, Stride);
		//C_OutPut.Reset(Filters * W * H, Batch);
		if (!Different_Bias) {
			Mat raw_OutPut = C_Weigh % im2col[idx] + C_b;
			//BN followed
			if (!checkBN())
				raw_OutPut.Conv_Image_Restore(W, C_OutPut);
			else {
				C_OutPut.Reset(raw_OutPut.GetRow(), raw_OutPut.GetCol());
				C_OutPut = raw_OutPut;
			}
		}
		//different bias for per Node
		else {
			Mat raw_OutPut = C_Weigh % im2col[idx];
			raw_OutPut.Conv_Image_Restore(W, C_OutPut);
			C_OutPut = C_OutPut + C_b;
		}
		Activate(C_OutPut);
		return 0;
	}
	bool Backward_Gradient(Mat*&result, Net*Pre_Net, int lstm_idx, int Max_Step) {
		Mat gradient = (!C_Weigh) % Gradient;
		result->_ZeroMemory();
		result->Conv_im2col(Pre_Depth, Pre_W, W, H, Receptive, ZeroPadding, Stride, &gradient);
		return true;
	}
	void Calculate_Bias(int lstm_idx) {
		Derivate(lstm_idx);
		//restore
		Mat raw_OutPut(Filters, W * H * Batch);
		if (!checkBN())
			raw_OutPut.Conv_Image_Restore(W, Gradient, true);
		else raw_OutPut = Gradient;
		if (!Different_Bias)
			C_Biasb -= raw_OutPut;
		else C_Biasb -= Gradient;
		C_BiasWeigh -= raw_OutPut % (!im2col[lstm_idx]);
		if (!checkBN())
			Gradient = move(raw_OutPut);
		im2col[lstm_idx] = NULL;
	}

	Net*Add_Forward(Net*Pre,int Image_Depth,int Image_W,ui Extra_Flag = Net_Flag_null) {
		new(this)ConvNet(Pre->Net_Node_Num, Image_Depth, Image_W, Func, Net_Flag);
		return Net::Add_Forward(CustomOps, Pre, Extra_Flag);
	}
	//for Conv-based
	Net*Add_Forward(Net*Pre, ui Extra_Flag = Net_Flag_null) {
		Conv*cv = dynamic_cast<Conv*>(Pre);
		DEBUG(!cv, "Pre Net Must Conv-based\n");
		new(this)ConvNet(cv->W*cv->H*cv->Filters, cv->Filters, cv->W, Func, Net_Flag);
		return Net::Add_Forward(CustomOps, Pre, Extra_Flag);
	}
	void Data_WR(fstream&file, Net**Net_Priority, int tot_Net_Num, bool Write, int Batch, int Max_Step) {
		Conv::Data_WR(file, Write, Batch, Max_Step);
		File_WR(file, (char*)&Different_Bias, sizeof(Different_Bias), Write);
		if (!Write) {
			new(this)ConvNet(Pre_W*Pre_H*Pre_Depth, Pre_Depth, Pre_W, Func, Net_Flag);
		}
	}
	void Reduction(int Max_Step) {
		for (int i = 0; i < Max_Step; i++)
			im2col[i].Clear();
		LSTMUnit::Reduction(Max_Step);
	}
};
class POOL :public RNN_LSTM::LSTMInput,public Conv {
	enum PoolType {
		MaxPool, AveragePool,DropOutPool
	}Type;
	Mat*Pool_Gradient_idx;
	POOL(int Pre_Node, int InPut_Image_Depth, int InPut_Image_W):LSTMInput(Max_Step, 1, Batch) {
		Filters = InPut_Image_Depth;
		cal_WH(Pre_Node, InPut_Image_Depth, InPut_Image_W);
		Net_Node_Num = W * H * Filters;
		C_InPut.Reset(Net_Node_Num, Batch, false);
		Pool_Gradient_idx = new Mat[Max_Step];
		for (int i = 0; i < Max_Step; i++)
			Pool_Gradient_idx[i].Reset(Net_Node_Num, Batch, false);
	}
public:
	POOL() {}
	POOL(int Receptive_Field, int Stride, int Padding, HyperParamSearcher&param) :Conv(0, Receptive_Field, Stride, Padding, param) {
		assert(Stride > 0 && Receptive_Field > 0);
	}
	~POOL() {
		delete[] Pool_Gradient_idx;
	}
	double Forward(int idx, int IsTrain, bool cal_loss) {
		C_InPut = Pre_Net_Head.front()->GetOutPut().Image_Pooling(Pre_Depth, Pre_W, W, H, Receptive, ZeroPadding, Stride, Pool_Gradient_idx[idx]);
		return 0;
	}
	bool Backward_Gradient(Mat*&result, Net*Pre_Net, int lstm_idx, int Max_Step) {
		result->_ZeroMemory();
		result->Image_Pooling(Pre_Depth, Pre_W, W, H, Receptive, ZeroPadding, Stride, Pool_Gradient_idx[lstm_idx], &Gradient);
		return true;
	}

	//only for Conv-based
	Net*Add_Forward(Net*Pre, int Image_Depth, int Image_W, ui Extra_Flag = Net_Flag_null) {
		new(this)POOL(Pre->Net_Node_Num, Image_Depth, Image_W);
		return Net::Add_Forward(CustomOps, Pre, Extra_Flag);
	}
	Net*Add_Forward(Net*Pre, ui Extra_Flag = Net_Flag_null) {
		Conv*cv = dynamic_cast<Conv*>(Pre);
		DEBUG(!cv, "Pre Net Must Conv-based\n");
		new(this)POOL(cv->W*cv->H*cv->Filters, cv->Filters, cv->W);
		return Net::Add_Forward(CustomOps, Pre, Extra_Flag);
	}
	void Data_WR(fstream&file, Net**Net_Priority, int tot_Net_Num, bool Write, int Batch, int Max_Step) {
		Conv::Data_WR(file, Write, Batch, Max_Step);
		if (!Write) {
			new(this)POOL(Pre_W*Pre_H*Pre_Depth, Pre_Depth, Pre_W);
		}
	}
	void Reduction(int Max_Step) {
		for (int i = 0; i < Max_Step; i++)
			Pool_Gradient_idx[i].Clear();
		LSTMInput::Reduction(Max_Step);
	}
};