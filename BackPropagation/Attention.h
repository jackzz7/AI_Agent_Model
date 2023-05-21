#pragma once
#ifdef Attention_EXPORTS
#define __Attention__ __declspec(dllexport)
#else
#define __Attention__ __declspec(dllimport)

#endif

#include"RNN_LSTM.h"
#include"Agent.h"

using namespace _Net_;
using namespace Base_Net;

struct Mat_Exp_Average {
	double Beta1;
	Mat res, ret;
	Mat_Exp_Average() {
		Beta1 = 1;// .Reset(1, 1, 1.0);
		//cal update conut not zero
		//res.Reset(Node, 1, 0.00001);
	}
	void update(Mat& val,const double&Beta,const double&Batch) {
		Beta1 *= Beta;
		if (!res.IsValid())res.Reset(val.GetRow(), 1, 0.00001f);
		res = Beta * res + (1 - Beta) * val.ScaleOneCol() / Batch;
		ret = res / (1 - Beta1);
	}
	operator Mat&(){
		return ret;
	}
	void Clear() {
		//res.Clear();
		ret.Clear();
		//Beta1.Reset(1, 1, 1.0);
	}
};
namespace Attention
{
	enum ReinforceDistribution {
		Gaussian, OneHot,RewardNet
	};
	class __Attention__ REINFORCE :public RNN_LSTM::LSTMInput {
		ReinforceDistribution DistributionType;
		Mat NormalGenerator, RandomStartLocation;
		//Expected Reward
		//Reward Baseline
		//exponent average
		Mat*lstm_Reward_Baseline;
		Mat_Exp_Average*Pre_OutPut_Mean;
		Mat_Exp_Average*Reward_Mean;
		//Step Reward
		Mat*lstm_Reward;
		Mat*Boundary_OutPut;
		NetParam Net_Param;


		int Batch;
		Net*Baseline;
	public:
		//PreInit
		REINFORCE() { Net_Param.clear(); }
		REINFORCE(ReinforceDistribution Distribution, int Node, HyperParamSearcher&param,NetFlag Net_Flag = Net_Flag_null, Activation_Func Func = thresholding) :LSTMInput(param["Max_Step"], Node, param["Batch"]) {
			Batch = param["Batch"];
			Baseline = NULL;
			int Max_RNN_Step = param["Max_Step"];
			//NormalGenerator.Reset(Node, Batch);
			lstm_Reward_Baseline = new Mat[Max_RNN_Step];
			Pre_OutPut_Mean = new Mat_Exp_Average[Max_RNN_Step];
			Reward_Mean = new Mat_Exp_Average[Max_RNN_Step];
			lstm_Reward = new Mat[Max_RNN_Step];
			Boundary_OutPut = new Mat[Max_RNN_Step];
			/*for (int i = 0; i < Max_RNN_Step; i++) {
				Pre_OutPut_Mean[i] = Mat_Exp_Average(Node, Batch);
				Reward_Mean[i] = Mat_Exp_Average(1, Batch);
			}*/
			Net_Param["Batch"] = 1.0;
			Net_Param["stdev"] = 0.1;// .Reset(1, 1, 0.1);
			Net_Param["Reward_Baseline_Beta1"] = 0.1;// Reset(1, 1, 0.1);
			Net_Param["Mean_Baseline_Beta1"] = 0.0;// .Reset(1, 1, 0.0);
			Net_Param["Discount"] = 1;// .Reset(1, 1, 1.0);
			Net_Param["Loc_Gradient_decay"] = 1;// .Reset(1, 1, 1.0);
			Net_Param["Given_OutPut"] = 0;// .Reset(1, 1, 0.0);
			Agent::Prase_Param(Net_Param, param);
			RandomStartLocation.Reset(Net_Node_Num, Batch);
			NormalGenerator.Reset(Net_Node_Num, Batch);
			Net_Node_Num = Node;
			this->Func = Func;
			this->DistributionType = Distribution;
			this->Net_Flag = Net_Flag | Net_Flag_Reinforce;
		}
		~REINFORCE() {
			delete[] lstm_Reward_Baseline;
			delete[] lstm_Reward;
			delete[] Boundary_OutPut;
			delete[] Pre_OutPut_Mean;
			delete[] Reward_Mean;
			Net_Param.clear();
		}

		double Forward(int idx, int IsTrain, bool cal_loss) {
			switch (DistributionType)
			{
				//sample from Normal distribution
				//Gaussian Distribution Explore
			case Gaussian:
				if (true) {
					Mat Mean;
					//using OutPut mean
					//better when train small Pic
					if (Net_Param["Mean_Baseline_Beta1"] > 0) {
						Pre_OutPut_Mean[idx].update(Pre_Net_Head.front()->GetOutPut(), Net_Param["Mean_Baseline_Beta1"], Net_Param["Batch"]);
						Mean = Pre_OutPut_Mean[idx];
					}//not using
					else  Mean = Pre_Net_Head.front()->GetOutPut();
					if (IsTrain) {
						Mean.f(Thresholding, 1.0);
						NormalGenerator.Normal_Sampling();
						//NormalGenerator.ReadFromDevice();
						C_InPut = Mean + Net_Param["stdev"] * NormalGenerator;
					}
					//direct OutPut mean
					else {
						if (Net_Param["Mean_Baseline_Beta1"] > 0) {
							C_InPut._ZeroMemory();
							C_InPut = C_InPut + Mean;
						}else C_InPut = Mean;
					}
					//OutPut Thresholding
					Boundary_OutPut[idx] = C_InPut.f(Thresholding, 1.0);
				}
				break;
				//discrete distribution
			case OneHot:
				if (IsTrain) {
					//e^(a/T) 
					C_InPut = Pre_Net_Head.front()->GetOutPut();
					Boundary_OutPut[idx] = C_InPut.OneHot_Sampling();
				}
				else {
					//softmax max
					C_InPut = Pre_Net_Head.front()->GetOutPut();
					C_InPut.SoftMax(true);
				}
				break;
			case RewardNet:
				C_InPut = Pre_Net_Head.front()->GetOutPut();
				C_InPut.f(Thresholding, 1.0);
				break;
			default:
				break;
			}return 0;
		}
		//Reinforce Only Backward Reward-Gradient
		bool Backward_Gradient(Mat*&result, Net*Pre_Net, int lstm_idx, int Max_Step) {
			Mat Pre_OutPut = NULL, *output_ptr = &Pre_OutPut;
			switch (DistributionType)
			{
			case Gaussian:
				//Lastest Step No reward
				if ((Net_Flag&Net_Flag_Random_Data) && lstm_idx + 1 >= Max_Step)
					return false;
				if (Net_Param["Mean_Baseline_Beta1"] > 0)
					Pre_OutPut = Pre_OutPut_Mean[lstm_idx];
				else Pre_OutPut = *GetPreOutPut(lstm_idx, Pre_Net);
				Pre_OutPut.f(Thresholding, 1.0);
				*result = (Pre_OutPut - Boundary_OutPut[lstm_idx])*Net_Param["Loc_Gradient_decay"];
				*result *= lstm_Reward[lstm_idx] -lstm_Reward_Baseline[lstm_idx];
				break;
			case OneHot:
				//Sampling Gradient
				output_ptr = GetPreOutPut(lstm_idx, Pre_Net);
				if (Next_Net_Head.front()->CostFunction == null_Cost_Func) {
					*result = *output_ptr - Boundary_OutPut[lstm_idx];
				}
				//OutPut Gradient
				else {
					*result = *output_ptr - Next_Net_Head.front()->GetOutPut(lstm_idx);
				}
				*result *= lstm_Reward[lstm_idx];// -Baseline->GetOutPut(lstm_idx);//lstm_Reward_Baseline[lstm_idx];
				break;
			case RewardNet:
				*result = lstm_Reward_Baseline[lstm_idx] - lstm_Reward[lstm_idx];
				break;
			default:
				return false;
				break;
			}
			return true;
		}
		//Step-based Reward
		//baseline
		//step recall
		void Cal_Reward(int idx, Mat&Reward) {
			lstm_Reward[idx] = Reward;
			//Reward_Mean[idx].update(Reward, Net_Param["Reward_Baseline_Beta1"], Net_Param["Batch"]);
			//lstm_Reward_Baseline[idx] = Reward_Mean[idx];
			/*if (Reward==NULL) {
				Mat*Mask = NULL;
				if (OutPut_Net->Pre_Net_Head.front()->Net_Flag&Net_Flag_OutPut_Mask)
					Mask = &OutPut_Net->Pre_Net_Head.front()->Pre_Net_Head.back()->GetOutPut();
				OutPut_Net->Test(cor, Mask, &lstm_Reward[idx]);
			}*/
			//Online Reward
			/*else {
				for (int i = 0; i < Batch; i++)cor += Reward[i];
				lstm_Reward[idx].Reset(1, Batch, Reward);
			}*/
			//Step Average Baseline
			/*reward = (cor / Batch);
			lstm_Reward_Baseline[idx][0] = Beta1 * lstm_Reward_Baseline[idx][0] + (1.0 - Beta1)*reward;*/
		}
		//Status-based Reward
		void Cal_Status_Reward(int idx, Mat&Reward, Net*Reward_Net) {
			lstm_Reward[idx] = Reward;
			if (!Baseline)Baseline = Reward_Net;
		}
		void Cal_TD_Reward(int idx,Net*Reward_Net,double*Reward,int Ins_Max_Step, int Type) {
			//Reward - Baseline
			//r(t)+V(t+1)-V(t)
			//Zero Sum
			lstm_Reward[idx].Reset(1, Batch, Reward);
			int Ridx = (Type == -1 ? idx : (idx * 2 + Type));
			if (Ridx + 1 < Ins_Max_Step) {
				//discount reward
				lstm_Reward[idx] += /*Mat(1, 1, 0.99)**/-1 * Reward_Net->GetOutPut(Ridx + 1);
			}
			lstm_Reward_Baseline[idx] = Reward_Net->GetOutPut(Ridx);
			//GAE
			//reverse iter Order

			//Reward>1
			/*if (idx + 1 < Ins_Max_Step) {
				double r = 0.95;
				lstm_Reward[idx] += Mat(1, 1, r)*lstm_Reward[idx + 1];
				lstm_Reward_Baseline[idx] += Mat(1, 1, r)*lstm_Reward_Baseline[idx + 1];
			}*/
			//set Reward Target OutPut
			//A(t)+V(t)
			//Gradient Thresholding
			//Reward_Net->Next_Net_Head.front()->GetOutPut(idx) = (lstm_Reward[idx] - lstm_Reward_Baseline[idx]).f(Device_Func::Thresholding, 0.005) + Reward_Net->GetOutPut(idx);
			//r(t)+V(t+1)
			//Reward_Net->GetOutPut(idx) = Mat(1, Batch, Reward);
			//if (step + 1 < lstm_Step + 1) {
			//	//discount reward
			//	Reward_OutPut_Net->GetOutPut(step) +=/*Mat(1,1,1.0)**/Reward_OutPut_Net->Pre_Net_Head.front()->GetOutPut(step + 1);
			//}
		}
		Net*Add_Forward(Net*Pre, ui Extra_Flag = Net_Flag_null) {
			return Net::Add_Forward(CustomOps, Pre, Extra_Flag);
		}
		void RandData() {
			C_InPut = RandomStartLocation.f(Uniform, 1.0);
		}
		Mat& GetStartRandomData() {
			return RandomStartLocation;
		}

		//Data WR
		void Data_WR(fstream&file, Net**Net_Priority, int tot_Net_Num, bool Write, int Batch, int Max_Step) {
			File_WR(file, (char*)&DistributionType, sizeof(ReinforceDistribution), Write);
			File_WR(file, (char*)&Net_Node_Num, sizeof(ui), Write);
			HyperParamSearcher param;
			Param_WR(file, Net_Param, param, Write);
			if (!Write) {
				//general param
				param["Max_Step"] = Max_Step;
				param["Batch"] = Batch;
				new(this)REINFORCE(DistributionType, Net_Node_Num, param);
			}
		}
		void Reduction(int Max_Step) {
			NormalGenerator.Clear();
			for (int i = 0; i < Max_Step; i++) {
				lstm_Reward_Baseline[i].Clear();
				lstm_Reward[i].Clear();
				Boundary_OutPut[i].Clear();
				Pre_OutPut_Mean[i].Clear();
				Reward_Mean[i].Clear();
			}
		}
	};
	class __Attention__ Glimpse :public RNN_LSTM::LSTMInput
	{
		ui Image_W;
		int Scale_Num;
		int Scale_Image_WH;
	public:
		//PreInit
		Glimpse() {}
		Glimpse(ui Image_W, ui Scale_Num, ui Scale_Image_WH, int Batch, int Max_RNN_Step) :LSTMInput(Max_RNN_Step, Scale_Image_WH*Scale_Image_WH*Scale_Num, Batch) {
			this->Image_W = Image_W;
			this->Scale_Num = Scale_Num;
			this->Scale_Image_WH = Scale_Image_WH;
			Net_Node_Num = Scale_Image_WH * Scale_Image_WH*Scale_Num;
		}

		//functions
		double Forward(int idx, int IsTrain, bool cal_loss) {
			switch (Ops)
			{
			case CustomOps:
				//First InPut Location,Second InPut Image
				if (true) {
					//Loc(x,y)
					Mat Loc = Pre_Net_Head.front()->GetOutPut()*(Image_W / 2.0) + Mat(2, C_InPut.GetCol(), Image_W / 2.0f);
					//Loc.ReadFromDevice();
					//printf("%lf:%lf", Loc[0], Loc[1]);
					//Improvement
					C_InPut = Pre_Net_Head.back()->GetOutPut().ScaleImage(Image_W,1, Loc, Scale_Image_WH, Scale_Num);
					/*double*M = C_InPut.ReadFromDevice();
					for (int i = 0; i < Scale_Num; i++) {
						for (int j = 0; j < Scale_Image_WH; j++) {
							for (int k = 0; k < Scale_Image_WH; k++) {
								printf("%d", (int)M[i*Scale_Image_WH*Scale_Image_WH + j * Scale_Image_WH + k]);
							}
							printf("\n");
						}
						printf("\n\n");
					}*/
				}
			break;
			default:
				break;
			}return 0;
		}
		Net*Add_Pair_Forward(Net*Location, Net*Image, ui Extra_First_Flag = Net_Flag_null, ui Extra_Second_Flag = Net_Flag_null) {
			return Net::Add_Pair_Forward(CustomOps, Location, Image, Extra_First_Flag, Extra_Second_Flag);
		}
		//Data WR
		void Data_WR(fstream&file, Net**Net_Priority, int tot_Net_Num, bool Write, int Batch, int Max_Step) {
			File_WR(file, (char*)&Image_W, sizeof(ui), Write);
			File_WR(file, (char*)&Scale_Num, sizeof(int), Write);
			File_WR(file, (char*)&Scale_Image_WH, sizeof(int), Write);
			if (!Write) {
				new(this)Glimpse(Image_W, Scale_Num, Scale_Image_WH, Batch, Max_Step);
			}
		}
	};
	__Attention__ void Init_Attention(HyperParamSearcher&param, Net*Image, ui ImageW, ui ImageH, ui Scale_Num, ui Scale_Image_WH, ui Out_Num, ui Hidden);
	__Attention__ Net* Init_Glimpse(HyperParamSearcher&param, Net*LocationNet, Net*InPut_Image);
};


