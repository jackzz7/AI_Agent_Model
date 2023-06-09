#pragma once
#ifdef BP_EXPORTS
#define __SGD__ __declspec(dllexport)
#else
#define __SGD__ __declspec(dllimport)

#endif

#define _CRT_SECURE_NO_WARNINGS

#include<fstream>
#include<math.h>
#include<CUDA_ACC.h>
#include<Param.h>

#include<tchar.h>

#include"Agent.h"

using std::fstream;
using std::swap;

using namespace _CUDA_;
using namespace Net_CPU;

namespace _Net_ {

	using namespace Base_Net;

	void TestP(double**ans, double*out, int No, int*S);
	void TestP(double*ans, double*out, int No, int*S);

	//typedef class Param;
	const double eps = 1e-8;

	class __SGD__ G_Net :protected _Net
	{
	protected:
		/*Mat C_In;
		Mat C_Out;

		std::map<string, Mat>Net_Param;

		Mat RandomGenerator;
		Mat C_Num;

		Mat Positive_OutPut_mask;*/
	public:
		struct __SGD__ InPutUnit :public Net
		{
			Mat C_InPut;

			Mat* lstm_Max_Minus_Min;
			InPutUnit() {}
			InPutUnit(int NodeNum, int Batch);

			//Net function
			void Reduction(int Max_Step) {
				C_InPut.Clear();
				Net::Reduction(Max_Step);
			}
			Mat& GetOutPut(int lstm_idx = -1) {
				return C_InPut;
			}
			bool Backward_Gradient(Mat*&result, Net*Pre_Net, int lstm_idx, int Max_Step) {
				//Gradient continous
				switch (Ops)
				{
				case Concatenate:
				{
					ui Start_Row = 0;
					for (auto k : Pre_Net_Head) {
						if (k == Pre_Net) {
							result->Append_(Gradient, Start_Row);
							break;
						}
						Start_Row += k->GetOutPut().GetRow();
					}
				}break;
				case DotProduct:
					//*result = Gradient;
					for (auto k : Pre_Net_Head) {
						if (k == Pre_Net)continue;
						*result = Gradient * (*GetPreOutPut(lstm_idx, k));
					}
					break;
				case DotPlus:
					result = &Gradient;
					break;
				case OpsType::function:
					//must Single feed-Forward
					assert(Pre_Net_Head.size() == 1);
					Derivate(lstm_idx);
					result = &Gradient;
					break;
				case RNNInPutSwitch:
					result = &Gradient;
					break;
				case SpatialConcatenate:
					result->Image_SpatialConcatenate(Gradient, *result, extra_param, true, Pre_Net_Head.front() == Pre_Net);
					break;
				case Scale:
					Gradient *= lstm_Max_Minus_Min[lstm_idx];
					result = &Gradient;
					break;
				case OpsType::OutPut:
					//Mat Pre_OutPut = NULL, *output_ptr = &Pre_OutPut;
					return Loss_Gradient(*GetPreOutPut(lstm_idx, Pre_Net), GetOutPut(lstm_idx), *result);
				break;
				case InPut:
				default:
					return false;
					break;
				}
				return true;
			}
			double Forward(int idx, int IsTrain, bool cal_loss) {
				switch (Ops)
				{
					//TU,In+TU,In=>In
				case Concatenate:
				{
					ui Start_Row = 0;
					for (auto k : Pre_Net_Head) {
						C_InPut.Append(k->GetOutPut(), Start_Row);
						Start_Row += k->GetOutPut().GetRow();
					}
				}
				break;
				//TU,In*TU,In=>In
				case DotProduct:
					C_InPut = Pre_Net_Head.front()->GetOutPut();
					//Test Process no DropOut
					if (!IsTrain && (Net_Flag&Net_Flag_DropOut))
						break;
					for (auto k : Pre_Net_Head) {
						if (k != Pre_Net_Head.front())
							C_InPut *= k->GetOutPut();
					}
					break;
				case DotPlus:
					C_InPut = Pre_Net_Head.front()->GetOutPut();
					for (auto k : Pre_Net_Head) {
						if (k != Pre_Net_Head.front())
							C_InPut += k->GetOutPut();
					}
					break;
				case OpsType::function:
					C_InPut = Pre_Net_Head.front()->GetOutPut();
					Activate(C_InPut);
					break;
				case RNNInPutSwitch:
					//RNN hidden state
					if ((idx == 0) ^ (Pre_Net_Head.front()->Net_Flag&Net_Flag_RNN_Initial_Step))
						C_InPut = Pre_Net_Head.back()->GetOutPut();
					else C_InPut = Pre_Net_Head.front()->GetOutPut();
					break;
				case SpatialConcatenate:
					C_InPut.Image_SpatialConcatenate(Pre_Net_Head.front()->GetOutPut(), Pre_Net_Head.back()->GetOutPut(), extra_param);
					break;
				case Scale:
					lstm_Max_Minus_Min[idx].Reset(1, C_InPut.GetCol());
					Pre_Net_Head.front()->GetOutPut().MinMax_Normalization(C_InPut, lstm_Max_Minus_Min[idx]);
					break;
				case OpsType::OutPut:
					if (cal_loss)
						return Cal_Loss();
					break;
				case InPut:
				default:
					break;
				}
				return 0;
			}
		}*IPU;
		struct __SGD__ TrainUnit :public Net
		{
			Mat C_b;
			Mat C_Biasb;
			Mat C_Momentumb;
			Mat C_Momentumb2;

			Mat C_Weigh;
			Mat C_BiasWeigh;
			Mat C_MomentumWeigh;
			Mat C_MomentumWeigh2;

			Mat C_OutPut;

			//Batch Normalized activations
			Mat*lstm_BN_OutPut;
			Mat*lstm_BN_SqrtVar;
			//memory optimize
			Mat*lstm_BN_InMinusMean;
			//BN average Mean,Variance
			Mat lstm_BN_avg_Mean;
			Mat lstm_BN_avg_Var;
			static const int BN_avg_miniBatch_Num = 1000;
			int BN_stack_Count = 0;
			Mat lstm_BN_Sum_Mean;
			Mat lstm_BN_Sum_Var;

			TrainUnit(int Node_Num, int Pre_Node_Num, int Batch, int b_Node = -1);
			TrainUnit();
			void Cuda_RefreshData(NetParam&Net_Param, double Gradient_Scale_Factor = 1.0, G_Net::TrainUnit*Send_Gradient = NULL);
			//void Gradient_Update(NetParam&Net_Param);
			void Data_WR(fstream&file, bool Write = true);
			void Matrix_WR(Mat&dst, fstream&file, bool Write) {
				floatType*W = Write ? dst.ReadFromDevice() : dst.GetMemData();
				//if (Write)
				File_WR(file, (char*)W, sizeof(floatType) * dst.GetRow() * dst.GetCol(), Write);
				/*else {
					for (int i = 0; i < dst.GetRow() * dst.GetCol(); i++) {
						double val; file.read((char*)&val, sizeof(double)); assert(file.gcount() == sizeof(double));
						assert(val < FLT_MAX && -FLT_MAX < val);
						W[i] = val;
					}
				}*/
				if (!Write)
					dst.WriteToDevice();
			}

			bool ConvCheck(Mat&Src, Mat*right, bool raw);

			//Base functions
			void Reduction(int Max_Step) {
				C_Biasb.Clear();
				C_Momentumb.Clear();
				C_Momentumb2.Clear();
				C_BiasWeigh.Clear();
				C_MomentumWeigh.Clear();
				C_MomentumWeigh2.Clear();
				C_OutPut.Clear();
				Net::Reduction(Max_Step);
			}
			double Forward(int idx, int IsTrain, bool cal_loss) {
				switch (Ops)
				{
					//Net=>Matrix
				case Transform:
					C_OutPut = C_Weigh % Pre_Net_Head.front()->GetOutPut() + C_b;
					Activate(C_OutPut);
					break;
					//Batch Normalization Transform:x0=(x-u)/√(σ),y=Ax0+B
				case BNTransform:
					//Batch estimate
					if (IsTrain) {
						Mat&InPut = Pre_Net_Head.front()->GetOutPut();
						Mat Mean = InPut.ScaleOneCol() / InPut.GetCol();
						/*Mat InMinusMean*/lstm_BN_InMinusMean[idx] = InPut - Mean;
						Mat Var = lstm_BN_InMinusMean[idx]._f(Pow2).ScaleOneCol() / InPut.GetCol();
						lstm_BN_SqrtVar[idx] = Var._f(Sqrt);
						lstm_BN_OutPut[idx] = lstm_BN_InMinusMean[idx] / lstm_BN_SqrtVar[idx];

						//sum Mean,Var at the end of epoch
						if (IsTrain == 2) {
							if (BN_stack_Count % BN_avg_miniBatch_Num == 0) {
								lstm_BN_Sum_Mean._ZeroMemory_Valid(InPut.GetRow(), 1);
								lstm_BN_Sum_Var._ZeroMemory_Valid(InPut.GetRow(), 1);
							}
							lstm_BN_Sum_Mean += Mean;
							lstm_BN_Sum_Var += Var;
							//reduce,Only cal when finish
							if ((++BN_stack_Count) % BN_avg_miniBatch_Num == 0) {
								lstm_BN_avg_Mean = lstm_BN_Sum_Mean / BN_avg_miniBatch_Num;
								lstm_BN_avg_Var = lstm_BN_Sum_Var * (1.0*InPut.GetCol() / (InPut.GetCol() - 1) / BN_avg_miniBatch_Num);
								BN_stack_Count = 0;
							}
						}
					}
					//inference
					else lstm_BN_OutPut[idx] = Pre_Net_Head.front()->GetOutPut().BN_Normalization(lstm_BN_avg_Mean, lstm_BN_avg_Var);

					C_OutPut = C_Weigh * lstm_BN_OutPut[idx] + C_b;
					Activate(C_OutPut);
					//restore
					ConvCheck(C_OutPut, NULL, false);
					break;
				default:
					break;
				}return 0;
			}
			bool Backward_Gradient(Mat*&result, Net*Pre_Net, int lstm_idx, int Max_Step) {
				switch (Ops)
				{
				case Base_Net::Transform:
					//Gradient continous
					*result = (!C_Weigh) % (Gradient);
					break;
				case Base_Net::BNTransform:
				{
					ui Batch = Gradient.GetCol();
					Mat dVar = C_Weigh * lstm_BN_OutPut[lstm_idx] * (-1.0 / Batch) / lstm_BN_SqrtVar[lstm_idx]._f(Pow2);
					Mat dMean = C_Weigh * Gradient.ScaleOneCol() / lstm_BN_SqrtVar[lstm_idx] + dVar * lstm_BN_InMinusMean[lstm_idx].ScaleOneCol();
					*result = Gradient * (C_Weigh / lstm_BN_SqrtVar[lstm_idx]) + dVar * lstm_BN_InMinusMean[lstm_idx] - dMean * (1.0 / Batch);

					lstm_BN_InMinusMean[lstm_idx] = NULL;lstm_BN_OutPut[lstm_idx] = NULL;lstm_BN_SqrtVar[lstm_idx] = NULL;
				}
				break;
				default:
					assert(false);
					return false;
					break;
				}
				return true;
			}
			void Calculate_Bias(int lstm_idx) {
				assert(!Pre_Net_Head.empty());
				switch (Ops)
				{
				case Transform:
					Derivate(lstm_idx);
					C_Biasb -= Gradient;
					C_BiasWeigh -= Gradient % (!(*GetPreOutPut(lstm_idx, Pre_Net_Head.front())));
					break;
				case BNTransform:
					Derivate(lstm_idx);
					ConvCheck(Gradient, NULL, true);
					//lstm_BN_InMinusMean = lstm_BN_OutPut[lstm_idx] * lstm_BN_SqrtVar[lstm_idx];
					//multiply Batch
					C_Biasb -= Gradient;// *(1.0*C_OutPut.GetCol() / Gradient.GetCol());
					lstm_BN_OutPut[lstm_idx] = (lstm_BN_OutPut[lstm_idx] * Gradient).ScaleOneCol();
					C_BiasWeigh -= lstm_BN_OutPut[lstm_idx];// *(1.0*C_OutPut.GetCol() / Gradient.GetCol());
					break;
				default:
					assert(false);
					break;
				}
			}
			void Update(NetParam& Net_Param, Net* dst_Net) {
				assert(!dst_Net || dynamic_cast<G_Net::TrainUnit*>(dst_Net));
				Cuda_RefreshData(Net_Param, Ops == BNTransform ? (1.0 * C_OutPut.GetCol() / Gradient.GetCol()) : 1.0, dynamic_cast<G_Net::TrainUnit*>(dst_Net));
			}
			Mat& GetOutPut(int lstm_idx) {
				return C_OutPut;
			}
			void Data_WR(fstream&file, Net**Net_Priority, int tot_Net_Num, bool Write, int Batch, int Max_Step) {
				//Write&Read Weight&b
				Data_WR(file, Write);
				//extra average Batch param
				if (Ops == BNTransform) {
					Matrix_WR(lstm_BN_avg_Mean, file, Write);
					Matrix_WR(lstm_BN_avg_Var, file, Write);
				}
			}
			void Data_Assignment(Net*dst_Net, int Max_Step) {
				TrainUnit*dst = (TrainUnit*)dst_Net;
				dst->C_Weigh = C_Weigh;
				dst->C_b = C_b;
				if (Ops == BNTransform) {
					dst->lstm_BN_avg_Mean = lstm_BN_avg_Mean;
					dst->lstm_BN_avg_Var = lstm_BN_avg_Var;
				}
			}
			void Assignment(double*Weight, double*b) {
				memcpy(C_Weigh.GetMemData(), Weight, sizeof(double)*C_Weigh.GetRow()*C_Weigh.GetCol());
				memcpy(C_b.GetMemData(), b, sizeof(double)*C_b.GetRow()*C_b.GetCol());
				C_Weigh.WriteToDevice();
				C_b.WriteToDevice();
			}
		}*TU, *OPU;
	protected:
		void Cuda_ComputerProc(Mat&In, bool DropOut = false);
		//double Cal_Loss(Mat&C_OutPut, Mat&C_Out, Mat*OutPut_Mask = NULL) {
		//	if (Output_Fun == softmax) {
		//		if (!OutPut_Mask)
		//			return (C_OutPut._f(Ln) *= C_Out).Sum().ReadFromDevice()[0] * -1.0;
		//		else return ((C_OutPut._f(Ln) *= C_Out) *= *OutPut_Mask).Sum().ReadFromDevice()[0] * -1.0;
		//	}
		//	else if (Output_Fun == sigmoid) {
		//		if (!OutPut_Mask)
		//			return (C_OutPut - C_Out).f(Pow2).Sum().ReadFromDevice()[0] / 2.0;
		//		else return ((C_OutPut - C_Out) *= *OutPut_Mask).f(Pow2).Sum().ReadFromDevice()[0] / 2.0;
		//	}
		//}
		//void Default_Param() {
		//	Net_Param["Speed"] = Mat(1, 1, 0.001);
		//	Net_Param["Forget"] = Mat(1, 1, 0.9);
		//	Net_Param["Beta1"] = Mat(1, 1, 0.9);
		//	Net_Param["Beta2"] = Mat(1, 1, 0.999);
		//	Net_Param["Hidden_DropOut"] = Mat(1, 1, 0.5);
		//	Net_Param["InPut_DropOut"] = Mat(1, 1, 0.2);
		//	Net_Param["OutPut_DropOut"] = Mat(1, 1, 0.5);
		//	Net_Param["Beta1_Pow"] = Mat(1, 1, 1.0);
		//	Net_Param["Beta2_Pow"] = Mat(1, 1, 1.0);
		//}
		//void Init_Param(string param) {
		//	Default_Param();
		//	//string prase
		//	size_t sz = param.size();
		//	int last = 0;
		//	string param_Name = "";
		//	for (int i = 0; i <= sz; i++) {
		//		if (i == sz || param[i] == ' ') {
		//			if (Net_Param.find(param_Name) != Net_Param.end())
		//				Net_Param[param_Name].Reset(1, 1, atof(param.substr(last, i - last).c_str()));
		//			else if (!param_Name.empty())printf("not find '%s' parameter \n", param_Name.c_str());
		//			last = i + 1;
		//		}
		//		else if (param[i] == ':') {
		//			param_Name = param.substr(last, i - last);
		//			last = i + 1;
		//		}
		//	}
		//	Net_Param["One_minus_Beta1"] = Mat(1, 1, 1 - Net_Param["Beta1"][0]);
		//	Net_Param["One_minus_Beta2"] = Mat(1, 1, 1 - Net_Param["Beta2"][0]);
		//}
		//use for Adaptive Speed
		struct Adam_param
		{
			double Data;
			double Momentum;
			double Momentum2;
			double Beta1;
			double Beta2;
			Adam_param() {}
			Adam_param(double Init_val) {
				Data = Init_val;
				Momentum = Momentum2 = 0;
				Beta1 = Beta2 = 1;
			}
			double Update(double val, double Speed, double beta1, double beta2) {
				Beta1 *= beta1;
				Beta2 *= beta2;
				Momentum = beta1 * Momentum + (1.0 - beta1)*val;
				Momentum2 = beta2 * Momentum2 + (1.0 - beta2)*val*val;
				return Speed * Momentum / (1.0 - Beta1) / sqrt(Momentum2 / (1.0 - Beta2) + DBL_MIN);
			}
			void Liner_Update(double Last_prec, double ins_prec, Mat&Speed) {
				//Speed Change
				double _Speed = Data;
				if (ins_prec < Last_prec)_Speed *= -0.01;
				//else if (Last_Test > best_Test)_Speed *= 1.1;
				else _Speed *= 0.0001;
				Speed = Mat(1, 1, (floatType)(Data += Update(_Speed, Data*0.001, 0.9, 0.999)));
			}
		};
	public:
		//TrainUnit&getTU(int id) const { return TU[id]; }
		//TrainUnit&getOPU() const { return *OPU; }
		//bool IsVaild();
		void Func_Init(int MaxRandomStates_rows);
		G_Net() {};
		G_Net(int L, int*S, int BatchNum = 1, _Activation_Func Output_Fun = softmax, double Sigmoid_eps = 0.5, int Negative_OutPut_No = -1, _Activation_Func*Hidden_Layer_Func = NULL);
		void Disponse();
		G_Net(const char*filePath, int BatchNum = 1, _Activation_Func Output_Fun = softmax, double Sigmoid_eps = 0.5, int Negative_OutPut_No = -1);
		/*
		Net_Param:: Param1:val1 Param2:val2
		Param include:: Speed Forget Beta1 Beta2 Hidden_DropOut InPut_DropOut
		*/
		int Cuda_Train(Param*param, int paramNum, int trainNum, const char*Net_Param, const char*Path = "");
		void DataWrite(const char*filePath);
		void DataWrite(const wchar_t*filePath);
		void WriteToCPU();
		//void WriteTo_Net();
		double Test(Mat&In, Mat&Out, int testNum);
		//bool DataRead(const char*filePath);
		void Response(Param&param);
	};
};
