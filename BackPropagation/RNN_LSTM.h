#pragma once
#include"G_Net.h"

using namespace Net_CPU;

namespace _Net_
{
	using namespace _CUDA_;
	//class LSTM_Param;
	class __SGD__ RNN_LSTM :protected G_Net
	{
	protected:

	public:
		struct LSTMInput :public InPutUnit {
			Mat*lstm_Input;
			LSTMInput() { lstm_Max_Minus_Min = lstm_Input = NULL; }
			LSTMInput(int Max_Step,int Node,int Batch) :InPutUnit(Node,Batch) {
				lstm_Input = new Mat[Max_Step];
				lstm_Max_Minus_Min = new Mat[Max_Step];
			}
			~LSTMInput() {
				delete[] lstm_Input;
				delete[] lstm_Max_Minus_Min;
			}

			//Net functions
			void Reduction(int Max_Step) {
				for (int i = 0; i < Max_Step; i++)
					lstm_Input[i].Clear();
				InPutUnit::Reduction(Max_Step);
			}
			void LSTM_Save(int idx) {
				if (CheckSaveCondition())
					lstm_Input[idx] = C_InPut;
			}
			Mat& GetOutPut(int lstm_idx) {
				return ((lstm_idx == -1) ? C_InPut : lstm_Input[lstm_idx]);
			}
		};

		struct LSTMUnit :TrainUnit {
			Mat*lstm_Output;
			void clear() { lstm_BN_InMinusMean = lstm_BN_SqrtVar = lstm_BN_OutPut = lstm_Output = NULL; }
			LSTMUnit() { clear(); }
			LSTMUnit(int Max_Step, int Node, int Pre_Node, int Batch, int b_Node = -1, bool BN_Node = false) :TrainUnit(Node, Pre_Node, Batch, b_Node) {
				clear();
				lstm_Output = new Mat[Max_Step];
				if (BN_Node) {
					//Batch Normalization
					lstm_BN_OutPut = new Mat[Max_Step];
					lstm_BN_SqrtVar = new Mat[Max_Step];
					lstm_BN_InMinusMean = new Mat[Max_Step];

					lstm_BN_avg_Mean.Reset(Node, 1); lstm_BN_avg_Mean._ZeroMemory();
					lstm_BN_avg_Var.Reset(Node, 1); lstm_BN_avg_Var.f(::Assignment, 1.0);
				}
			}
			~LSTMUnit() {
				delete[] lstm_Output;
				delete[] lstm_BN_OutPut;
				delete[] lstm_BN_SqrtVar;
				delete[] lstm_BN_InMinusMean;
			}
			
			//Net functions
			void Reduction(int Max_Step) {
				for (int i = 0; i < Max_Step; i++) {
					lstm_Output[i].Clear();
					lstm_BN_OutPut[i].Clear();
					lstm_BN_SqrtVar[i].Clear();
					lstm_BN_InMinusMean[i].Clear();
				}
				TrainUnit::Reduction(Max_Step);
			}
			void LSTM_Save(int idx) {
				if (CheckSaveCondition())
					lstm_Output[idx] = C_OutPut;
			}
			Mat& GetOutPut(int lstm_idx) {
				return ((lstm_idx == -1) ? C_OutPut : lstm_Output[lstm_idx]);
			}
		};
		/*LSTMUnit*ForgetGate;
		LSTMUnit*InputGate;
		LSTMUnit*CellUpdate;
		LSTMUnit*OutputGate;

		LSTMUnit*OutputLayer;*/


		/*void PointWiseOps(double*A, double*B, double*Out, int Num, bool isProduct = true, _Activation_Func R_Func = null) {
			for (int i = 0; i < Num; i++) {
				if (isProduct) {
					if (R_Func == tanh)
						Out[i] = A[i] * Tanh(B[i]);
					else  Out[i] = A[i] * B[i];
				}
				else Out[i] = A[i] + B[i];
			}
		}*/
	public:
		RNN_LSTM() {};
		//S={h+x,h,Out}
		//Input,forget gate,input gate,C,output gate
		/*RNN_LSTM(int*_S, int Max_Len, int BatchNum, _Activation_Func Output_Fun, double Sigmoid_eps = 0.5, int Negative_OutPut_No = -1,_Activation_Func Hidden_Func=tanh) :G_Net(3, _S, BatchNum, Output_Fun,Sigmoid_eps,Negative_OutPut_No) {
			_H = S[1];
			S[0] += S[1];
			Max_Length = Max_Len;
			this->Hidden_Func = Hidden_Func;
			D_Hidden_Func = (Hidden_Func == tanh) ? Device_Func::D_Tanh : Device_Func::D_ReLU;
			H_Input = new LSTMInput(Max_Length,S[0],Num);
			CellState = new LSTMInput(Max_Length,S[1],Num);
			_CellState = new LSTMInput(Max_Length,S[1],Num);
			CellOutPut = new LSTMInput(Max_Length,S[1],Num);
			
			ForgetGate = new LSTMUnit(Max_Length,S[1],S[0],BatchNum);
			InputGate = new LSTMUnit(Max_Length, S[1], S[0], BatchNum);
			CellUpdate = new LSTMUnit(Max_Length, S[1], S[0], BatchNum);
			OutputGate = new LSTMUnit(Max_Length, S[1], S[0], BatchNum);
			OutputLayer = new LSTMUnit(Max_Length, S[2], S[1], BatchNum);
		}*/

		//void ComputerProc(int idx, Mat&In, Mat&Out, bool DropOut = false) {
		//	CellState->C_InPut = _CellState->C_InPut;
		//	In = CellOutPut->C_InPut;
		//	Mat masked_In = In;
		//	/*if (DropOut)
		//		masked_In *= H_Input->C_DropOut;*/

		//	CellState->Save(idx);
		//	//forget gate
		//	ForgetGate->Cuda_ComputeOutPut(sigmoid, masked_In);
		//	CellState->C_InPut *= ForgetGate->C_OutPut;
		//	//update cell,input gate
		//	InputGate->Cuda_ComputeOutPut(sigmoid, masked_In);
		//	CellUpdate->Cuda_ComputeOutPut(Hidden_Func, masked_In);
		//	//DropOut
		//	Mat CellUpdate_masked_OutPut = CellUpdate->C_OutPut;
		//	/*if (DropOut)
		//		CellUpdate_masked_OutPut *= CellUpdate->C_DropOut;*/
		//	//X
		//	_CellState->C_InPut = InputGate->C_OutPut;
		//	_CellState->C_InPut *= CellUpdate_masked_OutPut;
		//	//+
		//	_CellState->C_InPut += CellState->C_InPut;
		//	//output gate
		//	OutputGate->Cuda_ComputeOutPut(sigmoid, masked_In);
		//	Device_Func _Func = (Hidden_Func == tanh) ? Device_Func::Tanh : Device_Func::ReLU;
		//	CellOutPut->C_InPut = _CellState->C_InPut._f(_Func);
		//	//save tanh(c)
		//	_CellState->Save(idx, &CellOutPut->C_InPut);

		//	CellOutPut->C_InPut *= OutputGate->C_OutPut;
		//	Mat masked_CellOutPut = CellOutPut->C_InPut;
		//	/*if (DropOut)
		//		masked_CellOutPut *= CellOutPut->C_DropOut;*/
		//	//Output
		//	OutputLayer->Cuda_ComputeOutPut(Output_Fun, masked_CellOutPut);

		//	//save params
		//	//H_Input->Save(idx);
		//	CellOutPut->Save(idx);
		//	ForgetGate->Save(idx);
		//	InputGate->Save(idx);
		//	CellUpdate->Save(idx);
		//	OutputGate->Save(idx);
		//	OutputLayer->Save_Gradient(idx, Out, Output_Fun);
		//}
		//void Backward_Gradient(int idx,Mat&C_In, Mat&C_Backward_H, Mat&C_Backward_C) {
		//	////Output J
		//	////Mat temp_H = C_Backward_H + (((!OutputLayer->C_Weigh)*OutputLayer->lstm_Output[idx]) *= CellOutPut->lstm_DropOut[idx]);
		//	////Gradient Variables
		//	//Mat temp_G = OutputGate->lstm_Output[idx];
		//	//temp_G *= _CellState->lstm_Input[idx]._f(D_Hidden_Func);
		//	////Mat temp_C = (temp_G *= temp_H) + C_Backward_C;

		//	////Cal bias
		//	//Mat F = CellState->lstm_Input[idx]; F *= ForgetGate->lstm_Output[idx]._f(D_Sigmoid);
		//	//Mat I = CellUpdate->lstm_Output[idx]; I *= InputGate->lstm_Output[idx]._f(D_Sigmoid); 
		//	//Mat C = InputGate->lstm_Output[idx]; C *= CellUpdate->lstm_Output[idx]._f(D_Hidden_Func);
		//	//Mat O = _CellState->lstm_Input[idx]; O *= OutputGate->lstm_Output[idx]._f(D_Sigmoid);
		//	////DropOut
		//	///*I *= CellUpdate->lstm_DropOut[idx];
		//	//C *= CellUpdate->lstm_DropOut[idx];*/


		//	//C_Backward_H = (!ForgetGate->C_Weigh).ResetRows(_H)*(F *= temp_C) + (!InputGate->C_Weigh).ResetRows(_H)*(I *= temp_C) + (!CellUpdate->C_Weigh).ResetRows(_H)*(C *= temp_C) + (!OutputGate->C_Weigh).ResetRows(_H)*(O *= temp_H);
		//	//C_Backward_C = ForgetGate->lstm_Output[idx];
		//	//C_Backward_C *= temp_C;
		//	//Mat _In = C_In;
		//	////_In = !(_In *= H_Input->lstm_DropOut[idx]);
		//	//ForgetGate->C_BiasWeigh -= F * _In;
		//	//InputGate->C_BiasWeigh -= I * _In;
		//	//CellUpdate->C_BiasWeigh -= C * _In;
		//	//OutputGate->C_BiasWeigh -= O * _In;
		//	//ForgetGate->C_Biasb -= F;
		//	//InputGate->C_Biasb -= I;
		//	//CellUpdate->C_Biasb -= C;
		//	//OutputGate->C_Biasb -= O;

		//	//Mat masked_CellOutPut = CellOutPut->lstm_Input[idx];
		//	////masked_CellOutPut *= CellOutPut->lstm_DropOut[idx];
		//	//OutputLayer->C_BiasWeigh -= OutputLayer->lstm_Output[idx] * (!masked_CellOutPut);
		//	//OutputLayer->C_Biasb -= OutputLayer->lstm_Output[idx];
		//}
		void Train_LSTM(LSTM_Param*param, int paramNum, int trainCnt, const char*Net_Param,double(*Data_Table)[30000]=NULL,const char*Path="");
		void Response(LSTM_Param&param);
		double Test(Mat*C_In, Mat*C_Out, Mat*OutPut_mask,int OutPut_Num, int Positive_Num, int testNum);

		//LSTM
		//Net Based
		static Net* LSTM_Net(Net*Net_In,ui Hidden_Node_Num) {
			ui Hidden = Hidden_Node_Num;
			Net*H_In = new Net(Net_In->Net_Node_Num + Hidden);
			Net*Forget_Gate = (new Net(Hidden, Base_Net::sigmoid))->Add_Forward(Transform, H_In);
			Net*InPut_Gate = (new Net(Hidden, Base_Net::sigmoid))->Add_Forward(Transform, H_In);
			Net*Cell_Update = (new Net(Hidden, Base_Net::tanh))->Add_Forward(Transform, H_In);
			Net*OutPut_Gate = (new Net(Hidden, Base_Net::sigmoid))->Add_Forward(Transform, H_In);
			//Cell Update DropOut
			//Cell_Update = (new Net(Hidden))->Add_Mask(Net_Flag_Hidden_DropOut, Cell_Update);
			Net*Old_In = (new Net(Hidden));
			Net*New_In = (new Net(Hidden))->Add_Pair_Forward(DotProduct, InPut_Gate, Cell_Update);
			Net*Cell = (new Net(Hidden))->Add_Pair_Forward(DotPlus, Old_In, New_In);
			Old_In->Add_Pair_Forward(DotProduct, Cell, Forget_Gate, Net_Flag_Reconnect);
			Net*Cell_Out = (new Net(Hidden, Base_Net::tanh))->Add_Forward(OpsType::function, Cell);
			Net*Hidden_State = (new Net(Hidden))->Add_Pair_Forward(DotProduct, OutPut_Gate, Cell_Out);
			H_In->Add_Pair_Forward(OpsType::Concatenate, Hidden_State, Net_In, Net_Flag_Reconnect);
			return Hidden_State;
		}
	};
};