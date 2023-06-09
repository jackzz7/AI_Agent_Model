#pragma once
#include"_Net.h"

namespace Net_CPU
{
	class LSTM_Param;
	class LSTM :_Net
	{
	protected:
		int _H;
		int Max_Length;
		struct LSTMInput :InPutUnit {
			double**lstm_Input;
			LSTMInput(int*S,int No,int Max_Length) :InPutUnit(S,No) {
				lstm_Input = new double*[Max_Length];
				for (int i = 0; i < Max_Length; i++)
					lstm_Input[i] = new double[S[No]]{ 0 };
			}
			void Save(int idx) {
				memcpy(lstm_Input[idx], InPut, sizeof(double)*S[No]);
			}
		};
		LSTMInput*H_Input;
		LSTMInput*CellState, *_CellState;
		LSTMInput*CellOutPut;

		struct LSTMUnit:TrainUnit {
			double**lstm_Output;
			LSTMUnit() :TrainUnit() {}
			LSTMUnit(int*S, int No,int Max_Length) :TrainUnit(S, No) {
				lstm_Output = new double*[Max_Length];
				for (int i = 0; i < Max_Length; i++)
					lstm_Output[i] = new double[S[No]]{ 0 };
			}
			void Save(int idx) {
				memcpy(lstm_Output[idx], OutPut, sizeof(double)*S[No]);
			}
			void Save_Gradent(int idx, double*Out, _Activation_Func Output_Fun) {
				if (Output_Fun == softmax)
					for (int i = 0; i < S[No]; i++)
						lstm_Output[idx][i] = (OutPut[i] - Out[i]);

				else if (Output_Fun == sigmoid)
					for (int i = 0; i < S[No]; i++) {
						double A = exp(-Net[i]);
						double B = A / ((1 + A)*(1 + A));
						lstm_Output[idx][i] = (OutPut[i] - Out[i])*B;
					}
			}
		};
		LSTMUnit*ForgetGate;
		LSTMUnit*InputGate;
		LSTMUnit*CellUpdate;
		LSTMUnit*OutputGate;

		LSTMUnit*OutputLayer;

		double*temp_H;
		double*temp_G;
		double**temp_H_H_C;
		double**temp_H_H_O;
		double*temp_C;

		
		void PointWiseOps(double*A, double*B, double*Out, int Num, bool isProduct=true,_Activation_Func R_Func=null) {
			for (int i = 0; i < Num; i++) {
				if (isProduct) {
					if (R_Func == tanh) 
						Out[i] = A[i] * Tanh(B[i]);
					else  Out[i] = A[i] * B[i];
				}
				else Out[i] = A[i] + B[i];
			}
		}
	public:
		//S={h+x,h,Out}
		//Input,forget gate,input gate,C,output gate
		LSTM(int*_S,int Max_Len,int BatchNum, _Activation_Func Output_Fun) :_Net(3, _S,BatchNum,Output_Fun) {
			_H = S[1];
			S[0] += S[1];
			Max_Length = Max_Len;
			H_Input = new LSTMInput(S, 0, Max_Length);
			CellState = new LSTMInput(S, 1, Max_Length);
			_CellState = new LSTMInput(S, 1, Max_Length);
			CellOutPut = new LSTMInput(S, 1, Max_Length);
			H_Input->Clear();
			CellState->Clear();
			_CellState->Clear();
			CellOutPut->Clear();

			ForgetGate = new LSTMUnit(S, 1, Max_Length);
			InputGate = new LSTMUnit(S, 1, Max_Length);
			CellUpdate = new LSTMUnit(S, 1, Max_Length);
			OutputGate = new LSTMUnit(S, 1, Max_Length);
			OutputLayer = new LSTMUnit(S, 2, Max_Length);

			temp_H = new double[_H] {0};
			temp_G = new double[_H] {0};
			temp_H_H_C = new double*[_H] {0};
			temp_H_H_O = new double*[_H] {0};
			for (int i = 0; i < _H; i++) {
				temp_H_H_C[i] = new double[_H] {0};
				temp_H_H_O[i] = new double[_H] {0};
			}
			temp_C = new double[_H] {0};
		}

		void ComputerProc(int idx, double*Out) {
			memcpy(CellState->InPut, _CellState->InPut, sizeof(double)*S[1]);
			memcpy(In, CellOutPut->InPut, sizeof(double)*S[1]);
			memcpy(H_Input->InPut, In, sizeof(double)*S[0]);

			CellState->Save(idx);

			//forget gate
			ForgetGate->ComputeOutPut(sigmoid, H_Input);
			PointWiseOps(ForgetGate->OutPut, CellState->InPut, CellState->InPut, _H);
			//update cell,input gate
			InputGate->ComputeOutPut(sigmoid, H_Input);
			CellUpdate->ComputeOutPut(tanh, H_Input);
			//X
			PointWiseOps(InputGate->OutPut, CellUpdate->OutPut, _CellState->InPut, _H);
			//+
			PointWiseOps(_CellState->InPut, CellState->InPut, _CellState->InPut, _H, false);
			//output gate
			OutputGate->ComputeOutPut(sigmoid, H_Input);
			PointWiseOps(OutputGate->OutPut, _CellState->InPut, CellOutPut->InPut, _H, true, tanh);

			//Output
			OutputLayer->ComputeOutPut(Output_Fun, CellOutPut);

			//save params
			H_Input->Save(idx);
			_CellState->Save(idx);
			CellOutPut->Save(idx);
			ForgetGate->Save(idx);
			InputGate->Save(idx);
			CellUpdate->Save(idx);
			OutputGate->Save(idx);
			OutputLayer->Save_Gradent(idx, Out, Output_Fun);
		}

		void Backward_Gradient(int idx,double*Backward_H,double*Backward_C,bool IsOutput) {
			//Output J
			for (int i = 0; i < _H; i++) {
				temp_H[i] = Backward_H[i];
				if (IsOutput)
					for (int j = 0; j < S[L - 1]; j++) {
						temp_H[i] += OutputLayer->Weigh[j][i] * OutputLayer->lstm_Output[idx][j];
					}
			}
			//Gradient Variables
			for (int i = 0; i < _H; i++) {
				double v = Tanh(_CellState->lstm_Input[idx][i]);
				temp_G[i] = OutputGate->lstm_Output[idx][i] * (1.0 - v * v);
			}
			for (int i = 0; i < _H; i++) {
				temp_C[i] = temp_H[i] * temp_G[i] + Backward_C[i];
			}

			//Cal bias
			for (int i = 0; i < _H; i++) {
				double F = CellState->lstm_Input[idx][i] * ForgetGate->lstm_Output[idx][i] * (1.0 - ForgetGate->lstm_Output[idx][i]);
				double I = CellUpdate->lstm_Output[idx][i] * InputGate->lstm_Output[idx][i] * (1.0 - InputGate->lstm_Output[idx][i]);
				double C = InputGate->lstm_Output[idx][i] * (1.0 - CellUpdate->lstm_Output[idx][i] * CellUpdate->lstm_Output[idx][i]);
				double O = Tanh(_CellState->lstm_Input[idx][i]) *  OutputGate->lstm_Output[idx][i] * (1.0 - OutputGate->lstm_Output[idx][i]);
				for (int j = 0; j < S[0]; j++) {
					//weigh
					ForgetGate->BiasWeigh[i][j] -= temp_C[i] * F * H_Input->lstm_Input[idx][j];
					InputGate->BiasWeigh[i][j] -= temp_C[i] * I * H_Input->lstm_Input[idx][j];
					CellUpdate->BiasWeigh[i][j] -= temp_C[i] * C * H_Input->lstm_Input[idx][j];
					OutputGate->BiasWeigh[i][j] -= temp_H[i] * O * H_Input->lstm_Input[idx][j];
				}
				//b
				ForgetGate->Biasb[i] -= temp_C[i] * F;
				InputGate->Biasb[i] -= temp_C[i] * I;
				CellUpdate->Biasb[i] -= temp_C[i] * C;
				OutputGate->Biasb[i] -= temp_H[i] * O;

				for (int j = 0; j < _H; j++) {
					temp_H_H_C[i][j] = F * ForgetGate->Weigh[i][j] + I * InputGate->Weigh[i][j] + C * CellUpdate->Weigh[i][j];
					temp_H_H_O[i][j] = O * OutputGate->Weigh[i][j];
				}
			}
			//Output layer
			if (IsOutput)
				for (int i = 0; i < S[L - 1]; i++) {
					for (int j = 0; j < _H; j++) {
						//weigh
						OutputLayer->BiasWeigh[i][j] -= OutputLayer->lstm_Output[idx][i]*CellOutPut->lstm_Input[idx][j];
					}
					//b
					OutputLayer->Biasb[i] -= OutputLayer->lstm_Output[idx][i];
				}
			for (int i = 0; i < _H; i++) {
				Backward_H[i] = 0.0;
				for (int j = 0; j < _H; j++) {
					Backward_H[i] += temp_H_H_C[j][i] * temp_C[j] + temp_H_H_O[j][i] * temp_H[j];
				}
			}
			for (int i = 0; i < _H; i++) {
				Backward_C[i] = ForgetGate->lstm_Output[idx][i] * temp_C[i];
			}
		}
		void Train_LSTM(LSTM_Param*param, int paramNum,int trainCnt, double Speed=0.001, double ForgetFactor=0.9, double Beta1=0.9, double Beta2=0.999);
		void Response(LSTM_Param&param);
		double Test(LSTM_Param*param, int testNum);
	};
};