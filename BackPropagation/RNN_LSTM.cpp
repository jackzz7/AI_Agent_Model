#define BP_EXPORTS

#include"Param.h"

#include<iostream>
#include<conio.h>

#include"RNN_LSTM.h"


#include<time.h>

using namespace _Net_;

//void RNN_LSTM::Train_LSTM(LSTM_Param*param, int paramNum, int trainCnt, const char*Net_param, double(*Data_Table)[30000], const char*Path) {
//	Init_Param(Net_param);
//	int TrainCount = 0;
//	int InPutNum = S[0] - S[1];
//
//	Mat C_Backward_H(S[1],Num);
//	Mat C_Backward_C(S[1], Num);
//
//	Mat C_Sum_Output_Num(1, 1);
//
//
//	Mat*All_In = new Mat[Max_Length];
//	Mat*All_Out = new Mat[Max_Length];
//	Mat*All_OutPut_mask = new Mat[Max_Length];
//	Mat*Test_In = new Mat[Max_Length];
//	Mat*Test_Out = new Mat[Max_Length];
//	Mat*Test_OutPut_mask = new Mat[Max_Length];
//	double**_In = new double*[InPutNum];
//	double**_Out = new double*[S[L - 1]];
//	double**_Output_mask = new double*[S[L - 1]];
//	for (int i = 0; i < InPutNum; i++) {
//		_In[i] = new double[paramNum];
//	}
//	for (int i = 0; i < S[L - 1]; i++) {
//		_Out[i] = new double[paramNum];
//		_Output_mask[i] = new double[paramNum];
//	}
//	Mat*C_In = new Mat[Max_Length];
//	Mat*C_Out = new Mat[Max_Length];
//	Mat*OutPut_mask = new Mat[Max_Length];
//	int tot_Train_Output_Num = 0, tot_Test_Output_Num = 0;
//	int tot_Train_Positive_Num = 0, tot_Test_Positive_Num = 0;
//	int _trainCnt = trainCnt / Num * Num, _testNum = (paramNum - _trainCnt) / Num * Num;
//	for (int i = 0; i < Max_Length; i++) {
//		for (int l = 0; l < paramNum; l++) {
//			for (int k = 0; k < InPutNum; k++)
//				//Data Compression
//				if (Data_Table&&param[l].param[i]._DataIn->Count) {
//					_In[k][l] = Data_Table[k][(int)(*param[l].param[i]._DataIn)[0]];
//				}
//				else if (param[l].param[i]._DataIn->Count == InPutNum)
//					_In[k][l] = (*param[l].param[i]._DataIn)[k];
//				else _In[k][l] = 0;
//			for (int k = 0; k < S[L - 1]; k++)
//				if (param[l].param[i]._DataOut->Count == S[L - 1])
//				{
//					_Out[k][l] = (*param[l].param[i]._DataOut)[k];
//					_Output_mask[k][l] = 1;
//				}
//				else {
//					_Out[k][l] = 0;
//					_Output_mask[k][l] = 0;
//				}
//		}
//		C_In[i].Reset(S[0], Num);
//		C_Out[i].Reset(S[L - 1], Num);
//		OutPut_mask[i].Reset(S[L - 1], Num);
//		for (int j = 0; j < paramNum / Num; j++) {
//			C_Out[i].Reset(S[L - 1], Num, j*Num, _Out);
//			OutPut_mask[i].Reset(S[L - 1], Num, j*Num, _Output_mask);
//			double val = OutPut_mask[i].ScaleOneCol(true).f(Bool).Sum().ReadFromDevice()[0];
//			int val1 = (C_Out[i] *= Positive_OutPut_mask).Sum().ReadFromDevice()[0];
//			if (j < trainCnt / Num)
//				tot_Train_Output_Num += val, tot_Train_Positive_Num += val1;
//			else tot_Test_Output_Num += val, tot_Test_Positive_Num += val1;
//		}
//		All_In[i].Reset(InPutNum, _trainCnt, _In);
//		All_Out[i].Reset(S[L - 1], _trainCnt, _Out);
//		All_OutPut_mask[i].Reset(S[L - 1], _trainCnt, _Output_mask);
//		Test_In[i].Reset(InPutNum, _testNum, _trainCnt, _In);
//		Test_Out[i].Reset(S[L - 1], _testNum, _trainCnt, _Out);
//		Test_OutPut_mask[i].Reset(S[L - 1], _testNum, _trainCnt, _Output_mask);
//	}
//
//	double Sum = 0, LastSum = -1, N = 0.0001*S[L - 1] * Num;
//	clock_t Start = clock();
//	while (true)
//	{
//		//epoch
//		for (int g = 0; g < trainCnt / Num; g++) {
//			TrainCount++;
//			Sum = 0.0;
//			_CellState->C_InPut._ZeroMemory();
//			CellOutPut->C_InPut._ZeroMemory();
//			C_Sum_Output_Num._ZeroMemory();
//			//random order of train examples
//			RandomGenerator.GenerateRandom();
//			for (int j = 0; j < Max_Length; j++) {
//				C_In[j].RandMatrix(All_In[j], RandomGenerator, S[1]);
//				C_Out[j].RandMatrix(All_Out[j], RandomGenerator);
//				OutPut_mask[j].RandMatrix(All_OutPut_mask[j], RandomGenerator);
//			}
//			//dymatic length
//			for (int j = 0; j < Max_Length; j++) {
//				//Step DropOut
//				/*CellUpdate->lstm_DropOut[j] = CellUpdate->C_DropOut.f(DropOut_Bernoulli, Net_Param["Hidden_DropOut"][0]);
//				H_Input->lstm_DropOut[j] = H_Input->C_DropOut.f(DropOut_Bernoulli, Net_Param["InPut_DropOut"][0], S[1]);
//				CellOutPut->lstm_DropOut[j] = CellOutPut->C_DropOut.f(DropOut_Bernoulli, Net_Param["OutPut_DropOut"][0]);*/
//
//				
//				ComputerProc(j, C_In[j], C_Out[j],true);
//				//whether have Gradient
//				OutputLayer->lstm_Output[j] *= OutPut_mask[j];
//
//				//Cal Total Loss
//				Sum += Cal_Loss(OutputLayer->C_OutPut, C_Out[j],&OutPut_mask[j]);
//				//Summary Output Num
//				C_Sum_Output_Num += OutPut_mask[j].ScaleOneCol(true).f(Bool).Sum();
//			}
//			C_Backward_H._ZeroMemory();
//			C_Backward_C._ZeroMemory();
//			for (int j = Max_Length - 1; j > -1; j--) {
//				Backward_Gradient(j, C_In[j],C_Backward_H, C_Backward_C);
//			}
//			Net_Param["Beta1_Pow"] *= Net_Param["Beta1"];
//			Net_Param["Beta2_Pow"] *= Net_Param["Beta2"];
//			ForgetGate->Cuda_RefreshData(Net_Param);
//			InputGate->Cuda_RefreshData(Net_Param);
//			CellUpdate->Cuda_RefreshData(Net_Param);
//			OutputGate->Cuda_RefreshData(Net_Param);
//			//OutPut Num
//			OutputLayer->Cuda_RefreshData(Net_Param);
//
//			std::cout << "Loss:" << Sum << ' ' << LastSum - Sum << ' ' << TrainCount << ' ' << N << '\n';
//			LastSum = Sum;
//		}
//		//train examples
//		Test(All_In, All_Out, All_OutPut_mask, tot_Train_Output_Num, tot_Train_Positive_Num, _trainCnt / Num);
//		//test examples
//		Test(Test_In, Test_Out, Test_OutPut_mask, tot_Test_Output_Num, tot_Test_Positive_Num, _testNum / Num);
//
//		if (_kbhit()) {
//			if (_getch() == (int)'c')
//				break;
//		}
//	}
//
//}
//void LSTM::Response(LSTM_Param&param)
//{
//	_CellState->Clear();
//	CellOutPut->Clear();
//	for (int j = 0; j < param.Count; j++) {
//		memset(In, 0, sizeof(double)*S[0]);
//		param.param[j]._DataIn->DeCode(In + S[1]);
//		ComputerProc(j, Out);
//		param.param[j]._DataOut->EnCode(OutputLayer->OutPut, S[L - 1]);
//	}
//}
//double RNN_LSTM::Test(Mat*C_In, Mat*C_Out, Mat*OutPut_mask, int OutPut_Num, int Positive_Num, int testNum) {
//	int cor = 0, tot = OutPut_Num, posi = 0, prec = 0;
//	if (Output_Fun == sigmoid)cor = tot;
//	Mat In(S[0], Num), Out(S[L - 1], Num), mask(S[L - 1], Num);
//	for (int i = 0; i < testNum; i++) {
//		_CellState->C_InPut._ZeroMemory();
//		CellOutPut->C_InPut._ZeroMemory();
//		for (int j = 0; j < Max_Length; j++) {
//			In.Append(C_In[j], S[1], i*Num);
//			Out.Append(C_Out[j], 0, i*Num);
//			mask.Append(OutPut_mask[j], 0, i*Num);
//
//			ComputerProc(j, In, Out);
//			if (Output_Fun == sigmoid) {
//				cor -= ((OutputLayer->C_OutPut - Out).f(Compare, Sigmoid_eps) *= mask).ScaleOneCol(true).f(Bool).Sum().ReadFromDevice()[0];
//			}
//			else if (Output_Fun == softmax) {
//				Mat tmp = (OutputLayer->C_OutPut.SoftMax(true) *= mask);
//				posi += (tmp *= Positive_OutPut_mask).Sum().ReadFromDevice()[0];
//				//乘以正样本掩码
//				cor += (OutputLayer->C_OutPut *= Out).Sum().ReadFromDevice()[0];
//				prec += (OutputLayer->C_OutPut *= Positive_OutPut_mask).Sum().ReadFromDevice()[0];
//			}
//		}
//	}
//	if (Output_Fun == softmax)
//		printf("Test:%0.2lf Pre:%0.2lf Rec:%0.2lf\n", 100.0*cor / tot, 100.0*prec / (posi+eps), 100.0*prec / Positive_Num);
//	else printf("Test:%0.2lf\n", 100.0*cor / tot);
//	return 100.0*cor / tot;
//}

