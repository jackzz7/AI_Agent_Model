#define NET_EXPORTS

#include<iostream>
#include<conio.h>

#include"LSTM.h"
#include"Param.h"

#include<time.h>


using namespace Net_CPU;

void LSTM::Train_LSTM(LSTM_Param*param, int paramNum, int trainCnt,double Speed, double ForgetFactor, double Beta1, double Beta2) {
	int TrainCount = 0;
	double M_Beta1 = Beta1, M_Beta2 = Beta2;
	double*ExamIn, *ExamOut;
	double*Backward_H, *Backward_C;

	ExamIn = new double[S[0]];
	ExamOut = new double[S[L - 1]];
	Backward_H = new double[S[1]];
	Backward_C = new double[S[1]];
	

	In = ExamIn;
	Out = ExamOut;


	double E = 0, Sum = 0, LastSum = -1, N = 0.0001*S[L - 1] * Num;
	clock_t Start = clock();
	while (true)
	{
		//epoch
		for (int g = 0; g < trainCnt / Num; g++) {
			TrainCount++;
			Sum = 0.0;
			int OutNum = 0;
			for (int t = 0; t < Num; t++) {
				memset(Backward_H, 0, sizeof(double)*S[1]);
				memset(Backward_C, 0, sizeof(double)*S[1]);
				_CellState->Clear();
				CellOutPut->Clear();

				for (int j = 0; j < param[g*Num+t].Count; j++) {
					memset(In, 0, sizeof(double)*S[0]);
					memset(Out, 0, sizeof(double)*S[L - 1]);
					param[g*Num + t].param[j]._DataIn->DeCode(In + S[1]);
					param[g*Num + t].param[j]._DataOut->DeCode(Out);

					ComputerProc(j, Out);

					//Cal Total Loss
					if (param[g*Num + t].param[j]._DataOut->Count == S[L - 1]) {
						OutNum++;
						if (Output_Fun == softmax)
							for (int k = 0; k < S[L - 1]; k++) {
								Sum += -1.0*Out[k] * log(OutputLayer->OutPut[k]);
							}
						else if (Output_Fun == sigmoid) {
							double E = 0;
							for (int k = 0; k < S[L - 1]; k++) {
								E += pow(OutputLayer->OutPut[k] - Out[k], 2);
							}E /= 2;
							Sum += E;
						}
					}
				}
				for (int j = param[g*Num + t].Count - 1; j > -1; j--) {
					Backward_Gradient(j, Backward_H, Backward_C, param[g*Num + t].param[j]._DataOut->Count == S[L - 1]);
				}
			}
			ForgetGate->RefreshData(Num, Speed, ForgetFactor, Beta1, Beta2);
			InputGate->RefreshData(Num, Speed, ForgetFactor, Beta1, Beta2);
			CellUpdate->RefreshData(Num, Speed, ForgetFactor, Beta1, Beta2);
			OutputGate->RefreshData(Num, Speed, ForgetFactor, Beta1, Beta2);
			OutputLayer->RefreshData(OutNum, Speed, ForgetFactor, Beta1, Beta2);

			std::cout << "Loss:" << Sum << ' ' << LastSum - Sum << ' ' << TrainCount << ' ' << N << '\n';
			LastSum = Sum;
		}
		//train examples
		Test(param, trainCnt);
		//test examples
		Test(&param[trainCnt], paramNum-trainCnt);


		if (_kbhit()) {
			if (_getch() == (int)'c')
				break;
		}
	}


	delete[] ExamIn;
	delete[] ExamOut;
	delete[] Backward_H;
	delete[] Backward_C;
}
void LSTM::Response(LSTM_Param&param)
{
	_CellState->Clear();
	CellOutPut->Clear();
	for (int j = 0; j < param.Count; j++) {
		memset(In, 0, sizeof(double)*S[0]);
		param.param[j]._DataIn->DeCode(In + S[1]);
		ComputerProc(j, Out);
		param.param[j]._DataOut->EnCode(OutputLayer->OutPut, S[L - 1]);
	}
}
double LSTM::Test(LSTM_Param*param, int testNum) {
	int cor = 0, tot = 0;
	for (int i = 0; i < testNum; i++) {
		LSTM_Param P = param[i];
		Response(P);
		for (int j = 0; j < P.Count; j++) {
			if (param[i].param[j]._DataOut->Count == S[L - 1]) {
				if ((Output_Fun == sigmoid && param[i].param[j]._DataOut->Compare(P.param[j]._DataOut, 1e-2)) || (Output_Fun == softmax && (*param[i].param[j]._DataOut)[P.param[j]._DataOut->max_value_id()]))
					cor++;
				tot++;
			}
		}
	}
	printf("Test:%0.2lf\n", 100.0*cor / tot);
	return 100.0*cor / tot;
}

