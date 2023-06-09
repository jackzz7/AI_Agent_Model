#define BP_EXPORTS
#include"G_Net.h"

#include<time.h>
#include<conio.h>

using namespace _Net_;
using std::cout;

void _Net_::TestP(double**ans, double*out, int No, int*S) {
	double eps = 1e-8;
	for (int i = 0; i < S[No]; i++)
		for (int j = 0; j < S[No - 1]; j++)
			if (abs(ans[i][j] - out[i * S[No - 1] + j]) > eps)cout << i << ' ' << j << " wrong\n";
}
void _Net_::TestP(double*ans, double*out, int No, int*S) {
	double eps = 1e-8;
	for (int i = 0; i < S[No]; i++)
		if (abs(ans[i] - out[i * 1 + 0]) > eps)cout << i << " wrong\n";
}


int G_Net::Cuda_Train(Param*param, int paramNum, int trainNum, const char*Net_param, const char*Path) {

	//double**ExamIn = new double*[S[0]], **ExamOut = new double*[S[L - 1]];
	//for (int i = 0; i < S[0]; i++) {
	//	ExamIn[i] = new double[paramNum];
	//}
	//for (int i = 0; i < S[L - 1]; i++) {
	//	ExamOut[i] = new double[paramNum];
	//}
	//for (int i = 0; i < S[0]; i++) {
	//	for (int j = 0; j < paramNum; j++)
	//		ExamIn[i][j] = (*param[j]._DataIn)[i];
	//}
	//for (int i = 0; i < S[L - 1]; i++) {
	//	for (int j = 0; j < paramNum; j++)
	//		ExamOut[i][j] = (*param[j]._DataOut)[i];
	//}
	//
	//Mat C_In(S[0], Num);
	//Mat C_Out(S[L - 1], Num);

	//int _trainNum = trainNum / Num * Num;
	//int _testNum = (paramNum - _trainNum) / Num * Num;
	//Mat Train_In(S[0], _trainNum, ExamIn);
	//Mat Train_Out(S[L - 1], _trainNum, ExamOut);
	//Mat Test_In(S[0], _testNum, _trainNum, ExamIn);
	//Mat Test_Out(S[L - 1],_testNum, _trainNum, ExamOut);
	//
	//
	//Init_Param(Net_param);

	//Adam_param Speed(Net_Param["Speed"][0]);

	//double E = 0, Sum = 0, LastSum = -1, N = -log(0.6) * Num;
	//double MaxCor = 0.0, MaxP = 40, PIdx = 0;
	//double Last_Test = 0, Best_Test = 0;
	//ui TrainCount = 0;
	//clock_t Start = clock();

	//while (true)
	//{
	//	//epoch
	//	for (int t = 0; t < trainNum / Num; t++) {
	//		//train example random order
	//		//RandomGenerator.GenerateRandom();
	//		//C_In.RandMatrix(Train_In, RandomGenerator);
	//		//C_Out.RandMatrix(Train_Out, RandomGenerator);
	//		////random DropOut for Hidden Units and Input Units
	//		//IPU->C_DropOut.f(DropOut_Bernoulli, Net_Param["InPut_DropOut"][0]);
	//		//for (int i = 0; i < L - 2; i++) {
	//		//	TU[i].C_DropOut.f(DropOut_Bernoulli, Net_Param["Hidden_DropOut"][0]);
	//		//}

	//		//Cuda_ComputerProc(C_In,true);
	//		//Sum = Cal_Loss(OPU->C_OutPut, C_Out);

	//		//OPU->Cuda_AdjustWeighb(Output_Fun, C_Out, TU[L - 3].C_OutPut, &TU[L - 3].C_DropOut, NULL);
	//		//for (int i = L - 3; i > -1; i--) {
	//		//	TU[i].Cuda_AdjustWeighb(Layer_Func[i], C_Out, i == 0 ? C_In:TU[i - 1].C_OutPut, i == 0 ? &IPU->C_DropOut: &TU[i - 1].C_DropOut , i == L - 3 ? OPU : &TU[i + 1]);
	//		//}

	//		//Net_Param["Beta1_Pow"] *= Net_Param["Beta1"];
	//		//Net_Param["Beta2_Pow"] *= Net_Param["Beta2"];
	//		//for (int i = 0; i < L - 2; i++) {
	//		//	TU[i].Cuda_RefreshData(Net_Param);
	//		//}
	//		//OPU->Cuda_RefreshData(Net_Param);
	//		

	//		printf("Loss:%f Count:%d epoch:%d last:%.02f best:%.02f Speed:%f Time:%d min(s)\n", Sum, ++TrainCount, TrainCount / (trainNum / Num), Last_Test, Best_Test, Speed.Data, (clock() - Start) / 1000 / 60);
	//		//std::cout << "Loss:" << Sum << ' ' << LastSum - Sum << ' ' << TrainCount << ' ' << N << '\n';
	//		//Speed = OriSpeed / (1 + TrainCount / 4000.0);
	//		//C_Speed.Reset(1, 1, &Speed, true);
	//	}
	//	
	//	//testing
	//	if ((TrainCount / (trainNum / Num)) % 1 == 0) {
	//		//WriteTo_Net();
	//		//Train example
	//		Test(Train_In, Train_Out, trainNum / Num);
	//		//Test example
	//		double Last = Last_Test;
	//		Last_Test = Test(Test_In, Test_Out, (paramNum - trainNum / Num * Num) / Num);
	//		Best_Test = max(Best_Test, Last_Test);

	//		Speed.Liner_Update(Last, Last_Test, Net_Param["Speed"]);
	//		if (Last_Test > MaxCor) {
	//			PIdx = 0;
	//			MaxCor = Last_Test;
	//			DataWrite(Path);
	//		}
	//		//else if (++PIdx > MaxP)break;
	//	}

	//	if (_kbhit()) {
	//		int get = _getch();
	//		if (get == (int)'c')
	//			break;
	//		else if (get == (int)'w') {
	//			DataWrite(Path);
	//		}
	//	}
	//}
	//for (int i = 0; i < S[0]; i++) {
	//	delete[] ExamIn[i];
	//}
	//for (int i = 0; i < S[L - 1]; i++) {
	//	delete[] ExamOut[i];
	//}
	//delete[] ExamIn;
	//delete[] ExamOut;

	return true;
}