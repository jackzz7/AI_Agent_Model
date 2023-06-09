#define NET_EXPORTS

#include<iostream>
#include<time.h>
#include<conio.h>

#include"Param.h"

using namespace Net_CPU;
using std::ios;

_Net::InPutUnit::InPutUnit(int*S):InPutUnit(S,0){}
_Net::InPutUnit::InPutUnit(int*S, int No):No(No),S(S)
{
	InPut = new double[S[No]]{ 0 };
	DropOut = new double[S[No]]{ 0 };
	for (int i = 0; i < S[No]; i++)
		DropOut[i] = 1.0;
}
void _Net::InPutUnit::Clear() {
	memset(InPut, 0, sizeof(double)*S[No]);
}
void _Net::InPutUnit::Disponse() {
	delete[] InPut;
	delete[] DropOut;
}
_Net::TrainUnit::TrainUnit(int Last_Node_Num, int Node_Num) {
	new(this)TrainUnit(new int[2]{ Last_Node_Num,Node_Num }, 1);
}
_Net::TrainUnit::TrainUnit(int*S, int No):S(S),No(No)
{
	//初始化偏置和权值
	b = new double[S[No]];
	Biasb = new double[S[No]];
	Lastb = new double[S[No]];
	Lastb2 = new double[S[No]];
	b_Beta1 = new double[S[No]];
	b_Beta2 = new double[S[No]];
	Weigh = new double*[S[No]];
	BiasWeigh = new double*[S[No]];
	LastWeigh = new double*[S[No]];
	LastWeigh2 = new double*[S[No]];
	Weigh_Beta1 = new double*[S[No]];
	Weigh_Beta2 = new double*[S[No]];
	DropOut = new double[S[No]];
	for (int i = 0; i < S[No]; i++) {
		b[i] = (double)(rand() % 500) / (1000 * S[0]);
		Lastb2[i] = Lastb[i] = Biasb[i] = 0.0;
		b_Beta1[i] = b_Beta2[i] = 1;
		Weigh[i] = new double[S[No - 1]];
		BiasWeigh[i] = new double[S[No - 1]];
		LastWeigh[i] = new double[S[No - 1]];
		LastWeigh2[i] = new double[S[No - 1]];
		Weigh_Beta1[i] = new double[S[No - 1]];
		Weigh_Beta2[i] = new double[S[No - 1]];
		for (int j = 0; j < S[No - 1]; j++) {
			Weigh[i][j] = (double)(rand() % 500) / (1000 * S[0]);
			BiasWeigh[i][j] = 0.0;
			LastWeigh[i][j] = 0.0;
			LastWeigh2[i][j] = 0.0;
			Weigh_Beta1[i][j] = Weigh_Beta2[i][j] = 1;
		}
		DropOut[i] = 1.0;
	}
	OutPut = new double[S[No]];
	Net = new double[S[No]];
	Param = new double[S[No]];
}
void _Net::TrainUnit::Disponse()
{
	for (int i = 0; i < S[No]; i++) {
		delete[] Weigh[i];
		delete[] BiasWeigh[i];
		delete[] LastWeigh[i];
		delete[] LastWeigh2[i];
		delete[] Weigh_Beta1[i];
		delete[] Weigh_Beta2[i];
	}
	delete[] b;
	delete[] Biasb;
	delete[] Lastb;
	delete[] Lastb2;
	delete[] b_Beta1;
	delete[] b_Beta2;
	delete[] Weigh;
	delete[] BiasWeigh;
	delete[] LastWeigh;
	delete[] LastWeigh2;
	delete[] Weigh_Beta1;
	delete[] Weigh_Beta2;
	delete[] OutPut;
	delete[] Net;
	delete[] Param;
	delete[] DropOut;
}
_Net::TrainUnit::TrainUnit() {}
void _Net::TrainUnit::ComputeOutPut(_Activation_Func Fun,InPutUnit*IPU, TrainUnit *LastTU)
{
	double Sum = 0;
	for (int i = 0; i < S[No]; i++) {
		Sum = 0;
		for (int j = 0; j < S[No - 1]; j++)
		{
			if (LastTU != NULL)
				Sum += Weigh[i][j] * LastTU->OutPut[j];
			else Sum += Weigh[i][j] * IPU->InPut[j];
		}
		Sum += b[i];
		Net[i] = Sum;
		if (Fun==softmax)OutPut[i] = Sum;
		else if(Fun==relu)OutPut[i] = ReLU(Sum);
		else if(Fun==sigmoid)OutPut[i] = Sigmoid(Sum);
		else if (Fun == tanh)OutPut[i] = Tanh(Sum);
	}if (Fun == softmax)
		SoftMax(OutPut, S[No]);
	//DropOut
	for (int i = 0; i < S[No]; i++) {
		OutPut[i] *= DropOut[i];
	}
}

void _Net::TrainUnit::AdjustWeighb(int L, double*Out, InPutUnit*IPU, TrainUnit *LastTU, TrainUnit*NextTU, _Net::_Activation_Func Output_Fun)
{
	for (int i = 0; i < S[No]; i++) {
		double B, C = 0;
		if (No == L - 1) {
			//Sigmoid
			if (Output_Fun == sigmoid) {
				B = OutPut[i] * (1.0 - OutPut[i]);
				C = (OutPut[i] - Out[i]);
			}
			//SoftMax+xent(Cross-Entropy Loss)
			else if (Output_Fun == softmax) {
				B = 1;
				C = OutPut[i] - Out[i];
			}
			/*for (int j = 0; j < S[No]; j++) {
				C += Out[j]*(OutPut[i]-((i == j) ? 1.0 : 0.0));
			}*/
			//for (int j = 0; j < S[No]; j++) {
				//C += (OutPut[j] - Out[j])* OutPut[i] * ((i == j ? 1 : 0) - OutPut[j]);
			//}
		}
		//Hidden
		else {
			//ReLU
			if (Net[i] <= 0)B = 0.01;
			else B = 1;
			//DropOut
			B *= DropOut[i];

			for (int j = 0; j < S[No + 1]; j++) {
				C += NextTU->Weigh[j][i] * NextTU->Param[j];
			}
		}
		C *= B;
		Biasb[i] -= C;
		Param[i] = C;

		for (int j = 0; j < S[No - 1]; j++)
		{
			if (LastTU != NULL)
				BiasWeigh[i][j] -= C*LastTU->OutPut[j];
			else BiasWeigh[i][j] -= C*IPU->InPut[j];
		}
	}


	////输出层
	//if (No == L - 1) {
	//	for (int i = 0; i < S[No]; i++) {
	//		double A = exp(-_Net[i]);
	//		double B = A / ((1 + A)*(1 + A));// pow(1 + A, 2);
	//		double C = (OutPut[i] - Out[i])*B;
	//		Biasb[i] -= Speed * C;
	//		Param[i] = C;
	//		for (int j = 0; j < S[No - 1]; j++)
	//		{
	//			BiasWeigh[i][j] -= Speed * (C*LastTU->OutPut[j]);
	//		}
	//	}
	//}
	////隐藏层
	//else {
	//	for (int k = 0; k < S[No]; k++) {
	//		double A = exp(-_Net[k]);
	//		double B = A / ((1 + A)*(1 + A));
	//		double C = 0;
	//		for (int i = 0; i < S[No + 1]; i++) {
	//			C += NextTU->Weigh[i][k] * NextTU->Param[i] * B;
	//		}
	//		Param[k] = C;
	//		Biasb[k] -= Speed * C;
	//		for (int j = 0; j < S[No - 1]; j++)
	//		{
	//			if (LastTU != NULL)
	//				BiasWeigh[k][j] -= Speed * C*LastTU->OutPut[j];
	//			else BiasWeigh[k][j] -= Speed * C*IPU->InPut[j];
	//		}
	//	}
	//}
}
void _Net::TrainUnit::RefreshData(int Num, double Speed, double ForgetFactor, double Beta1, double Beta2, TrainUnit*LastUnit)
{
	//Nesterov Momentum
	//for (int i = 0; i < S[No]; i++) {
	//	//DropOut
	//	if (DropOut[i] == 0.0)continue;
	//	Biasb[i] *= Speed / Num;
	//	Lastb[i] = Biasb[i] + ForgetFactor * Lastb[i];
	//	b[i] += ForgetFactor * Lastb[i] + Biasb[i];
	//	for (int j = 0; j < S[No - 1]; j++) {
	//		//DropOut
	//		if (LastUnit&&LastUnit->DropOut[j] == 0.0)continue;
	//		BiasWeigh[i][j] *= Speed / Num;
	//		LastWeigh[i][j] = BiasWeigh[i][j] + ForgetFactor * LastWeigh[i][j];
	//		Weigh[i][j] += ForgetFactor * LastWeigh[i][j] + BiasWeigh[i][j];
	//	}
	//}
	//Adam
	for (int i = 0; i < S[No]; i++) {
		//DropOut
		if (DropOut[i] == 0.0)continue;
		Biasb[i] /= Num;
		Lastb[i] = Beta1 * Lastb[i] + (1.0 - Beta1)*Biasb[i];
		Lastb2[i] = Beta2 * Lastb2[i] + (1.0 - Beta2)*Biasb[i] * Biasb[i];
		//modify
		b_Beta1[i] *= Beta1;
		b_Beta2[i] *= Beta2;
		double M_Lastb = Lastb[i] / (1.0 - b_Beta1[i]);
		double M_Lastb2 = Lastb2[i] / (1.0 - b_Beta2[i]);
		b[i] += Speed * M_Lastb / (sqrt(M_Lastb2+1e-16) + eps);
		for (int j = 0; j < S[No - 1]; j++) {
			//DropOut
			if (LastUnit&&LastUnit->DropOut[j] == 0.0)continue;
			BiasWeigh[i][j] /= Num;
			LastWeigh[i][j] = Beta1 * LastWeigh[i][j] + (1.0 - Beta1)*BiasWeigh[i][j];
			LastWeigh2[i][j] = Beta2 * LastWeigh2[i][j] + (1.0 - Beta2)*BiasWeigh[i][j] * BiasWeigh[i][j];
			//modify
			Weigh_Beta1[i][j] *= Beta1;
			Weigh_Beta2[i][j] *= Beta2;
			double M_LastWeigh = LastWeigh[i][j] / (1.0 - Weigh_Beta1[i][j]);
			double M_LastWeigh2 = LastWeigh2[i][j] / (1.0 - Weigh_Beta2[i][j]);
			Weigh[i][j] += Speed * M_LastWeigh / (sqrt(M_LastWeigh2+1e-16) + eps);
		}
	}

	//ZeroMemory
	memset(Biasb, 0, sizeof(double)*S[No]);
	for (int i = 0; i < S[No]; i++) {
		memset(BiasWeigh[i], 0, sizeof(double)*S[No - 1]);
	}
}
void _Net::TrainUnit::DataWrite(fstream*file)
{
	for (int i = 0; i < S[No]; i++)
		file->write((char*)Weigh[i], sizeof(double)*S[No - 1]);
	file->write((char*)b, sizeof(double)*S[No]);
}
void _Net::TrainUnit::DataRead(fstream*file)
{
	for (int i = 0; i < S[No]; i++)
		file->read((char*)Weigh[i], sizeof(double)*S[No - 1]);
	file->read((char*)b, sizeof(double)*S[No]);
}
bool _Net::IsVaild() {
	return L != -1;
}
_Net::_Net(const char*filePath, int BatchNum, _Activation_Func Output_Fun, double Sigmoid_eps, int Negative_OutPut_No) {
	fstream file;
	file.open(filePath, ios::app | ios::out); file.close();
	file.open(filePath, ios::in | ios::binary);
	file.read((char*)&L, sizeof(int));
	if (file.gcount() != sizeof(int)) {
		L = -1;return;
	}
	S = new int[L];
	file.read((char*)S, sizeof(int)*L);
	file.read((char*)&Output_Fun, sizeof(_Activation_Func));
	Layer_Func = new _Activation_Func[L - 2];
	file.read((char*)Layer_Func, sizeof(_Activation_Func)*(L - 2));
	new(this)_Net(L, S, BatchNum,Output_Fun,Sigmoid_eps,Negative_OutPut_No);
	for (int i = 0; i < L - 2; i++) {
		TU[i].DataRead(&file);
	}
	OPU->DataRead(&file);
	file.close();
}
void _Net::ComputerProc()
{
	for (int i = 0; i < L - 2; i++) {
		TU[i].ComputeOutPut(relu,IPU, i == 0 ? NULL : &TU[i - 1]);
	}
	OPU->ComputeOutPut(Output_Fun,IPU, &TU[L - 3]);
}

// 要拟合的函数 
double fx(double x1, double x2, double x3) {
	return x1 * x1*x1 + x2 * x2 + x3;
}
//f(a,b,c)=a^3+b^2+c+1;
#define Fun a*a*a+b*b+c+1
//double Net_CPU::f(double*In) {
//	double a = In[0], b = In[1], c = In[2];
//	return Fun;
//}
//double Net_CPU::f(double a, double b, double c) {
//	return Fun;
//}
void _Net::Net_Write(fstream&file) {
	file.write((char*)&L, sizeof(int));
	file.write((char*)S, sizeof(int)*L);
	file.write((char*)&Output_Fun, sizeof(_Activation_Func));
	file.write((char*)Layer_Func, sizeof(_Activation_Func)*(L - 2));
	for (int i = 0; i < L - 2; i++) {
		TU[i].DataWrite(&file);
	}
	OPU->DataWrite(&file);
	file.flush();
	file.close();
}
void _Net::DataWrite(const wchar_t*filePath) {
	fstream file;file.open(filePath, ios::trunc | ios::out | ios::binary);
	Net_Write(file);
}
void _Net::DataWrite(const char*filePath) {
	fstream file; file.open(filePath, ios::trunc | ios::out | ios::binary);
	Net_Write(file);
}
int _Net::Train(Param*param, int paramNum, int trainNum, double Speed, double ForgetFactor, double Beta1, double Beta2,double DropOut)
{
	double OriSpeed = Speed;
	int TrainCnt = 0;
	double*ExamIn, *ExamOut;
	
	ExamIn = new double[S[0]];
	ExamOut = new double[S[L - 1]];


	In = ExamIn;
	Out = ExamOut;

	int*Ord = new int[trainNum];
	int*W = new int[trainNum];
	
	double E = 0, Sum = 0, LastSum = -1, N = 0.0001*S[L - 1] * Num;
	clock_t Start = clock();
	while (true)
	{
		//train examples random order
		for (int i = 0; i < trainNum; i++)
			Ord[i] = i, W[i] = rand();
		std::sort(Ord, Ord + trainNum, [W](int a, int b) {return W[a] < W[b]; });

		for (int t = 0; t < trainNum / Num; t++) {
			Sum = 0;
			TrainCnt++;
			//random DropOut for Hidden Unit
			for (int i = 0; i < L - 2; i++) {
				for (int j = 0; j < S[i + 1]; j++)
					TU[i].DropOut[j] = Bernoulli(DropOut) ? (1 / (1.0 - DropOut)) : 0.0;
			}
			for (int j = 0; j < Num; j++) {
				memset(In, 0, sizeof(double)*S[0]);
				memset(Out, 0, sizeof(double)*S[L - 1]);

				param[Ord[t*Num + j]]._DataIn->DeCode(In);
				param[Ord[t*Num + j]]._DataOut->DeCode(Out);

				memcpy(IPU->InPut, In, sizeof(double)*S[0]);
				//前向计算
				ComputerProc();
				//累计批量梯度
				OPU->AdjustWeighb(L, Out, IPU, &TU[L - 3], NULL, Output_Fun);
				for (int i = L - 3; i > -1; i--) {
					TU[i].AdjustWeighb(L, Out, IPU, i == 0 ? NULL : &TU[i - 1], i == L - 3 ? OPU : &TU[i + 1]);
				}
				//损失函数
				{
					E = 0;
					if (Output_Fun == softmax) {
						for (int i = 0; i < S[L - 1]; i++) {
							E += -1.0*Out[i] * log(OPU->OutPut[i]+1e-16);
						}
					}
					else if (Output_Fun == sigmoid) {
						for (int i = 0; i < S[L - 1]; i++)
							E += pow(OPU->OutPut[i] - Out[i], 2);
						E /= 2;
					}
					Sum += E;
				}
			}
			for (int i = 0; i < L - 2; i++)
				TU[i].RefreshData(Num, Speed, ForgetFactor, Beta1, Beta2, i == 0 ? NULL:&TU[i - 1]);
			OPU->RefreshData(Num, Speed, ForgetFactor, Beta1, Beta2, &TU[L-3]);
			
			
			//Speed = OriSpeed / (1 + TrainCnt / 4000.0);

			std::cout << "Loss:" << Sum << ' ' << LastSum - Sum << ' ' << TrainCnt << ' ' << N << '\n';
			LastSum = Sum;
		}
		//reset DropOut
		for (int i = 0; i < L - 2; i++) {
			for (int j = 0; j < S[i + 1]; j++)
				TU[i].DropOut[j] = 1;
		}
		//train example
		Test(param, trainNum);
		//test example
		Test(&param[trainNum], paramNum-trainNum);


		if (_kbhit()) {
			if (_getch() == (int)'c')
				break;
		}
	}


	delete[] ExamIn;
	delete[] ExamOut;
	delete[] Ord;
	delete[] W;

	return true;
}
double _Net::Test(Param*param,int testNum) {
	int cor = 0;
	for (int i = 0; i < testNum; i++) {
		Param P = param[i];
		Response(P);
		if ((Output_Fun == sigmoid&&param[i]._DataOut->Compare(P._DataOut, 1e-4))||(Output_Fun==softmax&& (*param[i]._DataOut)[P._DataOut->max_value_id()]))
			cor++;
	}
	printf("Test:%0.2lf\n", 100.0*cor / testNum);
	return 100.0*cor / testNum;
}

_Net::_Net(int L,int*_S, int BatchNum, _Activation_Func Output_Fun, double Sigmoid_eps,int Negative_OutPut_No, _Activation_Func*Hidden_Layer_Func):L(L),Num(BatchNum), Output_Fun(Output_Fun),Sigmoid_eps(Sigmoid_eps),Negative_OutPut_No(Negative_OutPut_No)
{
	this->S = new int[L];
	for (int i = 0; i < L; i++) {
		if (_S == NULL)this->S[i] = 1;
		else this->S[i] = _S[i];
	}
	TU = new TrainUnit[L - 2];
	for (int i = 0; i < L - 2; i++) {
		TU[i] = TrainUnit(S, i + 1);
	}
	OPU = new TrainUnit(S, L - 1);
	IPU = new InPutUnit(S);

	Layer_Func = new _Activation_Func[L - 2];
	for (int i = 0; i < L - 2; i++)
		if (Hidden_Layer_Func)
			Layer_Func[i] = Hidden_Layer_Func[i];
		else Layer_Func[i] = relu;
}
void _Net::Disponse() {
	if (L == -1)return;
	L = -1;
	if (TU)
		for (int i = 0; i < L - 2; i++) {
			TU[i].Disponse();
		}
	if (OPU)OPU->Disponse();
	if (IPU)IPU->Disponse();
	delete[] TU, TU = NULL;
	delete OPU, OPU = NULL;
	delete IPU, IPU = NULL;
	delete[] S, S = NULL;
}
bool _Net::DataRead(const char*filePath)
{
	bool Flag = false;
	fstream file;
	file.open(filePath, ios::out | ios::app); file.close();
	file.open(filePath, ios::in | ios::binary);
	file.read((char*)this, sizeof(_Net));
	Flag = (file.gcount() == sizeof(_Net));
	file.close();
	return Flag;
}
void _Net::Response(Param&param)
{
	for (int j = 0; j < S[0]; j++)
		IPU->InPut[j] = 0;
	param._DataIn->DeCode(IPU->InPut);
	ComputerProc();
	param._DataOut->EnCode(OPU->OutPut, S[L - 1]);
}


double _Net::Sigmoid(double In)
{
	return 1.0 / (1.0 + exp(-In));
}
//f'=1-f^2
double _Net::Sigmoid_(double In)
{
	double A = 1 - exp(-In), B = 1 + exp(-In);
	return A / B;
}
double _Net::Tanh(double In) {
	double A = exp(In), B = exp(-In);
	return (A - B) / (A + B);
}
double _Net::ReLU(double In)
{
	return std::max(0.0, In) + 0.01*std::min(0.0, In);
}
void _Net::SoftMax(double*In, int Num)
{
	double Sum = 0, mx = In[0];
	for (int i = 0; i < Num; i++)
		if (In[i] > mx)mx = In[i];
	for (int i = 0; i < Num; i++) {
		Sum += exp(In[i] - mx);
	}
	for (int i = 0; i < Num; i++) {
		In[i] = exp(In[i] - mx) / Sum;
	}
}