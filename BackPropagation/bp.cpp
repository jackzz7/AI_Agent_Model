#define BP_EXPORTS

#include<iostream>
#include<time.h>
#include<conio.h>

#include"G_Net.h"
#include"ConvNet.h"

//using namespace Net_CPU;
using namespace _Net_;
using std::ios;


void G_Net::Func_Init(int MaxRandomStates_rows) {
	int max_rows = MaxRandomStates_rows;
	for (int i = 0; i < L; i++)if (S[i] > max_rows)max_rows = S[i];
	Matrix_Set_Function(NULL, NULL,max_rows);
}

G_Net::InPutUnit::InPutUnit(int Node, int Batch):Net(Node) {
	C_InPut.Reset(Node, Batch,false);
}

G_Net::TrainUnit::TrainUnit(int Node,int Pre_Node, int Batch, int b_Node):Net(Node)
{
	//Cuda Init
	if (b_Node == -1)b_Node = Node;
	C_b.Reset(b_Node, 1); C_b.RandData();
	C_Biasb.Reset(b_Node, 1, 0.0f);
	C_Momentumb.Reset(b_Node, 1, 0.0f);
	C_Momentumb2.Reset(b_Node, 1, 0.0f);

	C_Weigh.Reset(Node, Pre_Node); C_Weigh.RandData();
	C_BiasWeigh.Reset(Node, Pre_Node, 0.0f);
	C_MomentumWeigh.Reset(Node, Pre_Node, 0.0f);
	C_MomentumWeigh2.Reset(Node, Pre_Node, 0.0f);

	C_OutPut.Reset(Node, Batch,false);
}
G_Net::TrainUnit::TrainUnit() {}
//void G_Net::TrainUnit::Cuda_ComputeOutPut(_Activation_Func Activate_Fun,Mat&Last_Output,const Mat*Last_DropOut_Scale)
//{
//	//DropOut
//	/*Mat Input = Last_Output;
//	if (Last_DropOut_Scale)
//		Input *= *Last_DropOut_Scale;*/
//	/*C_Weigh.ReadFromDevice();
//	C_b.ReadFromDevice();
//	Last_Output.ReadFromDevice();*/
//	C_OutPut = C_Weigh * Last_Output + C_b;
//	//C_OutPut.ReadFromDevice();
//	//ReLU
//	if (Activate_Fun == relu)C_OutPut.f(Device_Func::ReLU);
//	//Sigmoid
//	else if (Activate_Fun == sigmoid)C_OutPut.f(Device_Func::Sigmoid);
//	//tanh
//	else if(Activate_Fun == tanh)C_OutPut.f(Device_Func::Tanh);
//	//SoftMax
//	else if (Activate_Fun == softmax)C_OutPut.SoftMax();
//	else if (Activate_Fun == null) {}
//	else assert(false);
//	//C_OutPut.ReadFromDevice();
//}
//void G_Net::TrainUnit::Cuda_AdjustWeighb(_Activation_Func Activate_Fun, Mat& C_Out, Mat&Last_Output, Mat*Last_DropOut_Scale, TrainUnit*NextTU) {
//	//SoftMax+xent(Cross-Entropy Loss)
//	if (Activate_Fun == softmax) {
//		C_OutPut -= C_Out;
//	}
//	else if (Activate_Fun == relu) {
//		//C_Weigh转置后与C_Param矩阵乘法
//		//运算顺序一致？？
//		C_OutPut.f(D_ReLU) *= (!NextTU->C_Weigh)*NextTU->C_OutPut;
//	}
//	else if (Activate_Fun == sigmoid) {
//		if (NextTU == NULL)
//			C_OutPut.f(D_Sigmoid) *= C_OutPut - C_Out;
//		else C_OutPut.f(D_Sigmoid) *= (!NextTU->C_Weigh)*NextTU->C_OutPut;
//	}
//	else if (Activate_Fun == null) {
//		C_OutPut = (!NextTU->C_Weigh)*NextTU->C_OutPut;
//	}
//	else if (Activate_Fun == tanh) {
//		C_OutPut.f(D_Tanh) *= (!NextTU->C_Weigh)*NextTU->C_OutPut;
//	}
//	//DropOut
//	//C_OutPut *= C_DropOut;
//	//同型矩阵对应位置元素相减
//	//不同型默认矩阵拓展
//	C_Biasb -= C_OutPut;
//	//DropOut
//	Mat Input = Last_Output;
//	if (Last_DropOut_Scale)
//		Input *= *Last_DropOut_Scale;
//	C_BiasWeigh -= C_OutPut * (!Input);
//}
//void G_Net::TrainUnit::Gradient_Update(NetParam&Net_Param) {
//	C_b += Net_Param["b_Gradient"];
//	C_Weigh += Net_Param["weigh_Gradient"];
//}

//Gradient Optimizers
typedef void(*Optimizer)(Mat&Weigh_b, Mat&Bias, Mat&C_Momentum, Mat&C_Momentum2, NetParam&Net_Param, Mat*);
void optimizers_Adam(Mat&Weigh_b, Mat&Bias, Mat&C_Momentum, Mat&C_Momentum2, NetParam&Net_Param, Mat*Send_Gradient) {
	C_Momentum = Net_Param["Beta1"] * C_Momentum + (1 - Net_Param["Beta1"]) * Bias;
	C_Momentum2 = Net_Param["Beta2"] * C_Momentum2 + (1 - Net_Param["Beta2"]) * Bias._f(Pow2);
	Mat C_M_Momentum = C_Momentum / (1 - Net_Param["Beta1_Pow"]);
	Mat C_M_Momentum2 = C_Momentum2 / (1 - Net_Param["Beta2_Pow"]);
	Weigh_b += Net_Param["Speed"] * (C_M_Momentum / C_M_Momentum2.f(Sqrt));
	if (Send_Gradient)*Send_Gradient += Net_Param["Speed"] * (C_M_Momentum / C_M_Momentum2);
	Bias._ZeroMemory();
}
void optimizers_SGD_no_Momentum(Mat&Weigh_b, Mat&Bias, Mat&C_Momentum, Mat&C_Momentum2, NetParam&Net_Param, Mat*Send_Gradient) {
	Weigh_b += Net_Param["Speed"] * Bias;
	if (Send_Gradient)*Send_Gradient += Net_Param["Speed"] * Bias;
	Bias._ZeroMemory();
}
void optimizers_SGD_with_Momentum(Mat&Weigh_b, Mat&Bias, Mat&C_Momentum, Mat&C_Momentum2, NetParam&Net_Param,Mat*Send_Gradient) {
	C_Momentum = Net_Param["Beta1"] * C_Momentum + Net_Param["Speed"]*Bias;
	Weigh_b += C_Momentum;
	if (Send_Gradient)*Send_Gradient += C_Momentum;
	Bias._ZeroMemory();
}
Optimizer Optimizers[10] = { optimizers_Adam,optimizers_SGD_no_Momentum,optimizers_SGD_with_Momentum };
void G_Net::TrainUnit::Cuda_RefreshData(NetParam&Net_Param, double Gradient_Scale_Factor, G_Net::TrainUnit*Send_Gradient) {
	C_Biasb = C_Biasb * (Gradient_Scale_Factor / Net_Param["Batch"]);
	C_BiasWeigh = C_BiasWeigh * (Gradient_Scale_Factor / Net_Param["Batch"]);
	//L2 Regularization penalty term
	C_Biasb -= 2.0*Net_Param["L2_Factor"] * C_b;
	C_BiasWeigh -= 2.0*Net_Param["L2_Factor"] * C_Weigh;

	Optimizers[(int)Net_Param["Optimizer"]](C_b, C_Biasb, C_Momentumb, C_Momentumb2, Net_Param, Send_Gradient ? &Send_Gradient->C_Biasb : NULL);
	Optimizers[(int)Net_Param["Optimizer"]](C_Weigh, C_BiasWeigh, C_MomentumWeigh, C_MomentumWeigh2, Net_Param, Send_Gradient ? &Send_Gradient->C_BiasWeigh : NULL);
}
void G_Net::TrainUnit::Data_WR(fstream&file, bool Write) {
	Matrix_WR(C_Weigh, file, Write);
	Matrix_WR(C_b, file, Write);
}
bool G_Net::TrainUnit::ConvCheck(Mat&Src, Mat*right, bool raw) {
	Conv*cv = dynamic_cast<Conv*>(Pre_Net_Head.front());
	if (cv) {
		Mat _right;
		if (right == NULL)_right = move(Src), right = &_right;
		if (raw) {
			Src.Reset(cv->Filters, cv->W*cv->H*cv->Batch);
			Src.Conv_Image_Restore(cv->W, *right, true);
		}
		else {
			Src.Reset(cv->Filters*cv->W*cv->H, cv->Batch);
			right->Conv_Image_Restore(cv->W, Src);
		}
	}
	return cv;
}
//void G_Net::TrainUnit::DataRead(fstream*file)
//{
//	int Node = C_Weigh.GetRow(), InPutNode = C_Weigh.GetCol();
//	double*W=C_Weigh.ReadFromDevice
//	for (int i = 0; i < Node; i++)
//		file->read((char*)Weigh[i], sizeof(double)*S[No - 1]);
//	file->read((char*)b, sizeof(double)*S[No]);
//}
G_Net::G_Net(const char*filePath,int BatchNum, _Activation_Func Output_Fun, double Sigmoid_eps, int Negative_OutPut_No):_Net(filePath,BatchNum,Output_Fun,Sigmoid_eps,Negative_OutPut_No) {
	/*fstream file;
	file.open(filePath, ios::app | ios::out); file.close();
	file.open(filePath, ios::in | ios::binary);
	file.read((char*)&L, sizeof(int));
	if (file.gcount() != sizeof(int)) {
		L = -1;
		return;
	}
	S = new int[L];
	file.read((char*)S, sizeof(int)*L);*/
	Func_Init(BatchNum);
	/*this->L = L;
	this->S = new int[L];
	for (int i = 0; i < L; i++) {
		this->S[i] = S[i];
	}*/
	TU = new TrainUnit[L - 2];
	for (int i = 0; i < L - 2; i++) {
		TU[i] = TrainUnit(S[i + 1], S[i], BatchNum);
	}
	OPU = new TrainUnit(S[L-1],S[L-2],BatchNum);
	IPU = new InPutUnit(S[0],BatchNum);
	//*this = G_Net(L, S);
	/*for (int i = 0; i < L - 2; i++) {
		TU[i].DataRead(&file);
	}
	OPU->DataRead(&file);
	file.close();*/
}
//void Net::ComputerProc()
//{
//	for (int i = 0; i < L - 2; i++) {
//		TU[i].ComputeOutPut(IPU, i == 0 ? NULL : &TU[i - 1]);
//	}
//	OPU->ComputeOutPut(IPU, &TU[L - 3]);
//}
void G_Net::Cuda_ComputerProc(Mat&In,bool DropOut)
{
	/*for (int i = 0; i < L - 2; i++) {
		TU[i].Cuda_ComputeOutPut(Layer_Func[i], i == 0 ? In : TU[i - 1].C_OutPut, (i == 0 ? (DropOut?&IPU->C_DropOut:NULL) : (DropOut?&TU[i - 1].C_DropOut:NULL)));
	}
	OPU->Cuda_ComputeOutPut(Output_Fun, TU[L - 3].C_OutPut, DropOut ? &TU[L - 3].C_DropOut : NULL);*/
}
void G_Net::DataWrite(const wchar_t*filePath) {
	WriteToCPU();
	_Net::DataWrite(filePath);
}
void G_Net::DataWrite(const char*filePath) {
	WriteToCPU();
	_Net::DataWrite(filePath);
}
void G_Net::WriteToCPU(){
	for (int i = 0; i < L - 2; i++) {
		//TU[i].DataWrite(NULL);
	}
	//OPU->DataWrite(NULL);
}
double G_Net::Test(Mat&C_In, Mat&C_Out, int testNum) {
	int cor = 0, tot = testNum * Num;
	Mat In(S[0], Num), Out(S[L - 1], Num);
	for (int i = 0; i < testNum; i++) {
		In.Append(C_In, 0, i*Num);
		Out.Append(C_Out, 0, i*Num);
		Cuda_ComputerProc(In);
		if (Output_Fun == sigmoid) {
			cor += Num - (OPU->C_OutPut - Out).f(Compare, 1e-2).ScaleOneCol(true).f(Bool).Sum().ReadFromDevice()[0];
		}
		else if (Output_Fun == softmax) {
			cor+=(OPU->C_OutPut.SoftMax(true) *= Out).Sum().ReadFromDevice()[0];
		}
	}
	printf("Test:%0.2lf\n", 100.0*cor / tot);
	return 100.0*cor / tot;
}

G_Net::G_Net(int L, int*_S, int BatchNum, _Activation_Func Output_Fun, double Sigmoid_eps, int Negative_OutPut_No, _Net::_Activation_Func*Hidden_Layer_Func) :_Net(L, _S, BatchNum, Output_Fun, Sigmoid_eps, Negative_OutPut_No,Hidden_Layer_Func)
{
	//enough Random States
	/*Func_Init(BatchNum);
	Net_Param["Batch"] = Mat(1, 1, (double)BatchNum);

	TU = new TrainUnit[L - 2];
	for (int i = 0; i < L - 2; i++) {
		TU[i] = TrainUnit(S[i + 1], S[i], BatchNum);
	}
	OPU = new TrainUnit(S[L - 1], S[L - 2], BatchNum);
	IPU = new InPutUnit(S[0], BatchNum);

	RandomGenerator.Reset(1, Num);
	C_Num.Reset(1, 1, (double)BatchNum);
	double*tmp = new double[S[L - 1]];
	fill(tmp, tmp + S[L - 1], 1.0);
	if (Negative_OutPut_No != -1)
		tmp[Negative_OutPut_No] = 0;
	Positive_OutPut_mask.Reset(S[L - 1], 1, tmp);
	delete tmp;*/
}
void G_Net::Disponse() {
	if (L == -1)return;
	//for (int i = 0; i < L - 2; i++)
		//TU[i].Disponse();
	//OPU->Disponse();
	//IPU->Disponse();
	delete[] TU;
	delete OPU;
	delete IPU;
	_Net::Disponse();
}
//bool Net::DataRead(const char*filePath)
//{
//	bool Flag = false;
//	fstream file;
//	file.open(filePath, ios::out | ios::app); file.close();
//	file.open(filePath, ios::in | ios::binary);
//	file.read((char*)this, sizeof(Net));
//	Flag = (file.gcount() == sizeof(Net));
//	file.close();
//	return Flag;
//}
//void Net::Response(Param&param)
//{
//	for (int j = 0; j < S[0]; j++)
//		IPU->InPut[j] = 0;
//	param._DataIn->DeCode(IPU->InPut);
//	ComputerProc();
//	param._DataOut->EnCode(OPU->OutPut, S[L - 1]);
//}


//double Net::f(double In)
//{
//	double A = 1 + exp(-In);
//	return 1.0 / A;
//}
////f'=1-f^2
//double Net::f1(double In)
//{
//	double A = 1 - exp(-In), B = 1 + exp(-In);
//	return A / B;
//}