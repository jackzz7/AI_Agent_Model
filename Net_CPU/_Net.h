#pragma once
#ifdef NET_EXPORTS
#define __Net__ __declspec(dllexport)
#else
#define __Net__ __declspec(dllimport)

#endif

#include<fstream>
#include<map>

namespace Net_CPU {

	using std::fstream;
	class Param;
	const double eps = 1e-8;
	
	class __Net__ _Net
	{
	public:
		enum _Activation_Func
		{
			null, sigmoid, tanh, relu, softmax,hardtanh,leakyhardtanh,
			Thresholding
		};
	protected:
		//总共有L层神经元
		//第一层为输入层，第L为输出层，2->（L-1）为隐含层
		//m个输入数，组成输入向量
		//n个输出数，组成输出向量
		//S[l]为相应层的神经元个数
		int L;
		//样本越多，需要的隐神经元越多
		int*S;
		//训练样本数
		int Num;
		//输入
		double*In;
		//输出
		double*Out;
		//double Speed;
		//double ForgetFactor;
		_Activation_Func Output_Fun;
		_Activation_Func*Layer_Func;
		double Sigmoid_eps;
		int Negative_OutPut_No;

		//std::map<,double>Net_Param;
		//还原DropOut缩放
		//double InPut_DropOut_Scale;
		//double Hidden_DropOut_Scale;
		//非线性激活函数
		//f'=f(1-f)
		static double Sigmoid(double In);
		//f'=1-f^2
		static double Sigmoid_(double In);
		static double Tanh(double In);
		static double ReLU(double In);
		static void SoftMax(double*In, int Num);
		//Drop out Possibility
		static bool Bernoulli(double P) {
			return (1.0*rand() / RAND_MAX) >= P;
		}
		void ComputerProc();
	public:
		//输入层
		struct __Net__ InPutUnit
		{
			int No;
			int*S;
			double *InPut;
			//DropOut
			double*DropOut;
			InPutUnit(int*S);
			InPutUnit(int*S,int No);
			void Clear();
			void Disponse();
		}*IPU;
		//输出层和隐含层
		struct __Net__ TrainUnit
		{
			//当前层数
			int No;
			//层数信息
			int*S;
			//偏置值
			double*b;
			//累计偏置差值
			double*Biasb;
			//动量项，提高稳定性
			double*Lastb;
			//动量项平方
			double*Lastb2;
			//Adam参数
			double*b_Beta1;
			double*b_Beta2;
			//对应于上一层各个神经层的连接权重
			double**Weigh;
			//累计权差值(多个样本输入)
			double**BiasWeigh;
			double**LastWeigh;
			double**LastWeigh2;
			double**Weigh_Beta1;
			double**Weigh_Beta2;
			//神经元输出
			double*OutPut;
			//保留各路权值和+偏置值，后用
			double*Net;
			//保留计算用参数
			double*Param;
			//DropOut
			//缓解过拟
			//尤其小训练样本
			double*DropOut;

			TrainUnit(int*S, int No);
			TrainUnit(int Last_Node_Num, int Node_Num);
			//释放资源
			void Disponse();
			TrainUnit();
			//自下而上
			//计算神经元输出
			void ComputeOutPut(_Activation_Func Activate_Fun,InPutUnit*IPU, TrainUnit *LastTU = NULL);
			//自上而下，梯度下降算法，修改权值
			void AdjustWeighb(int L, double*Out, InPutUnit*IPU, TrainUnit *LastTU = NULL, TrainUnit*NextTU = NULL,_Activation_Func Output_Fun=softmax);
			//完成一次批量训练时调用，权，偏置差值清零
			void RefreshData(int Num, double Speed,double ForgetFactor, double Beta1, double Beta2, TrainUnit*LastUnit=NULL);
			void DataWrite(fstream*file);
			void DataRead(fstream*file);

			bool operator==(TrainUnit&R) const {
				if (No != R.No)return false;
				for (int i = 0; i < S[No]; i++)
					if (abs(b[i] - R.b[i]) > eps)return false;
				for (int i = 0; i < S[No]; i++)
					for (int j = 0; j < S[No - 1]; j++)
						if (abs(Weigh[i][j] - R.Weigh[i][j]) > eps)return false;
				return true;
			}
			bool operator!=(TrainUnit&R)const {
				return !this->operator==(R);
			}
		}*TU, *OPU;
	public:
		TrainUnit&getTU(int id) const { return TU[id]; }
		TrainUnit&getOPU() const { return *OPU; }
		bool IsVaild();
		_Net() {};
		_Net(int L,int*S, int BatchNum = 1, _Activation_Func Output_Fun = softmax,double Sigmoid_eps=0.5,int Negative_OutPut_No=-1, _Activation_Func*Hidden_Layer_Func=NULL);
		void Disponse();
		_Net(const char*filePath, int BatchNum = 1, _Activation_Func Output_Fun = softmax, double Sigmoid_eps = 0.5, int Negative_OutPut_No = -1);
		int Train(Param*param, int paramNum, int trainNum, double Speed = 0.001, double ForgetFactor = 0.9,double Beta1=0.9,double Beta2=0.999,double DropOut=0.5);
		//void AdjustParam(double Beta1, double Beta2, double&M_Beta1, double&M_Beta2);
		void DataWrite(const char*filePath);
		void DataWrite(const wchar_t*filePath);
		void Net_Write(fstream&file);
		bool DataRead(const char*filePath);
		void Response(Param&param);
		double Test(Param*param, int testNum);
		bool operator==(_Net&R) const {
			if (L != R.L)return false;
			for (int i = 0; i < L; i++)
				if (S[i] != R.S[i])return false;
			for (int i = 0; i < L - 2; i++)
				if (TU[i] != R.getTU(i))return false;
			if (*OPU != R.getOPU())return false;
			return true;
		}
		bool operator!=(_Net&R) const { return !(*this == R); }
		~_Net() {
			Disponse();
		}
	};
}