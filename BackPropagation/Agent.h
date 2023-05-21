#pragma once
#ifdef Net_EXPORTS
#define __Net__ __declspec(dllexport)
#else
#define __Net__ __declspec(dllimport)

#endif

#define _CRT_SECURE_NO_WARNINGS


#include<CUDA_ACC.h>
#include<Param.h>
#include<conio.h>
#include<time.h>

#include<vector>
#include<list>

#include<Windows.h>

#include<functional>

#include<mutex>

//#include<algorithm>

using std::list;
using std::max;

#define InvalidFunc printf("Must overwrite this function\n"),assert(false)

using namespace _CUDA_;
using namespace Net_CPU;

namespace Base_Net {
	
	//Net Ops
	enum OpsType {
		null_Ops, InPut, OutPut, Transform, Concatenate, DotProduct, DotPlus, function,
		CustomOps,BNTransform, SpatialConcatenate,RNNInPutSwitch,Scale
	};
	enum CostFunc {
		null_Cost_Func,CrossEntropy,MeanSquaredError,BinaryCrossEntropy
	};
	enum Activation_Func
	{
		null_Func, sigmoid, tanh, relu, softmax, hardtanh, leakyhardtanh,
		thresholding
	};
	static const Device_Func Func_To_DeviceFunc[10] = { Null_Func,Sigmoid,Tanh,ReLU,Null_Func,HardTanh,Leaky_HardTanh,Thresholding };
	static const Device_Func Func_To_Derivate[10] = { Null_Func,D_Sigmoid,D_Tanh,D_ReLU,Null_Func,D_HardTanh,D_Leaky_HardTanh };
	//Net Flag
	typedef ui NetFlag;
	static const NetFlag Net_Flag_null = 0;
	static const NetFlag Net_Flag_RNN_Memory_Reset = 1 << 0;
	static const NetFlag Net_Flag_InPut_DropOut = 1 << 1;
	static const NetFlag Net_Flag_OutPut_DropOut = 1 << 2;
	static const NetFlag Net_Flag_Hidden_DropOut = 1 << 3;
	static const NetFlag Net_Flag_OutPut_Mask = 1 << 4;
	static const NetFlag Net_Flag_Factor = 1 << 5;
	static const NetFlag Net_Flag_Reconnect = 1 << 6;
	static const NetFlag Net_Flag_Reinforce = 1 << 7;
	static const NetFlag Net_Flag_Random_Data = 1 << 8;
	static const NetFlag Net_Flag_Gradient_Mask = 1 << 9;
	static const NetFlag Net_Flag_Reward_Net = 1 << 10;
	static const NetFlag Net_Flag_RNN_Initial_Step = 1 << 11;
	static const NetFlag Net_Flag_RNN_non_Initial_Step = 1 << 12;

	static const NetFlag Net_Flag_DropOut = (Net_Flag_InPut_DropOut | Net_Flag_Hidden_DropOut | Net_Flag_OutPut_DropOut);

	//Net Data file Write&Read
	typedef class Net;
	typedef __Net__ std::map<string, double> NetParam;
	typedef struct HyperParamSearcher;
	bool cmp(Net*&a, Net*&b);
	template<class T>
	bool cmp(T&it, Net*&b);
	template<class T>
	void Insert(list<T>&ls, const T&val);
	template<class T, class T1>
	void Insert(map<T, T1>&mp, const T&val);
	template<class T>
	void iter_WR(T&iter, fstream&file, Net**Net_Priority, int tot_Net_Num, bool W = true);
	void Param_WR(fstream&file, NetParam&Net_Param, HyperParamSearcher&param, bool Write = true);
	__Net__ void Train_Test(Mat&Step_Data, Mat&Data, Mat*RandomGenerator, int test_start_col);

	class __Net__ Net {
	public:
		ui Net_Node_Num;
		Activation_Func Func;

		list<Net*>Pre_Net_Head;
		list<Net*>Next_Net_Head;

		map<Net*, bool>Reconnect_Net;
		
		OpsType Ops;
		NetFlag Net_Flag;
		CostFunc CostFunction;
		
		int extra_param;

		//bp tmp variable
		Mat Gradient;
		//int grow, gcol;

		Net() {
			//grow = gcol = -1;
			Net_Node_Num = 0;
			Func = null_Func;
			Ops = null_Ops;
			Net_Flag = 0;
			CostFunction = null_Cost_Func;
			extra_param = -1;
			Clear();
		}
		Net(ui Node_Num, OpsType Op = null_Ops, NetFlag Net_flag = Net_Flag_null) :Net() {
			Net_Node_Num = Node_Num;
			Ops = Op;
			this->Net_Flag = Net_flag;
		}
		Net(ui Node_Num, CostFunc Cost_Func, ui Net_flag = Net_Flag_null, int Scale_Factor = -1) :Net() {
			Net_Node_Num = Node_Num;
			this->Net_Flag = Net_flag;
			CostFunction = Cost_Func;
			this->extra_param = Scale_Factor;
		}
		Net(ui Node_Num, Activation_Func Function, ui Net_flag = Net_Flag_null) :Net() {
			Net_Node_Num = Node_Num, Func = Function;
			this->Net_Flag = Net_flag;
		}
		Net(Activation_Func Function, ui Net_flag = Net_Flag_null) :Net() {
			Func = Function;
			this->Net_Flag = Net_flag;
		}
		Net(ui Node_Num, int extra_param, ui Net_flag = Net_Flag_null):Net() {
			Net_Node_Num = Node_Num;
			this->extra_param = extra_param;
			this->Net_Flag = Net_flag;
		}
		virtual ~Net() {
			Clear();
		}
		void Clear() {
			Pre_Net_Head.clear();
			Next_Net_Head.clear();
			Reconnect_Net.clear();
		}
		//A=>B
		Net*Add_Forward(OpsType Op, Net*Pre, ui Extra_Flag = Net_Flag_null) {
			Ops = Op;
			Pre_Net_Head.push_back(Pre);
			Pre->Next_Net_Head.push_back(this);
			if (Extra_Flag&Net_Flag_Reconnect) {
				Pre->Net_Flag |= Net_Flag_RNN_Memory_Reset;
				Reconnect_Net[Pre] = true;
			}
			if (Ops == BNTransform || Ops == Scale)
				Net_Node_Num = Pre->Net_Node_Num;
			return this;
		}
		//muli Forward available
		//A,B=>C
		Net*Add_Pair_Forward(OpsType Op, Net*First_Pre, Net*Second_Pre, ui Extra_First_Flag = Net_Flag_null, ui Extra_Second_Flag = Net_Flag_null) {
			Add_Forward(Op, First_Pre, Extra_First_Flag);
			Add_Forward(Op, Second_Pre, Extra_Second_Flag);
			return this;
		}
		//DropOut,OutPut Mask
		//DorpOut must around Transform
		Net*Add_Mask(ui Mask_Flag, Net*Mask_Net, OpsType Op = DotProduct) {
			this->Net_Flag = Mask_Flag;
			this->Net_Node_Num = Mask_Net->Net_Node_Num;
			//First OutPut value,second DropOut
			Add_Pair_Forward(Op, Mask_Net, new Net(Mask_Net->Net_Node_Num, InPut, Net_Flag_Factor));
			return this;
		}
		//Disponse all no-data resource
		virtual void Reduction(int Max_Step = -1) { /*Gradient.Clear();*/ /*result.Clear();*/ }
		virtual Mat& GetOutPut(int lstm_idx = -1) { Mat ret(1, 1); InvalidFunc; return ret; }
		virtual bool Backward_Gradient(Mat*&result,Net*Recall_Pre_Net,int lstm_idx = -2,int Max_Step=-1) { InvalidFunc; return false; }
		virtual void Calculate_Bias(int lstm_idx) {}
		virtual double Forward(int lstm_idx = -1, int IsTrain = false,bool cal_loss=true) { return InvalidFunc,0; }
		virtual void LSTM_Save(int idx) {}
		virtual void Cal_Reward(int idx, Mat&Reward) { InvalidFunc; }
		virtual void Cal_Status_Reward(int idx, Mat&Reward, Net*Reward_Net) { InvalidFunc; }
		virtual void Cal_TD_Reward(int idx, Net*Reward_Net, double*Reward, int Ins_Max_Step, int Type) { InvalidFunc; }
		virtual void Sum_Reward(int idx, int Max_Step) { InvalidFunc; }
		virtual void RandData() { InvalidFunc; }
		virtual Mat& GetStartRandomData() { Mat ret(1, 1); InvalidFunc; return ret; }
		//Data file Write&Read
		virtual void Data_WR(fstream&file, Net**Net_Priority, int tot_Net_Num, bool Write = true,int Batch=1,int Max_Step=1) { Data_Write_Read(file, Net_Priority, tot_Net_Num, Write); }

		inline Mat* GetPreOutPut(int lstm_idx, Net*Pre) {
			int _idx = lstm_idx;
			//Recycle Pre Step Node
			if (Reconnect_Net.find(Pre) != Reconnect_Net.end())
				_idx--;
			//RNN
			if (lstm_idx >= 0 && _idx < 0) {
				//Random Data
				if (Pre->Net_Flag & Net_Flag_Random_Data)
					return &Pre->GetStartRandomData();
				//Zero
				else {
					printf("not implement\n"), assert(false);
					//result->_ZeroMemory_Valid(Pre->GetOutPut().GetRow(), Pre->GetOutPut().GetCol());
				}
			}
			else return &Pre->GetOutPut(_idx);
		}
		void Backward(int idx = -2, int Max_RNN_Step = -1) {
			//Gradient = &GetOutPut();
			//Gradient->_ZeroMemory_Valid(getRow(), getCol());
			Gradient._ZeroMemory_Valid(GetOutPut().GetRow(), GetOutPut().GetCol());
			Mat result(Gradient.GetRow(), Gradient.GetCol(), false);
			for (auto k : Next_Net_Head) {
				int _idx = idx;
				if (k->Reconnect_Net.find(this) != k->Reconnect_Net.end())
					_idx++;
				//Recycle Next Step Node
				if (_idx >= Max_RNN_Step)continue;
				Mat*result_ptr = &result;
				if (k->Backward_Gradient(result_ptr, this, _idx, Max_RNN_Step)) {
					//reduce hidden Gradient,to stay constant
					if (_idx != idx)
						Gradient += 0.5f*(*result_ptr);
					else Gradient += (*result_ptr);
				}
			}
			Calculate_Bias(idx);
		}
		bool Loss_Gradient(Mat&OutPut, Mat&Out, Mat&result) {
			switch (CostFunction)
			{
			case CrossEntropy:
			case MeanSquaredError:
				result = (OutPut - Out)*(extra_param == -1 ? 1 : (1.0 / extra_param));
				return true;
				break;
			case BinaryCrossEntropy:
				result = -1.0*(extra_param == -1 ? 1 : (1.0 / extra_param))*Out / OutPut;
				return true;
				break;
			default:
				assert(false);
				break;
			}
			return false;
		}
		virtual void Update(NetParam&Net_Param, Net*dst_Net) {}
		virtual void Data_Assignment(Net*dst_Net, int Max_Step) { InvalidFunc; }
		virtual void Assignment(double*Weight, double*b) {}
		double Cal_Loss() {
			Net*Pre = Pre_Net_Head.front();
			switch (CostFunction)
			{
			case CrossEntropy:
				return (Pre->GetOutPut()._f(Ln) *= GetOutPut()).Sum().ReadFromDevice(true)[0] * -1.0*(extra_param == -1 ? 1 : (1.0 / extra_param));
				break;
			case MeanSquaredError:
				return (Pre->GetOutPut() - GetOutPut()).f(Pow2).Sum().ReadFromDevice(true)[0] / 2.0*(extra_param == -1 ? 1 : (1.0 / extra_param));
				break;
			case BinaryCrossEntropy:
				break;
			default:
				assert(false);
				break;
			}
			return 0;
		}
		void Test(double*cor, Mat*Reward = NULL, double*Loss = NULL) {
			switch (CostFunction)
			{
			case CrossEntropy: 
			{
				Mat OutPut = Pre_Net_Head.front()->GetOutPut();
				//OutPut.ReadFromDevice();
				if (cor || Reward) {
					Mat Out = GetOutPut();
					OutPut.SoftMax(true) *= Out.SoftMax(true);
				}
				if (cor) {
					double val = OutPut.ScaleOneCol(true).f(Bool).Sum().ReadFromDevice(true)[0];
					assert(val - (int)val == 0);
					assert(val <= OutPut.GetCol());
					*cor += val;
				}
				if (Reward)
					*Reward = OutPut.ScaleOneCol(true);
			}
				break;
			case MeanSquaredError:
				//OutPut.ReadFromDevice();
				break;
			case BinaryCrossEntropy:
				break;
			default:
				break;
			}
			//GetOutPut().ReadFromDevice();
			if (Loss)*Loss += Cal_Loss();
		}
		void Activate(Mat&OutPut) {
			if (Func == null_Func) {}
			else if (Func == softmax) {
				OutPut.SoftMax();
			}
			else if (Func == thresholding)
				OutPut.f(Thresholding, 1.0);
			else OutPut.f(Func_To_DeviceFunc[Func]);
		}
		void Derivate(int lstm_idx) {
			if (Func == null_Func || Func == softmax || Func == thresholding) {}
			else Gradient *= GetOutPut(lstm_idx)._f(Func_To_Derivate[Func]);
		}
		Net*GetMask(NetFlag Mask_Flag) const{
			if (Net_Flag&Mask_Flag) {
				assert(Pre_Net_Head.size() == 2);
				return Pre_Net_Head.back();
			}return NULL;
		}
		
		void Data_Write_Read(fstream&file, Net**Net_Priority, int tot_Net_Num, bool Write = true) {
			File_WR(file, (char*)&Net_Node_Num, sizeof(ui), Write);
			File_WR(file, (char*)&Func, sizeof(Activation_Func), Write);
			File_WR(file, (char*)&Ops, sizeof(OpsType), Write);
			File_WR(file, (char*)&Net_Flag, sizeof(ui), Write);
			File_WR(file, (char*)&CostFunction, sizeof(CostFunc), Write);
			File_WR(file, (char*)&extra_param, sizeof(extra_param), Write);
			iter_WR(Pre_Net_Head, file, Net_Priority, tot_Net_Num, Write);
			iter_WR(Next_Net_Head, file, Net_Priority, tot_Net_Num, Write);
			iter_WR(Reconnect_Net, file, Net_Priority, tot_Net_Num, Write);
		}
		//simple net campare
		bool operator==(Net*const R) const {
			//except Net_Flag
			return Net_Node_Num == R->Net_Node_Num&&Func == R->Func&&
				Ops == R->Ops&&CostFunction == R->CostFunction&&extra_param == R->extra_param;
		}
		//check if require store OutPut
		bool CheckSaveCondition() {
			if (Ops == OpsType::function || Func != null_Func || Ops == OpsType::OutPut)return true;
			for (auto&nxt : Next_Net_Head) {
				if (nxt->Ops == DotProduct || nxt->Ops == OpsType::OutPut || nxt->Ops == Transform)return true;
			}
			return false;
		}
		/*inline int getRow() {
			if (grow == -1)grow = GetOutPut().GetRow();
			return grow;
		}
		inline int getCol() {
			if (gcol == -1)gcol = GetOutPut().GetCol();
			return gcol;
		}*/
	};
	struct __Net__ HyperParamSearcher {
		double _best;
		map<string, double>_param, _best_param;
		string _param_path, _best_path;
		HyperParamSearcher(const string&path = "_train_hyper_param", const string&best_path = "_best_hyper_param") {
			clear();
			_param_path = path, _best_path = best_path;
		}
		~HyperParamSearcher() {
			clear();
		}
		void SetPath(const string&path, const string&best_path) {
			_param_path = path, _best_path = best_path;
		}
		void Random_Uniform(const string&param, double Max, double Min) {
			_param[param] = Min + rand() % ((ui)(RAND_MAX + 1))*(Max - Min) / RAND_MAX;
		}
		void Random_Uniform_Step(const string&param, double Step, int Max_Step_Num) {
			_param[param] = ((rand() % Max_Step_Num) + 1)*Step;
		}
		//0<Min<Max
		void Draw_Geometric(const string&param, double Min, double Max, bool round = false,bool OneMinus=false) {
			double a = log(Min), b = log(Max);
			double res = (rand() % (RAND_MAX + 1))*(b - a) / RAND_MAX + a;
			if (round)
				_param[param] = ::round(exp(res));
			else _param[param] = exp(res);
			if (OneMinus)
				_param[param] = 1.0 - _param[param];
		}
		void Compare(const double&correct) {
			if (correct > _best) {
				_best = correct;
				_best_param = _param;
				hyper_param_Save(_best_path.c_str(), _best, _best_param, false);
			}
			hyper_param_Save(_param_path.c_str(), correct, _param, true);
		}
		void hyper_param_Save(const char* path, double best, map<string, double>&param, bool append) {
			fstream file;
			if (append)
				file.open(path, ios::out | ios::app);
			else file.open(path, ios::out | ios::trunc);
			char str[100];
			ui sz = sprintf(str, "correct:%lf%\n", best);
			file.write(str, sz);
			for (auto k : param) {
				sz = sprintf(str, "--%s: %lf\n", k.first.c_str(), k.second);
				file.write(str, sz);
			}file.write("\n", 2);
			file.close();
		}
		void Read_Param(string path = "") {
			if (path == "")path = _best_path;
			fstream file; file.open(path.c_str(), ios::in);
			if (!file)printf("Param file %s not exist\n",path.c_str()),assert(false);
			//freopen(path.c_str(), "r", stdin);
			char Name[100], val[100], buf[100];
			int sz = 0;
			file.getline(buf, 100);
			sscanf(buf,"%[^:]:%[^\n]", Name, val); _best = atof(val);
			while (file.getline(buf, 100),file.gcount()>0) {
				if (sscanf(buf, " --%[^:]: %[^\n]", Name, val) < 2)break;
				_param[Name] = atof(val);
			}
			_best_param = _param;
			//freopen("CON", "r", stdin);
			file.close();
		}
		double& operator[](const string&param) {
			return _param[param];
		}
		void clear() {
			_param.clear();
			_best_param.clear();
			_best = 0;
		}
	};
	class __Net__ Agent {
	public:
		enum Net_Type {
			Full_Connected, RNN
		};
		Net_Type NT;
		//Agent params
		NetParam Net_Param;

		list<Net*>InPut_Net;
		list<Net*>OutPut_Net;
		list<Net*>Reinforce_Net;
		//Net*Reward_OutPut_Net;

		int RNN_Max_Steps;

		int tot_Net_Num;
		Net**Net_Priority;

		//tmp variable
		Mat*All_In;
		Mat*All_Out;
		

		Agent(Net_Type Type, int Batch, Net*Start_Net, int RNN_Max_Steps = 0x7fffffff, bool Reset_All_Cuda_States = true, time_t random_seed = time(0), int Max_srand_row = -1, bool ReadFile = false);
		~Agent() {
			for (int i = 0; i < tot_Net_Num; i++)
				delete Net_Priority[i];
			delete[] Net_Priority;
			delete[] All_In;
			delete[] All_Out;
			Net_Param.clear();
			OutPut_Net.clear();
			Reinforce_Net.clear();
		}
		void Func_Init(int MaxRandomStates_rows) {
			Matrix_Set_Function(NULL, NULL, MaxRandomStates_rows);
		}
		//RNN Based Net
		typedef bool(*Train_Option)(Agent*agent, Mat*RandomGenerator, int step, int test_start_col);
		typedef void(*DataWrite2Device)(Base_Param**data, int paramNum, Agent*agent, bool W_Optimize);
		static bool Default_Train_Func(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col = -1);
		static void Default_Data2Device(Base_Param**data, int paramNum, Agent*agent, bool W_Optimize = false);
		template<typename Train_Func>
		double Train(Base_Param** param, int paramNum, int trainNum, const char* Train_Param, condition_variable* data_CV = NULL, Train_Func train_func = Default_Train_Func, DataWrite2Device data_proc_func = Default_Data2Device);
		double Forward(int IsTrain = false, int Rnn_Step = -1, bool backward = true, bool cal_loss = true, double* L2_Loss = NULL, double* MSE_Loss = NULL);
		void Backward(int idx = -2, int RNN_Max_Steps = -1);
		void Update(Agent*Server = NULL);
		void getGradient(Agent*Worker,double Scale);
		//double Test(Mat&C_In, Mat&C_Out, int testNum);
		typedef double(*TestScoreFunc)(double Cor, vector<double>&part_Loss, double TestNum);
		template<typename Train_Func>
		double Test(int OutPut_Num, int Positive_Num, int trainNum, int testNum, Train_Func func = Default_Train_Func, TestScoreFunc Score = NULL);
		double Data_Test(Base_Param**param, int paramNum, Train_Option train_func = Default_Train_Func, DataWrite2Device data_proc_func = Default_Data2Device, TestScoreFunc Score = NULL);
		void DropOut(bool IsTrain, bool Used = true);
		void Reinforce_Reward(int Step, Net*Reward_Net, double*Step_Reward,int Ins_Max_Step,int Type);
		//Online Train
		int lstm_Step = -1, BatchTrainCount = 0;
		int BatchRandomNumber = 0;
		 //Init Agent
		void Online_Init(HyperParamSearcher&Agent_Param) {
			Init_Param(Agent_Param);
			Speeder = Learning_rate(Net_Param["Speed"], Net_Param["Speed_decay_time"]);
			BatchTrainCount = 0;
		}
		//New Turn
		void Online_New_Turn() {
			//Random Start InPut
			RNN_Init();
			lstm_Step = -1;
		}    
		void Online_Next_Step() {
			lstm_Step++;
			//Reset Gradient mask
			Net*mask = NULL;
			for (int j = 0; j < tot_Net_Num; j++)
				if (mask = Net_Priority[j]->GetMask(Net_Flag_Gradient_Mask | Net_Flag_OutPut_Mask))
					mask->GetOutPut().f(Assignment, 1.0);
		}
		//Info fed to Agent 
		void Online_InPut(Base_Param**InPut_Data) {
			assert(false);
			//int Batch = Net_Param["Batch"];
			//InPut_Net->GetOutPut().Reset(InPut_Net->Net_Node_Num, Batch, false);
			////double**Data = new double*[InPut_Net->Net_Node_Num];
			//for (int i = 0; i < InPut_Net->Net_Node_Num; i++) {
			//	//Data[i] = new double[Batch];
			//	for (int j = 0; j < Batch; j++)
			//		InPut_Net->GetOutPut()[i*Batch + j] = InPut_Data[j]->In(lstm_Step)[i];
			//}
			//InPut_Net->GetOutPut().WriteToDevice();
		}
		//Agent Response
		typedef void(*Encode)(list<Net*>OutPut_Nets, Base_Param**param, ui Batch, int lstm_step);
		void Online_OutPut(Base_Param**Agent_Response, bool IsTrain,Encode encoder,bool backward) {
			//not use DropOut
			DropOut(IsTrain, false);
			Forward(IsTrain, lstm_Step, backward, false);
			if (encoder)
				encoder(OutPut_Net, Agent_Response, Net_Param["Batch"], lstm_Step);
		}
		template<class T>
		void Online_Mask(ui row, ui col, T mask, ui mask_Net_Flag, int lstm_step = -1, bool mul = true) {
			//mask
			Mat Mask(row, col, mask);
			for (int j = 0; j < tot_Net_Num; j++)
				if (Net_Priority[j]->Net_Flag&mask_Net_Flag) {
					if (mul)
						Net_Priority[j]->Pre_Net_Head.back()->GetOutPut(lstm_step) *= Mask;
					else Net_Priority[j]->Pre_Net_Head.back()->GetOutPut(lstm_step) -= Mask;
				}
		}
		void Online_GetReward(int step, double*Step_Reward,Agent*RewardNet,int Type) {
			Reinforce_Reward(step, RewardNet->OutPut_Net.front()->Pre_Net_Head.front(), Step_Reward, RewardNet->lstm_Step + 1,Type);
		}
		void Online_GetReward(int step, Mat&Step_Reward) {
			for (auto RL : Reinforce_Net)
				RL->Cal_Reward(step, Step_Reward);
		}
		void Online_Update() {
			for (int i = lstm_Step; i > -1; i--)
				Backward(i, lstm_Step + 1);
			Update();
			Speeder.Update(++BatchTrainCount, Net_Param["Speed"]);
		}


		void RNN_Init() {
			BatchRandomNumber = rand();
			for (int i = 0; i < tot_Net_Num; i++)
				if (Net_Priority[i]->Net_Flag&Net_Flag_RNN_Memory_Reset)
					if (Net_Priority[i]->Net_Flag&Net_Flag_Random_Data)
						Net_Priority[i]->RandData();
					else Net_Priority[i]->GetOutPut()._ZeroMemory();
		}

		void Default_Param() {
			Net_Param["Batch"] = 1.0;
			Net_Param["Max_Step"] = 1.0;
			Net_Param["Speed"] = 0.001;
			Net_Param["Beta1"] = 0.9;
			Net_Param["Beta2"] = 0.999;
			Net_Param["Hidden_DropOut"] = 0.5;
			Net_Param["InPut_DropOut"] = 0.2;
			Net_Param["OutPut_DropOut"] = 0.5;
			Net_Param["Stochastic"] = 1.0;
			Net_Param["Beta1_Pow"] = 1.0;
			Net_Param["Beta2_Pow"] = 1.0;
			Net_Param["Max_Epoch"] = 1000;
			Net_Param["Speed_decay_time"] = 10000;
			Net_Param["EarlyStop"] = 1.0;
			Net_Param["ExtraTrainFactorPerEpoch"] = 1;
			Net_Param["ExtraTestFactorPerEpoch"] = 1;
			Net_Param["Reward_From_SL"] = 0;
			Net_Param["Optimizer"] = 0;
			Net_Param["EpochUpdate"] = 0;
			Net_Param["L2_Factor"] = 1e-4;
		}
		template<class T>
		void Init_Param(const T&param) {
			Prase_Param(Net_Param, param);
		}
		static void Prase_Param(NetParam&Net_Param,const HyperParamSearcher&param) {
			for (auto k : param._param) {
				if (Net_Param.find(k.first) != Net_Param.end()) {
					Net_Param[k.first] = k.second;
				}
			}
		}
		//string prase
		static void Prase_Param(NetParam&Net_Param,const string&param){
			size_t sz = param.size();
			int last = 0;
			string param_Name = "";
			for (int i = 0; i <= sz; i++) {
				if (i == sz || param[i] == ' ') {
					if (Net_Param.find(param_Name) != Net_Param.end())
						Net_Param[param_Name] = atof(param.substr(last, i - last).c_str());
					else if (!param_Name.empty())printf("not find '%s' parameter \n", param_Name.c_str());
					last = i + 1;
				}
				else if (param[i] == ':') {
					param_Name = param.substr(last, i - last);
					last = i + 1;
				}
			}
		}
		//use for Adaptive Speed
		struct __Net__ Learning_rate
		{
			double ori_Speed;
			double iter_Point;
			Learning_rate() {}
			Learning_rate(const double&Speed, const double&decay_time_Point) :ori_Speed(Speed), iter_Point(decay_time_Point) {}
			void Update(ui Batch_iterations,double&_Speed) {
				if (Batch_iterations < iter_Point) {}
				//Min Speed 0.00001
				else if(_Speed>0.00001){
					_Speed = ori_Speed * iter_Point / Batch_iterations;
				}
			}
			void Step_Decay(ui Epochs_Drop,double&_Speed) {
				_Speed = max(ori_Speed * pow(0.1, floor(Epochs_Drop / iter_Point)), 1e-5);
			}
		};
		Learning_rate Speeder;

		Agent(const char*file_Path, int Batch, int RNN_Max_Steps = 0x7fffffff, int Max_srand_row = -1, bool Reset_Cuda = true) {
			fstream file; file.open(file_Path, ios::in | ios::binary);
			if (file) {
				//Net Param
				HyperParamSearcher param;
				Param_WR(file, Net_Param, param, false);
				Net*net = Net_WR(file, Batch, RNN_Max_Steps, false);
				if (net) {
					new(this)Agent(NT, Batch, net, RNN_Max_Steps, Reset_Cuda, time(0), Max_srand_row, true);
					//Read Net Weight&b
					NetData_WR(file, false);
					//Init agent param
					Online_Init(param);
				}
				else DEBUG(true, "Net Read Error\n");
			}
			else DEBUG(true, "Agent File Not Exist\n");
			file.close();
		}
		void Write_to_File(const char*file_Path) {
			fstream file; file.open(file_Path, ios::out | ios::trunc | ios::binary);
			if (file) {
				//Write Net Param
				HyperParamSearcher param;
				Param_WR(file, Net_Param, param);
				//Write Net Structure
				Net_WR(file);
				//Net Weight&b Data
				NetData_WR(file);
			}
			else DEBUG(true,"Agent File Can't Open\n");
			file.close();
		}
		//Data WR
		Net* Net_WR(fstream&file, int Batch = 1, int RNN_Max_Steps = 1, bool Write = true);
		void NetData_WR(fstream&file, bool Write = true);
		void Data_Assignment(Agent*dst_Agent);
		//Recycle no-data resource
		void reduction() {
			for (int i = 0; i < tot_Net_Num; i++)
				Net_Priority[i]->Reduction(RNN_Max_Steps);
		}
	};
}