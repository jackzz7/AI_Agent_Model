#pragma once
#define __Param__ __declspec(dllexport)

#include"_Net.h"

#include<algorithm>
#include<assert.h>
#include<fstream>
#include<iostream>
//typedef unsigned int ui;

namespace Net_CPU {

	__Net__ void CharToWChar(const char*src, wchar_t*dst);
	__Net__ int File_WR(fstream&file, void*data_ptr, size_t sz, bool Write = true);
	__Net__ void ReadFromMemory(const char*& src, void* data_ptr, size_t sz);
	
	using std::swap;
	//inline void Swap(const void*L, const void*R, ui sz, void*space) {
	//	//void*dst = malloc(sz);
	//	memcpy(space, L, sz);
	//	memcpy((void*)L, R, sz);
	//	memcpy((void*)R, space, sz);
	//	//free(dst);
	//}
	template<class T>struct Data {
		T*data;
		unsigned int Count;
		unsigned int MaxCount;

		Data() {
			data = NULL;
			MaxCount = Count = 0;
		}
		Data(int dataNum) {
			if (dataNum > 0) {
				data = new T[dataNum]{ 0 };
				MaxCount = dataNum;
				Count = 0;
			}
			else new(this)Data();
		}
		void Clear() {
			MaxCount = Count = 0;
			if (data)delete[] data, data = NULL;
		}
		void _Clear() {
			memset(this, 0, sizeof(Data<T>));
		}
		void Reset(int dataNum) {
			Clear();
			new(this)Data(dataNum);
		}
		~Data() {
			Clear();
		}
		void DeCode(double*dst) {
			for (int i = 0; i < Count; i++)
				dst[i] = data[i];
		}
		void EnCode(double*src, int Num) {
			//不同堆，分配释放一致
			if (Num != MaxCount)return;
			//Reset(Num);
			for (int i = 0; i < Num; i++)
				data[i] = src[i];
			Count = Num;
		}
		template<class C>
		void push_back(C In) {
			data[Count++] = In;
		}
		void Order_Unique_Insert(T val) {
			if (Count >= MaxCount)return;
			int l = 0, r = Count;
			while (l < r) {
				int M = l + r >> 1;
				if (data[M] <= val)l = M + 1;
				else r = M;
			}
			if (l - 1 >= 0 && data[l - 1] == val)return;
			for (int i = Count; i > l; i--) {
				data[i] = data[i - 1];
			}data[l] = val;
			Count++;
		}
		void SetValue(int ParamNum, int idx,T val = 1.0,bool Val_Normalize=true) {
			idx = (std::max)(idx, 0);
			idx = (std::min)(idx, ParamNum - 1);
			if (Val_Normalize) {
				val = (std::max)(val, 0.0);
				val = (std::min)(val, 1.0);
			}
			if (Count + idx >= MaxCount)return;
			data[Count + idx] = val;
			Count += ParamNum;
		}
		void SetValue(int idx) {
			SetValue(MaxCount, idx);
		}
		void Set(int idx,const T&val = 1.0) {
			data[idx] = val;
		}
		void _ZeroMemory(int size = 0) {
			if (size == 0)size = MaxCount;
			memset(data, 0, sizeof(T)*size);
		}
		int max_value_id() {
			double mx = 0, tar = -1;
			for (int i = 0; i < Count; i++) {
				if (data[i] > mx)mx = data[i], tar = i;
			}return tar;
		}
		int Probability_id() {
			double val = 1.0*rand() / ((int)RAND_MAX);
			double sum = 0; int tar = -1;
			for (int i = 0; i < Count; i++) {
				sum += data[i];
				if (val <= sum+eps && tar == -1)tar = i;
			}
			return tar;
		}
		bool Compare(Data<T>*tar,double Range) {
			for (int i = 0; i < Count; i++) {
				//printf("%.6lf\n", abs(data[i] - tar->data[i]));
				if (abs(data[i] - tar->data[i]) > Range)return false;
			}
			return true;
		}


		inline T& operator[](int idx) {
			return data[idx];
		}
		Data<T>&operator=(const Data<T>&&R) {
			//Swap(this, &R, sizeof(Data<T>));
			swap(data, ((Data<T>*)&R)->data);
			swap(Count, ((Data<T>*)&R)->Count);
			swap(MaxCount, ((Data<T>*)&R)->MaxCount);
			return *this;
		}
		Data<T>&operator=(Data<T>&R) {
			Reset(R.MaxCount);
			memcpy(data, R.data, sizeof(T)*R.Count);
			Count = R.Count;
			return *this;
		}
		void WR(fstream&file, bool Write) {
			File_WR(file, &MaxCount, sizeof(MaxCount), Write);
			if (!Write) {
				Reset(MaxCount);
			}
			File_WR(file, &Count, sizeof(Count), Write);
			File_WR(file, data, sizeof(T)*Count, Write);
		}
		void ReadMem(const char*& src) {
			ReadFromMemory(src, &MaxCount, sizeof(MaxCount));
			if (true) {
				Reset(MaxCount);
			}
			ReadFromMemory(src, &Count, sizeof(Count));
			ReadFromMemory(src, data, sizeof(T) * Count);
		}
	};
	//typedef __Net__ Data<double> D;
	//f(a,b,c)=a^3+b^2+c+1;
	//double __Net__ f(double*In);
	//double __Net__ f(double a, double b, double c);

	struct __Param__ Base_Param {
		virtual Data<double>& In(int lstm_step) = 0;
		virtual Data<double>& Out(int lstm_step) = 0;
		virtual int LSTM_Count() = 0;
	};
	//__Net__ extern unsigned char p_space[];
	class __Param__ Param:public Base_Param
	{
	public:
		//输入数据
		Data<double>*_DataIn;
		//输出数据
		Data<double>*_DataOut;


		Param() {
			_DataIn = new Data<double>();
			_DataOut = new Data<double>();
		}
		~Param() {
			if (_DataIn)
				delete _DataIn, _DataIn = NULL;
			if (_DataOut)
				delete _DataOut, _DataOut = NULL;
		}
		Param(Param&R) :Param() {
			*this = R;
		}
		Param(const Param&R) :Param() {
			*this = R;
		}
		Param(const Param&&R) :Param() {
			*this = R;
		}
		Param&operator=(const Param&R) {
			swap(_DataIn, ((Param*)&R)->_DataIn);
			swap(_DataOut, ((Param*)&R)->_DataOut);
			//Net_CPU::Swap(this, &R, sizeof(Param), p_space);
			//Swap(_DataIn, R._DataIn);
			//Swap(_DataOut, R._DataOut);
			return *this;
		}
		Param&operator=(Param&R) {
			*_DataIn = *R._DataIn;
			*_DataOut = *R._DataOut;
			return *this;
		}
		void WR(fstream&file, bool Write) {
			_DataIn->WR(file, Write);
			_DataOut->WR(file, Write);
		}
		void ReadMem(const char*& src) {
			_DataIn->ReadMem(src);
			_DataOut->ReadMem(src);
		}
		
		Data<double>& In(int lstm_step) {
			return (*_DataIn);
		}
		Data<double>& Out(int lstm_step) {
			return (*_DataOut);
		}
		int LSTM_Count() {
			return 1;
		}
	};
	//__Net__ extern unsigned char lp_space[];
	class __Param__ LSTM_Param:public Base_Param
	{
	public:
		Param*param;
		int Count;
		LSTM_Param() {
			param = NULL;
			Count = 0;
		}
		LSTM_Param(int Count):Count(Count){
			param = new Net_CPU::Param[Count];
		}
		/*LSTM_Param(int Seq_length, int Split_Num, int element_Num) :LSTM_Param(Split_Num) {
			int len = Seq_length / Split_Num;
			for (int i = 0; i < Split_Num; i++) {
				param[i]._DataIn->Reset(len);
				for (int j = 0; j < len; j++)
					param[i]._DataIn->SetValue(1, 0, 1.0*((rand() % (element_Num - 2)) + 2) / element_Num);
			}
			for (int i = 0; i < 2; i++) {
				int No, val, idx;
				do {
					idx = rand() % len;
					No = rand() % Split_Num;
				} while ((*param[No]._DataIn)[idx] < 2.0 / element_Num);
				val = rand() % 2;
				param[No]._DataIn->Set(idx, 1.0*val/element_Num);
			}
			int Out = 0, cc = 0;
			for (int i = 0; i < Split_Num; i++) {
				for (int j = 0; j < len; j++) {
					int val = (*param[i]._DataIn)[j] * element_Num;
					if (val < 2)
						Out |= (val << cc), cc++;
				}
			}
			param[Split_Num - 1]._DataOut->Reset(1<<2);
			param[Split_Num - 1]._DataOut->SetValue(1<<2, Out, 1.0);
		}*/
		//LSTM_Param(double*In, int Num, double MaxIn):LSTM_Param(Num+1) {
		//	for (int i = 0; i < Num; i++) {
		//		param[i]._DataIn->Reset(1);
		//		param[i]._DataIn->SetValue(1, 0, In[i]/MaxIn);
		//		//param[i]._DataIn->Reset(10);
		//		//int v = In[i];
		//		//int ins = 10;
		//		//while (ins-- > 0) {
		//		//	//param[i*MaxV + MaxV - ins - 1]._DataIn->Reset(1);
		//		//	param[i]._DataIn->SetValue(1, 0, bool(v & 1));
		//		//	v >>= 1;
		//		//}
		//	}
		//	double v = f(In), MaxN = 31;
		//	double MaxOut = f(MaxIn, MaxIn, MaxIn);
		//	param[Num]._DataOut->Reset(1);
		//	param[Num]._DataOut->SetValue(1, 0, v / MaxOut);
		//	/*int ins = MaxN;
		//	param[Num]._DataOut->Reset(MaxN);
		//	while (ins-- > 0) {
		//		param[Num]._DataOut->SetValue(1, 0, bool(v & 1));
		//		v >>= 1;
		//	}*/
		//}
		LSTM_Param(LSTM_Param&R) :LSTM_Param(R.Count) {
			*this = R;
		}
		LSTM_Param(const LSTM_Param&R) :LSTM_Param() {
			*this = R;
		}
		LSTM_Param(const LSTM_Param&&R) :LSTM_Param() {
			*this = R;
		}
		LSTM_Param&operator=(const LSTM_Param&R) {
			swap(param, ((LSTM_Param*)&R)->param);
			swap(Count, ((LSTM_Param*)&R)->Count);
			//Net_CPU::Swap(this, &R, sizeof(LSTM_Param), lp_space);
			return *this;
		}
		LSTM_Param&operator=(LSTM_Param&R) {
			for (int i = 0; i < Count; i++) {
				param[i] = R.param[i];
			}
			return *this;
		}
		void WR(fstream&file,bool Write) {
			File_WR(file, &Count, sizeof(Count), Write);
			if (!Write) {
				delete[] param;
				param = new Net_CPU::Param[Count];
			}
			for (int i = 0; i < Count; i++)
				param[i].WR(file, Write);
		}
		void ReadMem(const char*&src) {
			ReadFromMemory(src, &Count, sizeof(Count));
			if (true) {
				delete[] param;
				param = new Net_CPU::Param[Count];
			}
			for (int i = 0; i < Count; i++)
				param[i].ReadMem(src);
		}

		~LSTM_Param() {
			delete[] param;
			Count = 0; param = NULL;
		}


		Data<double>& In(int lstm_step) {
			return (*param[lstm_step]._DataIn);
		}
		Data<double>& Out(int lstm_step) {
			return (*param[lstm_step]._DataOut);
		}
		int LSTM_Count() {
			return Count;
		}
	};
}