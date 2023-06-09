// Net_CPU.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include"Param.h"

#include<CUDA_ACC.h>

#include <atlconv.h>
#include <fcntl.h>

#include<time.h>

//using namespace Net_CPU;

void Net_CPU::CharToWChar(const char*src, wchar_t*dst) {
	USES_CONVERSION;
	wchar_t*tmp = A2W(src);
	lstrcpyW(dst, tmp);
}
int Net_CPU::File_WR(fstream&file, void*data_ptr, size_t sz, bool Write) {
	if (Write)
		return file.write((const char*)data_ptr, sz), -1;
	else return file.read((char*)data_ptr, sz), (file.gcount() == sz ? sz : (assert(file.gcount() == 0), 0));
}
void Net_CPU::ReadFromMemory(const char*& src, void* data_ptr, size_t sz) {
	memcpy(data_ptr, src, sz);
	src += sz;
}
//unsigned char Net_CPU::p_space[sizeof(Param)];
//unsigned char Net_CPU::lp_space[sizeof(LSTM_Param)];
using namespace _CUDA_;
int main()
{

	srand((ui)time(0));

	//int cnt = 5000;
	//LSTM_Param*param = new LSTM_Param[cnt];
	//for (int i = 0; i < cnt; i++) {
	//	int v = 16;
	//	double In[3];
	//	for (int j = 0; j < 3; j++)
	//		In[j] = rand() % v;
	//	param[i] = LSTM_Param(In, 3, v);
	//	//param[i] = Pic(TExample[i], TENum[i], Ou[i]);
	//}

	//static const int L = 3;
	//int*S = new int[L] {(int)param[0].param[0]._DataIn->Count, 100, (int)param[0].param[3]._DataOut->Count};
	//LSTM*AI = new LSTM(S, param[0].Count, 100, _Net::sigmoid);
	////G_Net*AI = new G_Net(L, S, 100,_Net::sigmoid);
	//AI->Train_LSTM(param, cnt, 4000, 0.001, 0.9, 0.9, 0.999);

	//int Cnt = 1000;
	//Param*P = new Param[Cnt];
	//for (int i = 0; i < Cnt; i++) {
	//	int prec = 16;
	//	double In[3] = { rand() % prec,rand() % prec,rand() % prec };
	//	P[i] = Param(In, 3, prec, f(prec,prec,prec));
	//}
	//const int L = 5;
	///*int *S = new int[L] { (int)P->param[0]._DataIn->Count+64, 64, (int)P->param[P->Count-1]._DataOut->Count };
	//Net_CPU::LSTM*AI = new LSTM(3, S, P->Count, 10, _Net::sigmoid);
	//AI->Train_LSTM(P, Cnt,Cnt*0.9, 0.01, 0.9, 0.9, 0.999);*/

	//int *S = new int[L]{ (int)P[0]._DataIn->Count,32,32,32,(int)P[0]._DataOut->Count };
	//Net_CPU::_Net AI(L, S, 1, _Net::sigmoid);
	//AI.Train(P, Cnt, Cnt*0.01, 0.0001, 0.9, 0.9, 0.999,0.0);

	//AI.DataWrite("_Test");
	
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
