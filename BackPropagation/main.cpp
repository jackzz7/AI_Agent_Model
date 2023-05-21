#include"G_Net.h"
#include"RNN_LSTM.h"
#include<vector>
#include<time.h>

#include<Windows.h>

#include"Agent.h"
#include"Attention.h"
#include"ConvNet.h"

const int ExmMaxNum = 2 * (60000 + 10000);
const int MaxPicPoint = 400;
POINT TExample[ExmMaxNum][MaxPicPoint];
int TENum[ExmMaxNum];
int TNum = 0;
//像素倒序
bool BWPicData[100][100];
POINT PicBPoint[MaxPicPoint];
int PBPNum = 0;
int PicW, PicH;

int Ou[ExmMaxNum];

//namespace _Win {

#include<gdiplus.h>
#include<atlconv.h>
#pragma comment(lib, "Gdiplus.lib")
#include"Header.h"

Gdiplus::GdiplusStartupInput gdiplusStartupInput;
ULONG_PTR gdiplusToken;
using namespace Gdiplus;
#define Bitmap  Gdiplus::Bitmap
WCHAR A[ExmMaxNum][100];
	
void AchievePicFromFile(WCHAR*Path)
{
	BitmapData *BD1 = new BitmapData();
	Bitmap *bitmap = new Bitmap(Path);
	int MWidth = bitmap->GetWidth(), MHeight = bitmap->GetHeight();
	PicW = MWidth, PicH = MHeight;
	Rect R = { 0,0,MWidth,MHeight };
	bitmap->LockBits(&R, ImageLockMode::ImageLockModeRead, bitmap->GetPixelFormat(), BD1);
	if (MWidth == 0)
	{
		assert(false);
		//std::cout << "This Pic File Don't Exist!";
		return;
	}
	int k = 0, Stride = abs(BD1->Stride) / MWidth * MWidth;

	memset(BWPicData, 0, sizeof(bool)*MWidth*MHeight);
	PBPNum = 0;
	char*Data = (char*)BD1->Scan0;
	for (int i = 0; i < MHeight; i++)
		for (int j = 0; j < Stride; j += 3)
		{
			int J = Data[i*BD1->Stride + j];
			if ((ui)Data[i*BD1->Stride + j] >50 && (ui)Data[i*BD1->Stride + j + 1] >50 && (ui)Data[i*BD1->Stride + j + 2] >50)
			{
				BWPicData[i][j / 3] = true;
				PicBPoint[PBPNum++] = { i,j / 3 };
			}
		}
	bitmap->UnlockBits(BD1);
	delete BD1;
	delete bitmap;
}
void GetPic(string TrainPath,string TestPath,int&Train_Num,int&Test_Num) {
	//CreateThread(NULL, 0, ThreadProc, NULL, 0, NULL);
	Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
	srand((ui)time(0));

	int NP = 0;
	USES_CONVERSION;
	LPCTSTR lpFileName = A2W((TrainPath + "*.jpg").c_str());
	WIN32_FIND_DATA pNextInfo = {};

	HANDLE handle = FindFirstFile(lpFileName, &pNextInfo);
	if (handle != INVALID_HANDLE_VALUE)
	{
		lstrcpy(A[0], pNextInfo.cFileName);
		Ou[0] = ((int)A[NP][lstrlen(A[0]) - 5] - 48);
		NP++;
		while (FindNextFile(handle, &pNextInfo))
		{
			lstrcpy(A[NP], pNextInfo.cFileName);

			Ou[NP] = ((int)A[NP][lstrlen(A[NP]) - 5] - 48);
			NP++;
		}
	}
	Train_Num = NP;

	//AchievePicFromFile(L"test.jpg");
	LPCTSTR dir = A2W((TrainPath).c_str());
	WCHAR Text[100];
	for (int i = 0; i < NP; i++) {
		//CharToWChar(A[i], Text);
		WCHAR path[100];
		lstrcpy(path, dir);
		AchievePicFromFile(lstrcat(path, A[i]));
		memcpy_s(TExample[i], sizeof(TExample[i]), PicBPoint, sizeof(PicBPoint));
		TENum[i] = PBPNum;
		if (PBPNum > MaxPicPoint) {
			assert(false);
			//std::cout << "Wrong\n";
		}
		TNum++;
	}
	//Test example
	lpFileName = A2W((TestPath + "*.jpg").c_str());
	pNextInfo = {};
	int Start = NP;
	handle = FindFirstFile(lpFileName, &pNextInfo);
	if (handle != INVALID_HANDLE_VALUE)
	{
		lstrcpy(A[NP], pNextInfo.cFileName);
		Ou[NP] = ((int)A[NP][lstrlen(A[NP]) - 5] - 48);
		NP++;
		while (FindNextFile(handle, &pNextInfo))
		{
			lstrcpy(A[NP], pNextInfo.cFileName);

			Ou[NP] = ((int)A[NP][lstrlen(A[NP]) - 5] - 48);
			NP++;
		}
	}
	Test_Num = NP - Start;
	dir = A2W((TestPath).c_str());
	for (int i = Start; i < NP; i++) {
		//CharToWChar(A[i], Text);
		WCHAR path[100];
		lstrcpy(path, dir);
		AchievePicFromFile(lstrcat(path, A[i]));
		memcpy_s(TExample[i], sizeof(TExample[i]), PicBPoint, sizeof(PicBPoint));
		TENum[i] = PBPNum;
		if (PBPNum > MaxPicPoint) {
			assert(false);
			//std::cout << "Wrong\n";
		}
		TNum++;
	}
}
Bitmap *bitmap = NULL;
void GdiBitmapSave(const WCHAR*Form, WCHAR*path)
{
	CLSID imageCL;
	WCHAR Format[12] = L"image/";
	lstrcat(Format, Form);
	if (lstrcmp(Format, L"image/jpg") == 0)
		lstrcpy(Format, L"image/jpeg");
	{
		using namespace Gdiplus;
		UINT num = 0, size = 0;
		ImageCodecInfo*ICI = NULL;
		GetImageEncodersSize(&num, &size);
		ICI = (ImageCodecInfo*)malloc(size);
		GetImageEncoders(num, size, ICI);
		for (int i = 0; i < num; i++)
		{
			if (wcscmp(ICI[i].MimeType, Format) == 0)
			{
				imageCL = ICI[i].Clsid;
				free(ICI);
				break;
			}
		}
	}
	bitmap->Save(path, &imageCL);
}
void GererateNumberPic(const char*InPath,const char*LabelPath, int Num,const WCHAR*OutPath)
{
	//Graphics G;
	byte*Data = new byte[28 * 28 * Num + 4 * 4], *LabData = new byte[1 * Num + 4 * 2];
	FILE*ImageF = NULL;
	FILE*LabelF = NULL;
	fopen_s(&ImageF, InPath, "r+b");
	fread_s(Data, 28 * 28 * Num + 4 * 4, 1, 28 * 28 * Num + 4 * 4, ImageF);
	fopen_s(&LabelF, LabelPath, "r+b");
	fread_s(LabData, 1 * Num + 4 * 2, 1, 1 * Num + 4 * 2, LabelF);

	//BitmapData *BD1 = new BitmapData();
	const int w = 60, h = 60;
	HDC WDC = GetDC(GetConsoleWindow());
	HDC hdc = CreateCompatibleDC(WDC);
	

	//Rect R = { 0,0,28,28 };
	WCHAR Name[200];
	for (int k = 0; k < Num; k++) {
		HBITMAP BM = CreateCompatibleBitmap(WDC, w, h);
		SelectObject(hdc, BM);
		//offset
		int cx = (rand() % 41) - 20 + 16, cy = (rand() % 41) - 20 + 16;
		for (int i = 0; i < 28; i++)
			for (int j = 0; j < 28; j++)
				if (j + cx < 0 || i + cy < 0 || j + cx >= 60 || i + cy >= 60)continue;
				else SetPixel(hdc, j+cx, i+cy, RGB(Data[i * 28 + j + 16 + 28 * 28 * k], Data[i * 28 + j + 16 + 28 * 28 * k], Data[i * 28 + j + 16 + 28 * 28 * k]));

		bitmap = new Bitmap(BM, NULL);
		wsprintfW(Name, L"%s%d%d-%d.jpg", OutPath, k,rand(), (int)LabData[4 * 2 + k]);
		GdiBitmapSave(L"jpg", Name);
		DeleteObject(BM);
		delete bitmap;
	}
	delete[]Data;
	delete[]LabData;
}

struct Pic:Param
{
	Pic():Param(){}
	Pic(POINT poi[], int Num, int ans,int WH) {
		_DataIn->Reset(WH*WH);
		for (int i = 0; i < Num; i++) {
			assert(poi[i].x * WH + poi[i].y < WH * WH);
			_DataIn->Set(poi[i].x * WH + poi[i].y);
		}_DataIn->Count += _DataIn->MaxCount;
		_DataOut->Reset(10);
		_DataOut->SetValue(ans);
	}
	void Decode(int&ans) {
		ans = _DataOut->max_value_id();
	}
};

struct LSTM_Pic :LSTM_Param {
	LSTM_Pic() {}
	LSTM_Pic(POINT poi[], int Num, int ans, int WH) :LSTM_Param(1) {
		param[0] = Pic(poi, Num, ans, WH);
	}
};
bool MNIST_Train_Func(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col) {
	if (_step == 0)
		Train_Test(agent->InPut_Net.front()->GetOutPut(), agent->All_In[0], RandomGenerator, test_start_col);
	if (_step + 1 == agent->RNN_Max_Steps)
		Train_Test(agent->OutPut_Net.front()->GetOutPut(), agent->All_Out[0], RandomGenerator, test_start_col);
	else agent->OutPut_Net.front()->GetOutPut()._ZeroMemory();
	//Gradient mask
	for (int j = 0; j < agent->tot_Net_Num; j++)
		if (agent->Net_Priority[j]->Net_Flag&Net_Flag_Gradient_Mask) {
			if (_step + 1 == agent->RNN_Max_Steps)
				agent->Net_Priority[j]->Pre_Net_Head.back()->GetOutPut().f(Device_Func::Assignment, 1.0);
			//getmask
			else agent->Net_Priority[j]->Pre_Net_Head.back()->GetOutPut()._ZeroMemory();
		}
	return true;
}


using namespace _Net_;
using namespace Base_Net;


struct Test :Param
{
	Test() :Param() {}
	Test(int a) {
		_DataIn->Reset(3);
		double val[3] = { 0.5,0.3,-0.2 };
		for (int i = 0; i < 3; i++) {
			_DataIn->SetValue(1, i, val[i],false);
		}_DataIn->Count += _DataIn->MaxCount;
		_DataOut->Reset(1);
		_DataOut->SetValue(1, 0, 1, false);
	}
	void Decode(int&ans) {
		ans = _DataOut->max_value_id();
	}
};

extern void findDirPic(string TrainPath, int& Num, int shift = 0);
//extern struct Pixel {
//	//0-1==>0,255
//	float r, g, b;
//};
extern Pixel* pix[2500][10];
extern char Name[2500][100];
extern int dx, dy;
extern int SampleNum;
//extern int MaxW, MaxH;
struct CatPic :Param {
	CatPic() :Param() {}
	CatPic(int WH,Pixel*pel,int ans) {
		_DataIn->Reset(WH * WH * 3);
		for(int i=0;i<WH;i++)
			for (int j = 0; j < WH; j++) {
				(*_DataIn)[i * WH + j] = pel[i * WH + j].r;
				(*_DataIn)[i * WH + j+1] = pel[i * WH + j].g;
				(*_DataIn)[i * WH + j+2] = pel[i * WH + j].b;
			}
		_DataIn->Count += _DataIn->MaxCount;
		_DataOut->Reset(12);
		_DataOut->SetValue(ans);
	}
};
map<string, int>output;
void main()
{
	srand(time(0));

	//freopen("Test_Data", "w", stdout);

	////homework
	//Net*net = (new Net(3, InPut));
	//net = (new Net(2, _Net::relu))->Add_Forward(Transform, net);
	//net = (new Net(2, _Net::tanh))->Add_Forward(Transform, net);
	//net = (new Net(1, _Net::sigmoid))->Add_Forward(Transform, net);
	//net = (new Net(1, BinaryCrossEntropy))->Add_Forward(OutPut, net);
	//Agent*ag = new Agent(Agent::Full_Connected, 1, net, -1);
	//Test p = Test(2);
	//ag->Data_Assignment();

	//ag->Train(&p, 1, 1, "");

	//return;
	





	/*Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
	for (int i = 0; i < 1; i++) {
		GererateNumberPic("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1, L"TranslatedPic/");
		GererateNumberPic("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10000, L"TranslatedTest/");
	}
	Gdiplus::GdiplusShutdown(gdiplusToken);
	return;*/
	
	/*const int L1 = 3;
	int*S1 = new int[L1] {MX_Word_Len, 128, (int)Q.size()};
	_Net::Activation_Func*_Func = new _Net::Activation_Func[L1 - 2]{ _Net::null };
	G_Net*Words = new G_Net(L1, S1, 100, _Net::softmax, 0,-1, _Func);
	Words->Cuda_Train(word, cnt1, cnt1*0.8, "Speed:0.001 Forget:0.9 Beta1:0.9 Beta2:0.999 Hidden_DropOut:0.5 InPut_DropOut:0.2","_Words");*/


	//int cnt = tot;
	//Ner*param = new Ner[cnt];
	//for (int i = 0; i < cnt; i++) {
	//	param[i] = Ner(Sen[i], Tag[i]);
	//}
	//static const int L = 3;
	//int*S = new int[L] {MX_Word_Len, 256, tag_Size};
	//RNN_LSTM*AI = new RNN_LSTM(S, 5, 32, _Net::softmax,0,Tags["O"]);

	////min batch size

	//AI->Train_LSTM(param, cnt, Train_Cnt, 0.001, 0.9, 0.9, 0.999);
	//srand(time(0));
	//int cnt = 32 * 200 + 32 * 300;
	//LSTM_Param*param = new LSTM_Param[cnt];
	//for (int i = 0; i < cnt; i++) {
	//	/*int v = 128;
	//	double In[3];
	//	for (int j = 0; j < 3; j++)
	//		In[j] = (rand() % v) / 128.0;*/
	//	param[i] = LSTM_Param(30, 3, 4);
	//	//param[i] = Pic(TExample[i], TENum[i], Ou[i]);
	//}

	///*static const int L = 3;
	//int*S = new int[L] {(int)param[0].param[0]._DataIn->Count, 128, (int)param[0].param[2]._DataOut->Count};
	//RNN_LSTM*AI = new RNN_LSTM(S, param[0].Count, 100,_Net::softmax);
	//AI->Train_LSTM(param, cnt, 32*200, "Speed:0.01");*/
	////LSTM
	//ui In = param[0].param[0]._DataIn->MaxCount, Out = param[0].param[2]._DataOut->MaxCount, Hidden = 128;
	//Net*Net_In = new Net(In, InPut), *Net_Out = NULL;
	//Net_In = (new Net(In))->Add_Mask(Net_Flag_InPut_DropOut, Net_In);
	//RNN_LSTM::LSTM_Net(Net_In, Net_Out, Hidden);
	////OutPut Layer
	//Net*net = (new Net(Hidden))->Add_Mask(Net_Flag_OutPut_DropOut, Net_Out);
	//net = (new Net(Out, _Net::softmax))->Add_Forward(Transform, net);
	//net = (new Net(Out))->Add_Mask(Net_Flag_OutPut_Mask, net);
	//net = (new Net(Out))->Add_Forward(OutPut, net);
	//Agent*agent = new Agent(Agent::RNN, 128, net, param[0].Count);
	//agent->Train(param, cnt, 32*200, "Speed:0.01 InPut_DropOut:0 OutPut_DropOut:0.5");
	//return;

	/*int err = 0;
	for (int i = 0; i < 4000; i++) {
		int ans = -1;
		TestExample[i].Num_Decode(ans, *AI);
		if (ans != Ou[cnt + i])err++;
	}
	std::cout << 100.0*(4000 - err) / 4000 << '\n';*/

	HyperParamSearcher param;
	param.Read_Param();
	int Cnt = 0, Test = 0;
	//GetPic("TranslatedPic/", "TranslatedTest/", Cnt, Test);
	////conv 99.24% conv+pool+fc+dropout
	////GetPic("TrainPic/", "SS/", Cnt, Test);
	//Cnt += Test;
	//int Pic_WH = 28;
	//LSTM_Pic*P = new LSTM_Pic[Cnt];
	//int Glimpse = 6;
	//for (int i = 0; i < Cnt; i++) {
	//	P[i] = LSTM_Pic(TExample[i], TENum[i], Ou[i], Pic_WH);
	//}
	//
	//int cnt = 1, max_epoch = 3, EarlyStop = 1;
	//while (cnt-- > 0) {
	//	//Random Search param
	//	param.Random_Uniform("stdev", 20, 0.01);
	//	param.Random_Uniform_Step("Batch", 64, 8);
	//	param.Draw_Geometric("Speed", 0.0001, 0.1);
	//	param.Draw_Geometric("Speed_decay_time",5000,50000,true);
	//	param.Draw_Geometric("Reward_Baseline_Beta1", 0.001, 1,false,true);
	//	param.Draw_Geometric("Mean_Baseline_Beta1", 0.001, 1,false,true);
	//	//Start Train
	//	if (cnt <= 0) {
	//		param.Read_Param();
	//		param["Max_Epoch"] = 100000;
	//		param["EarlyStop"] = 0;
	//	}
	//	param["Max_Step"] = Glimpse;
	//	//FC
	//	//Net*net = new Net(Pic_WH*Pic_WH, InPut);
	//	////net = (new Net(Pic_WH*Pic_WH))->Add_Mask(Net_Flag_InPut_DropOut, net);
	//	//net = (new Net(256, relu))->Add_Forward(Transform, net);
	//	//net = (new Net(256))->Add_Mask(Net_Flag_Hidden_DropOut, net);
	//	//net = (new Net(256, relu))->Add_Forward(Transform, net);
	//	//net = (new Net(256))->Add_Mask(Net_Flag_Hidden_DropOut, net);
	//	//net = (new Net(10, softmax))->Add_Forward(Transform, net);
	//	//net = (new Net(10))->Add_Forward(OutPut, net);
	//	//ConvNet
		//Net*Image = (new Net(Pic_WH*Pic_WH, InPut));
		////Image = (new Net())->Add_Mask(Net_Flag_InPut_DropOut, Image);
		//Image = (new ConvNet(8, 5, 1, (5 - 1) / 2, param, null_Func))->Add_Forward(Image, 1, Pic_WH);
		//Image = (new Net(relu))->Add_Forward(BNTransform, Image);
		//Image = (new POOL(4, 2, 0, param))->Add_Forward(Image, 8, Pic_WH);
		//Image = (new Net(256, null_Func))->Add_Forward(Transform, Image);
		//Image = (new Net(relu))->Add_Forward(BNTransform, Image);
		////Image = (new Net())->Add_Mask(Net_Flag_Hidden_DropOut, Image);
		//Image = (new Net(10, softmax))->Add_Forward(Transform, Image);
		////Image = (new Net(softmax))->Add_Forward(BNTransform, Image);
		//Image = (new Net(10))->Add_Forward(OutPut, Image);

	//	Attention::Init_Attention(param, Image, Pic_WH, Pic_WH, 3, 12, 10, 256);
	//	Agent*agent = new Agent(Agent::RNN, param["Batch"], Image, Glimpse);
	//	agent->Online_Init(param);
	//	param.Compare(agent->Train(P, Cnt, Cnt - Test, "", MNIST_Train_Func));
	//	delete agent;
	//}

    GetPic("TrainPic/", "SS/", Cnt, Test);
	Pic**P = new Pic*[Cnt+Test];
	for (int i = 0; i < Cnt + Test; i++) {
		P[i] = new Pic(TExample[i], TENum[i], Ou[i], 28);
	}

	//findDirPic("cat_12_train/", Cnt);
	////printf("W:%d H: %d\n", MaxW, MaxH);

	//freopen("train_list.txt", "r", stdin);
	//char str[1000]; int num;
	//while (scanf("%*[^/]/%[^\t]\t%d", str, &num) == 2) {
	//	output[str] = num;
	//}
	//printf("out:%u\n", output.size());
	////fstream file;file.open("train_list.txt", ios::in);
	//Cnt *= SampleNum;
	//CatPic** P = new CatPic * [Cnt + Test];
	//for (int i = 0; i < Cnt + Test; i++) {
	//	assert(output.find(Name[i/SampleNum]) != output.end());
	//	P[i] = new CatPic(dx, pix[i / SampleNum][i % SampleNum], output[Name[i / SampleNum]]);//(TExample[i], TENum[i], Ou[i], 28);
	//}
	//Test = 100 * SampleNum;
	//Cnt -= Test;


	//Net*net = new Net(28 * 28, InPut);
	////net = (new Net(28 * 28, relu))->Add_Forward(BNTransform, net);
	////net = (new Net(28 * 28))->Add_Mask(Net_Flag_InPut_DropOut, net);
	//net = (new Net(100, null_Func))->Add_Forward(Transform, net);
	//net = (new Net(100, sigmoid))->Add_Forward(BNTransform, net);
	////net = (new Net(128))->Add_Mask(Net_Flag_Hidden_DropOut, net);
	//net = (new Net(100, null_Func))->Add_Forward(Transform, net);
	//net = (new Net(100, sigmoid))->Add_Forward(BNTransform, net);
	////net = (new Net(128))->Add_Mask(Net_Flag_Hidden_DropOut, net);
	//net = (new Net(100, null_Func))->Add_Forward(Transform, net);
	//net = (new Net(100, sigmoid))->Add_Forward(BNTransform, net);

	//net = (new Net(10, softmax))->Add_Forward(Transform, net);
	////net = (new Net(10, softmax))->Add_Forward(BNTransform, net);
	//net = (new Net(10))->Add_Forward(OutPut, net);
	//Agent*agent = new Agent(Agent::Full_Connected,100,net);
	//agent->Train(P, Cnt, 60000, "Speed:0.001");
	//return;
	int ImageWH = 28;
	int filters = 64;
	Net* Image = (new Net(ImageWH* ImageWH, InPut));
	//Image = (new Net())->Add_Forward(Scale, Image);
	//Image = (new Net())->Add_Mask(Net_Flag_InPut_DropOut, Image);
	Image = (new ConvNet(filters, 5, 1, (5 - 1) / 2, param, null_Func))->Add_Forward(Image, 1, ImageWH);
	Image = (new Net(relu))->Add_Forward(BNTransform, Image);
	for (int i = 0; i < 2; i++) {
		Net*ori_Image = Image;
		Image = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(Image, filters, ImageWH);
		Image = (new Net(relu))->Add_Forward(BNTransform, Image);
		Image = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(Image, filters, ImageWH);
		Image = (new Net(null_Func))->Add_Forward(BNTransform, Image);
		Net*Cell = (new Net(Image->Net_Node_Num))->Add_Pair_Forward(DotPlus, Image, ori_Image);
		Image = (new Net(Image->Net_Node_Num, relu))->Add_Forward(OpsType::function, Cell);
	}
	Image = (new ConvNet(1, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(Image, filters, ImageWH);
	Image = (new Net(relu))->Add_Forward(BNTransform, Image);
	//Image = (new POOL(2, 2, 0, param))->Add_Forward(Image, filters, ImageWH / 4 - 1);
	//Image = (new ConvNet(filters, 2, 2, 0, param, null_Func))->Add_Forward(Image, filters, ImageWH / 4 - 1);
	//Image = (new Net(relu))->Add_Forward(BNTransform, Image);
	//Image = (new Net(256, relu))->Add_Forward(Transform, Image);

	//Image = (new Net(256, relu))->Add_Forward(Transform, Image);
	//Image = (new Net())->Add_Mask(Net_Flag_Hidden_DropOut, Image);
	Image = (new Net(10, softmax))->Add_Forward(Transform, Image);
	Image = (new Net(10))->Add_Forward(OutPut, Image);

	//ConvNet
	/*Net*Image = (new Net(28 * 28, InPut));
	Image = (new ConvNet(8, 5, 1, (5 - 1) / 2, param, null_Func))->Add_Forward(Image, 1, 28);
	Image = (new Net(relu))->Add_Forward(BNTransform, Image);
	Image = (new ConvNet(8, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(Image, 8, 28);
	Image = (new Net(relu))->Add_Forward(BNTransform, Image);
	Image = (new ConvNet(8, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(Image, 8, 28);
	Image = (new Net(relu))->Add_Forward(BNTransform, Image);

	Image = (new Net(10, softmax))->Add_Forward(Transform, Image);
	Image = (new Net(10))->Add_Forward(OutPut, Image);*/

	//Agent*_agent = new Agent("BN_Net", param["Batch"], param["Max_Step"], -1, true);
	Agent*agent = new Agent(Agent::RNN, param["Batch"], Image, param["Max_Step"], true, time(0));
	////_agent->Online_Init(param);
	agent->Online_Init(param);
	////_agent->Data_Assignment(agent);
	agent->Train<Agent::Train_Option>((Base_Param**)P, Cnt + Test, Cnt, "");
	agent->Write_to_File("BN_Net");
	//Agent*agent2 = new Agent("SL_Net", param["Batch"], param["Max_Step"], -1, true);
	//_agent->Data_Test((Base_Param**)&P[Cnt], Test);
	return;



	//Pic Test
 //   Matrix_Set_Function(0, 0, 1);
	//const int sz = 1, D = 1;
	//for (int t = 0; t < sz; t++) {
	//	Pic&G = P[t];
	//	for (int i = 0; i < 28; i++) {
	//		for (int j = 0; j < 28; j++) {
	//			cout << (int)(*G._DataIn)[i * 28 + j];
	//		}
	//		cout << endl;
	//	}cout << endl;
	//}
	//double Mp[28 * 28 * sz * D];
	//for (int i = 0; i < sz; i++) {
	//	Pic&H = P[i];
	//	for (int j = 0; j < 28 * 28; j++) {
	//		Mp[(j * D) * sz + i] = (*H._DataIn)[j];
	//		//Mp[(j*D + 1)*sz + i] = 0;
	//	}
	//}
	//double image[] = {
	//	3,3,
	//	3,3,
	//	0,0,
	//	0,0,
	//	2,2,
	//	2,2,
	//	1,1,
	//	1,1,
	//	0,0,
	//	0,0,
	//	-1,-1,
	//	-1,-1,
	//	0,0,
	//	0,0,
	//	1,1,
	//	1,1,
	//	-3,-3,
	//	-3,-3,
	//	-4,-4,
	//	-4,-4,
	//	1,1,
	//	1,1,
	//	3,3,
	//	3,3,
	//	-2,-2,
	//	-2,-2,
	//	-5,-5,
	//	-5,-5,
	//	1,1,
	//	1,1,
	//	-4,-4,
	//	-4,-4
	//};
	//Mat Image(4 * 4 * D, sz);
	//Image._ZeroMemory();
	//int _W = 4, _H = 4, F = 3, Pad = 0, S = 1;
	//int W = (_W - F + 2 * Pad) / S + 1;
	//int H = (_H - F + 2 * Pad) / S + 1;
	//double data[] = {
	//	1,1,
	//	1,1,
	//	0.5,0.5,
	//	0.5,0.5,
	//	-2,-2,
	//	-2,-2,
	//	-1.5,-1.5,
	//	-1.5,-1.5,
	//	/*0.5,
	//	0,
	//	1,
	//	2,
	//	1*/
	//}, data1[] = {
	//	5*D,5 * D,
	//	5*D+1,5 * D + 1,
	//	5*D,5 * D,
	//	5*D+1,5 * D + 1,
	//	5*D,5 * D,
	//	5*D+1,5 * D + 1,
	//	10*D,10 * D,
	//	10*D+1,10 * D + 1,
	//	/*10,
	//	11,
	//	12,
	//	14,
	//	15*/
	//};
	//Mat Gradient(W*H*D, sz, data);
	//Mat Pool_Idx(W*H*D, sz,data1);
	////Mat Img = Image.Conv_im2col(D, _W, W, H, F, Pad, S, &Gradient);
	//Mat Img = Image.Image_Pooling(D, _W, W, H, F, Pad, S, Pool_Idx, &Gradient);

	//double*M = Image.ReadFromDevice();
	//double*M1 = Pool_Idx.ReadFromDevice();
	//for (int t = 0; t < sz; t++) {
	//	for (int i = 0; i < _H; i++) {
	//		for (int j = 0; j < _W; j++) {
	//			for (int k = 0; k < D; k++)
	//				std::cout << M[(D*_W*i + D * j + k)*sz + t] << ' ';// << M1[(D*W*i + D * j + k)*sz + t] << ' ';
	//		}cout << endl;
	//	}cout << endl;
	//	cout << endl;
	//} Matrix_Set_Function(0, 0, 1);
	//const int sz = 2, D = 2;
	////for (int t = 0; t < sz; t++) {
	////	Pic&G = P[t];
	////	for (int i = 0; i < 28; i++) {
	////		for (int j = 0; j < 28; j++) {
	////			cout << (int)(*G._DataIn)[i * 28 + j];
	////		}
	////		cout << endl;
	////	}cout << endl;
	////}
	////double Mp[28 * 28 * sz * D];
	////for (int i = 0; i < sz; i++) {
	////	Pic&H = P[i];
	////	for (int j = 0; j < 28 * 28; j++) {
	////		Mp[(j * D) * sz + i] = (*H._DataIn)[j];
	////		//Mp[(j*D + 1)*sz + i] = 0;
	////	}
	////}
	//double image[] = {
	//	3,3,
	//	3,3,
	//	0,0,
	//	0,0,
	//	2,2,
	//	2,2,
	//	1,1,
	//	1,1,
	//	0,0,
	//	0,0,
	//	-1,-1,
	//	-1,-1,
	//	0,0,
	//	0,0,
	//	1,1,
	//	1,1,
	//	-3,-3,
	//	-3,-3,
	//	-4,-4,
	//	-4,-4,
	//	1,1,
	//	1,1,
	//	3,3,
	//	3,3,
	//	-2,-2,
	//	-2,-2,
	//	-5,-5,
	//	-5,-5,
	//	1,1,
	//	1,1,
	//	-4,-4,
	//	-4,-4
	//};
	//Mat Image(4 * 4 * D, sz);
	//Image._ZeroMemory();
	//int _W = 4, _H = 4, F = 3, Pad = 0, S = 1;
	//int W = (_W - F + 2 * Pad) / S + 1;
	//int H = (_H - F + 2 * Pad) / S + 1;
	//double data[] = {
	//	1,1,
	//	1,1,
	//	0.5,0.5,
	//	0.5,0.5,
	//	-2,-2,
	//	-2,-2,
	//	-1.5,-1.5,
	//	-1.5,-1.5,
	//	/*0.5,
	//	0,
	//	1,
	//	2,
	//	1*/
	//}, data1[] = {
	//	5*D,5 * D,
	//	5*D+1,5 * D + 1,
	//	5*D,5 * D,
	//	5*D+1,5 * D + 1,
	//	5*D,5 * D,
	//	5*D+1,5 * D + 1,
	//	10*D,10 * D,
	//	10*D+1,10 * D + 1,
	//	/*10,
	//	11,
	//	12,
	//	14,
	//	15*/
	//};
	//Mat Gradient(W*H*D, sz, data);
	//Mat Pool_Idx(W*H*D, sz,data1);
	////Mat Img = Image.Conv_im2col(D, _W, W, H, F, Pad, S, &Gradient);
	//Mat Img = Image.Image_Pooling(D, _W, W, H, F, Pad, S, Pool_Idx, &Gradient);

	//double*M = Image.ReadFromDevice();
	//double*M1 = Pool_Idx.ReadFromDevice();
	//for (int t = 0; t < sz; t++) {
	//	for (int i = 0; i < _H; i++) {
	//		for (int j = 0; j < _W; j++) {
	//			for (int k = 0; k < D; k++)
	//				std::cout << M[(D*_W*i + D * j + k)*sz + t] << ' ';// << M1[(D*W*i + D * j + k)*sz + t] << ' ';
	//		}cout << endl;
	//	}cout << endl;
	//	cout << endl;
	//}
 //   Mat Image(28 * 28 * D, sz, Mp);
	//double p[14] = { 14,14,14,14,14,14,14,14,14,14,14,14,14,14};
	//Mat Location(2, sz, p);
	//int Scale_Cnt = 5, WH = 8;
	//Mat Img = Image.ScaleImage(28, D, Location, WH, Scale_Cnt);
	//double*M = Img.ReadFromDevice();
	///*for (int t = 0; t < sz; t++) {
	//	for (int y = 0; y < H; y++)
	//		for (int x = 0; x < W; x++) {
	//			for (int i = 0; i < F; i++) {
	//				for (int j = 0; j < F; j++)
	//					for (int k = 0; k < D; k++) {
	//						std::cout << M[(i*F*D + D * j + k)*W*H*sz + t * W*H + y * W + x];
	//					}
	//				cout << endl;
	//			}printf("%d %d %d\n", x, y, t);
	//		}
	//}*/
	//for (int i = 0; i < Scale_Cnt; i++) {
	//	for (int j = 0; j < WH; j++) {
	//		for (int t = 0; t < sz; t++) {
	//			for (int k = 0; k < WH; k++) {
	//				for (int u = 0; u < D; u++)
	//					cout << M[sz*((i*WH*WH + j * WH + k)*D + u) + t];
	//			}
	//			cout << "  ";
	//		}
	//		cout << endl;
	//	}cout << endl;
	//}

	//return;





	//int S[] = { (int)P[0]._DataIn->Count, 128,128,(int)P[0]._DataOut->Count };
	////G_Net*AI = new G_Net("_Test",10, _Net::sigmoid);
	//G_Net*AI=new G_Net(sizeof(S)/sizeof(int), S, 100, _Net::softmax);
	//AI->Cuda_Train(P, Cnt, 60000,"Speed:0.001 InPut_DropOut:0 Hidden_DropOut:0","_Test");
	//Net AI1("Data");
	////if (*AI == AI1)std::cout << "Ok\n";
	//int cnt = 0, err = 0;
	//int ins = cnt;
	//int Max = 100; int In[3];
	//for(int i=0;i<Max;i++)
	//	for(int j=0;j<Max;j++)
	//		for (int k = 0; k < Max; k++) {
	//			for (int g = 0; g < 3; g++)
	//				In[g] = { (int)(1.0*rand() / 32767 * 10) };
	//			int ans1 = 0;
	//			//scanf(" %d%d%d", &In[0], &In[1], &In[2]);
	//			Param(In, 3, MaxV).f_DeCode(ans1, AI1);
	//			int ans = f(In);
	//			//std::cout << ans <<' '<<ans1<< '\n';
	//			if (ans != ans1)err++;
	//			cnt++;
	//		}
	//std::cout << 100.0*(cnt-err) / cnt << '\n';


	//getchar();

}