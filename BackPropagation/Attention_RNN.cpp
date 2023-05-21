#define Attention_EXPORTS

#include"Attention.h"
#include"ConvNet.h"

void Attention::Init_Attention(HyperParamSearcher&param, Net*Image, ui ImageW, ui ImageH, ui Scale_Num, ui Scale_Image_WH, ui Out_Num, ui Hidden) {
	ui Batch = param["Batch"], Max_Step = param["Max_Step"];
	Net*Glimpse_L = (new Net(128, relu));
	//Image,L
	Glimpse*Scale_Image = (new Glimpse(ImageW, Scale_Num, Scale_Image_WH, Batch, Max_Step));
	Net*Glimpse_Scale = (new Net(128, relu))->Add_Forward(Transform, Scale_Image);
	Net*Scale_L = (new Net(256))->Add_Pair_Forward(OpsType::Concatenate, Glimpse_Scale, Glimpse_L);
	Net*Glimpse_Net = (new Net(256, relu))->Add_Forward(Transform, Scale_L);
	ui Out = Out_Num;
	//LSTM
	Net*Hidden_State = RNN_LSTM::LSTM_Net(Glimpse_Net, Hidden);
	//Action OutPut Layer
	Net*net = (new Net(Out, softmax))->Add_Forward(Transform, Hidden_State);
	net = (new Net(Out))->Add_Mask(Net_Flag_Gradient_Mask, net);
	net = (new Net(Out,CostFunc::CrossEntropy))->Add_Forward(OutPut, net);
	//Location Policy Net
	/*net = (new Net(128, _Net::relu))->Add_Forward(Transform, Hidden_State);
	net = (new Net(64, _Net::relu))->Add_Forward(Transform, net);*/
	Net*Location_mean = (new Net(2, null_Func))->Add_Forward(Transform, Hidden_State);
	//Net*Location_stdev = (new Net(2, _Net::sigmoid))->Add_Forward(Transform, Hidden_State);
	Net*ReinforceLocation = (new REINFORCE(Gaussian,2, param, Net_Flag_Random_Data))->Add_Forward(Location_mean);
	//RNN Recycle
	Glimpse_L->Add_Forward(Transform, ReinforceLocation, Net_Flag_Reconnect);
	Scale_Image->Add_Pair_Forward(ReinforceLocation, Image, Net_Flag_Reconnect);
}
Net*Attention::Init_Glimpse(HyperParamSearcher&param, Net*LocationNet, Net*InPut_Image) {
	int Node = 128;
	Net*Glimpse_L = (new Net(Node, relu))->Add_Forward(Transform, LocationNet, Net_Flag_Reconnect);
	Net*Scale_Image = (new Glimpse(param["Screen_Width"], param["Scale_Num"], param["Scale_Width_Heighth"], param["Batch"], param["Max_Step"]))
		->Add_Pair_Forward(LocationNet, InPut_Image, Net_Flag_Reconnect);
	//conv
	Net*Glimpse_Scale = (new ConvNet(16, 3, 1, (3-1)/2, param))->Add_Forward(Scale_Image, 1, param["Scale_Width_Heighth"]);
	Glimpse_Scale = (new Net(Node, relu))->Add_Forward(Transform, Glimpse_Scale);
	//Net*Glimpse_Scale = (new Net(Node, relu))->Add_Forward(Transform, Scale_Image);
	Net*Scale_L = (new Net(Node*2))->Add_Pair_Forward(OpsType::Concatenate, Glimpse_Scale, Glimpse_L);
	Net*Glimpse_Net = (new Net(Node*2, relu))->Add_Forward(Transform, Scale_L);
	return Glimpse_Net;
}


   