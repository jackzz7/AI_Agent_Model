#include"RewardPredictor.h"

int tot_data_set_Count = 0, tot_train_set_Count = 0;
void train_Param(HyperParamSearcher& param) {
	param["paramNum"] = tot_data_set_Count;
	param["trainNum"] = tot_train_set_Count;
	param["ExtraTrainFactorPerEpoch"] = 1;
	param["ExtraTestFactorPerEpoch"] = 1;
	param["Stochastic"] = 1;
	param["EpochUpdate"] = 0;
	param["Max_Epoch"] = 1;
	param["EarlyStop"] = 0;
}
void RewardAgent::combine_ds(int idx, int ds_number) {
	auto res = DataSet_Combine(0, ds_number, [](int idx) {char path[100]; sprintf(path, "extend_trainSet_data_thr%d_reward", idx); return path; });
	if (res->dataCount) {
		char path[100]; sprintf(path, "extend_trainSet_data_reward_%d", idx);
		res->trainSet_Save_Load(true, -1, path);
		delete res;
	}
}
void RewardAgent::train(const char* ds_path, int epoches) {
	DataSet ds;
	ds.trainSet_Save_Load(false, -1, ds_path);
	tot_data_set_Count = ds.dataCount;
	tot_train_set_Count = 0.8 * tot_data_set_Count;
	for (int i = 0; i < epoches; i++) {
		ds.miniTrain_Start(sim->agent, NULL, train_Param, Train_In_Out_Process);
		sim->agent->Write_to_File("Reward_Net");
		//9873 2784(2745) 0.06
		//9921 2757 
		//10704 2756

		//train loss/test loss
		//46957/12944 0.3524 12epoches

		//test loss 40400 7epoches
	}
}

const static int Unroll_Step_Length = 1;
DataSet* Combine_MP_ds(DataSet* ds, DataSet* ds1) {
	DataSet* comb = ds;
	int offset = ds->dataCount;
	assert(ds->dataCount > 0 && ds1->dataCount > 0);
	//new dynamic DataSet size
	if (comb->MaxCount < ds->dataCount + ds1->dataCount) {
		comb = new DataSet();
		comb->trainSet_Init(ds->dataCount + ds1->dataCount);
		for (int i = 0; i < ds->dataCount; i++) {
			comb->trainSet_Add_data(LSTM_Param(Unroll_Step_Length));
			comb->trainSet_Param(comb->dataCount - 1) = ds->trainSet_Param(i);
		}
		comb->gameCount += ds->gameCount;
		delete ds;
	}
	for (int i = 0; i < ds1->dataCount; i++) {
		comb->trainSet_Add_data(LSTM_Param(Unroll_Step_Length));
		comb->trainSet_Param(comb->dataCount - 1) = ds1->trainSet_Param(i);
	}
	comb->gameCount += ds1->gameCount;
	return comb;
}


DataSet* DataSet_Combine(int l, int r, std::function<string(int)>ds_path_func) {
	int M = l + r >> 1;
	if (l >= r)return NULL;
	if (l + 1 == r) {
		DataSet* ds = new DataSet(); ds->trainSet_Save_Load(false, 1, ds_path_func(l).c_str());
		return ds;
	}
	auto* left = DataSet_Combine(l, M, ds_path_func);
	auto* right = DataSet_Combine(M, r, ds_path_func);
	if (!left) {
		assert(right); return right;
	}
	else {
		assert(left && right);
		assert(left->dataCount > 0);
		if (left->dataCount > 0 && right->dataCount > 0)
			left = Combine_MP_ds(left, right);
		delete right;
		return left;
	}
}