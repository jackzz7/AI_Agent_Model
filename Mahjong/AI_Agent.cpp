#include"AI_Agent.h"

namespace AI_Agent {
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
}
using namespace AI_Agent;

void Sampling_DataSet(DataSet& train_ds, int data_Count, int recent_Max_data, int dataSet_Max_ID) {
	train_ds.trainSet_Init(data_Count);
	mt19937 rng(time(0));
	vector<int>idx;
	for (int i = 0; i < data_Count; i++)idx.push_back(rng() % recent_Max_data);
	sort(idx.begin(), idx.end(), greater<int>());
	int shift = 0; DataSet ds;
	for (auto i : idx) {
		int Start = ds.trainSet_dataCount() - (recent_Max_data - i - shift);
		while (Start < 0) {
			shift += ds.dataCount;
			assert(dataSet_Max_ID >= 0);
			char Path[100]; sprintf(Path, "extend_trainSet_data_%d", dataSet_Max_ID);
			/*const char* data = ReadFileFromZip("train_1.zip", Path);
			ds.Read_From_Memory(data);
			delete[]data;*/
			ds.trainSet_Save_Load(false, -1, Path);
			Start += ds.dataCount;
			dataSet_Max_ID--;
		}
		for (int j = 0; j < 1/*ds.trainSet_Param(Start).Count*/; j++) {
			train_ds.trainSet_Add_data(LSTM_Param(1));
			train_ds.trainSet_Param(train_ds.gameCount).In(j) = ds.trainSet_Param(Start).In(j);
			train_ds.trainSet_Param(train_ds.gameCount).In(j)[0] = j + train_ds.gameCount;
			train_ds.trainSet_Param(train_ds.dataCount - 1).Out(0) = ds.trainSet_Param(ds.trainSet_Param(Start).In(j)[0]).Out(0);
		}
		train_ds.gameCount++;
	}
	tot_data_set_Count = train_ds.trainSet_dataCount();
	tot_train_set_Count = train_ds.gameCount;
	//train_ds.trainSet_Save_Load(true, -1, "trainSet_data_sampling");
}
mutex train_mux, data_mux;
DataSet* sampling_ptr;
condition_variable train_CV, data_CV;

const int One_Patch_dataSet_Size = 200000;
void async_Sampling(int TrainDataNum, int Maximum_DataSet_Number) {
	std::unique_lock<std::mutex> data_locker(data_mux);
	while (true) {
		Sampling_DataSet(*sampling_ptr, One_Patch_dataSet_Size, TrainDataNum, Maximum_DataSet_Number);
		{
			lock_guard<mutex>locker(train_mux);
			printf("swap dataset\n");
			train_CV.notify_one();
		}
		data_CV.wait(data_locker);
	}
}

void Riichi_Agent::train(const char* ds_path, int epoches, int Maximum_DataSet_Number) {
	//sample dataSet
	int train_cnt = 0;
	double ori_Speed = sim->agent->Net_Param["Speed"];
	DataSet train_ds;// , train_ds_swap;
	std::unique_lock<std::mutex> train_locker(train_mux);
	sampling_ptr = &train_ds;
	int TrainDataNum = sim->param["paramNum"];
	new thread(&async_Sampling, TrainDataNum, Maximum_DataSet_Number);


	//tot_data_set_Count = ds.dataCount;
	//tot_train_set_Count = 0.8 * tot_data_set_Count;

	for (int i = 0; i < epoches; i++) {
		{
			sim->agent->Net_Param["Speed"] = max(ori_Speed * pow(0.1, floor(train_cnt / sim->agent->Net_Param["Speed_decay_time"])), 2e-5);
			int cnt = TrainDataNum / One_Patch_dataSet_Size;
			while (cnt-- > 0) {
				train_CV.wait(train_locker);
				sampling_ptr->miniTrain_Start(sim->agent, &data_CV, train_Param, Train_In_Out_Process);
				sim->agent->Write_to_File("riichi_Net");
			}
		}

		char path[50]; sprintf(path, "riichi_Net_%d", train_cnt);
		sim->agent->Write_to_File(path);

		//control train time
		train_cnt++;
	}
}