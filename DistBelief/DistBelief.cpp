#define SERVER_EXPORTS

#include"dist.h"




struct Worker {
	int id, tot_Worker_Num;
	Agent*Server, *shardServer;

	Agent*worker;
	mutex mux;
	static const int Fetch_Step = 10;
	static const int Push_Step = 1;
	Worker(int id,int tot_Worker_Num, Agent*Server) :id(id), tot_Worker_Num(tot_Worker_Num), Server(Server), shardServer(shardServer) {
		//worker = new Agent(Agent::RNN, param["Batch"], net, param["Max_Step"], false, time(0), param["Max_srand_row"]);
		//worker->Online_Init(param);
	}
	~Worker() {
		//delete worker;
	}
	enum State {
		FETCH, PUSH, FINISH, READY,Update
	};
	volatile State state;
	//get parameter
	void fetch() {
		/*{
			lock_guard<mutex>locker(mux);
			state = FETCH;
		}*/
		//while (state != FETCH)this_thread::yield();
		Server->Data_Assignment(worker);
	}
	//push gradient
	void push() {
		{
			lock_guard<mutex>locker(mux);
			state = PUSH;
		}
		while (state != Update)this_thread::yield();
		//Update server parameter
		Server->getGradient(shardServer, 1.0 / tot_Worker_Num);
		//sync parameter
		StreamSynchronize();
		state = READY;
	}
	template<typename Train_Func, typename NetFunc>
	void run(NetFunc Net, HyperParamSearcher&param,int trainNum, Train_Func train_func, ui BatchsPerEpoch) {
		shardServer = new Agent(Agent::RNN, 1, Net(param), 1, false, time(0), param["Max_srand_row"]);
		worker = new Agent(Agent::RNN, param["Batch"], Net(param), param["Max_Step"], false, time(0), param["Max_srand_row"]);
		worker->Online_Init(param);
		train(trainNum, train_func, BatchsPerEpoch);
		lock_guard<mutex>locker(mux);
		state = FINISH;

		delete shardServer;
		delete worker;
		unBindStm();
	}
	template<typename Train_Func>
	void train(int trainNum, Train_Func train_func, ui BatchCountPerEpoch, bool Cal_BN = false) {
		int Batch = worker->Net_Param["Batch"];
		int _trainNum = trainNum / Batch * Batch, _testNum = (Server->All_In[0].GetCol() - _trainNum) / Batch * Batch;
		Agent::Learning_rate Speeder(worker->Net_Param["Speed"], worker->Net_Param["Speed_decay_time"]);
		Mat RandomGenerator(1, Batch);
		//ui BatchCountPerEpoch = std::max(_trainNum / Batch, 1);
		ui BatchTrainCount = 0; state = FETCH;
		clock_t Start = clock();
		bool Stochastic = worker->Net_Param["Stochastic"];
		double tot_Loss = 0, Last_Loss = 0, min_Loss = 1e18;
		for (int t = 0; t < worker->Net_Param["Max_Epoch"]; t++) {
			while (true) {
				if (BatchTrainCount%Fetch_Step == 0 && !Cal_BN)
					fetch();
				tot_Loss = 0;
				worker->RNN_Init();
				worker->lstm_Step = -1;
				double Loss = 0.0, L2_Loss = 0;
				//train example random order
				if (Stochastic) {
					RandomGenerator.GenerateRandom(_trainNum);
				}
				for (int i = 0; i < worker->RNN_Max_Steps; i++) {
					if (!train_func(worker, &RandomGenerator, i, Stochastic ? -1 : (BatchTrainCount*Batch), Server))
						break;
					//Step DropOut
					//DropOut(true);
					//Feed forward&Cal Loss
					Loss += worker->Forward((BatchTrainCount%BatchCountPerEpoch + _Net_::G_Net::TrainUnit::BN_avg_miniBatch_Num * 2) >= BatchCountPerEpoch ? 2 : true, i, true, true, &L2_Loss);
					//Reinforce_Reward(i);
					assert(!isnan(Loss));
					worker->lstm_Step++;
				}
				if (worker->lstm_Step > -1&& !Cal_BN) {
					//Backward
					for (int i = worker->lstm_Step; i > -1; i--) {
						//save memory
						/*if (!train_func(worker, &RandomGenerator, i, Stochastic ? -1 : (BatchTrainCount * Batch), Server))
							break;
						trans2LSTM_Step(i);*/
						worker->Backward(i, worker->lstm_Step + 1);
					}
					//Update Weigh
					//if (!Net_Param["EpochUpdate"] || (BatchTrainCount + 1) % BatchCountPerEpoch == 0)
					worker->Update(shardServer);
					//push Gradient
					if (BatchTrainCount%Push_Step == 0)
						push();

					//Log OutPut
					printf("id:%d Loss:%f L2_Loss:%.02lf Count:%d epoch:%d Speed:%f Time:%d min(s)\n", id, Loss, L2_Loss, BatchTrainCount, BatchTrainCount / BatchCountPerEpoch, worker->Net_Param["Speed"], (clock() - Start) / 1000 / 60);
					tot_Loss += Loss;
				}
				//Speed decay
				Speeder.Step_Decay(++BatchTrainCount / BatchCountPerEpoch, worker->Net_Param["Speed"]);
				if (BatchTrainCount%BatchCountPerEpoch == 0)
					break;
			}
		}
	}
	void trans2LSTM_Step(int step) {
		for (auto& k : worker->InPut_Net) {
			k->GetOutPut(step) = move(k->GetOutPut((step + 1) % worker->RNN_Max_Steps));
			k->GetOutPut(step).Reset(k->GetOutPut().GetRow(), k->GetOutPut().GetCol());
			k->GetOutPut(step) = move(k->GetOutPut());
		}
		for (auto& k : worker->OutPut_Net) {
			k->GetOutPut(step) = move(k->GetOutPut((step + 1) % worker->RNN_Max_Steps));
			k->GetOutPut(step).Reset(k->GetOutPut().GetRow(), k->GetOutPut().GetCol());
			k->GetOutPut(step) = move(k->GetOutPut());
		}
	}
};

bool Default_Train_Func(Agent*agent, Mat*RandomGenerator, int _step, int test_start_col, Agent*Server) {
	Train_Test(agent->InPut_Net.front()->GetOutPut(), Server->All_In[_step], RandomGenerator, test_start_col);
	Train_Test(agent->OutPut_Net.front()->GetOutPut(), Server->All_Out[_step], RandomGenerator, test_start_col);
	//for (int j = 0; j < agent->tot_Net_Num; j++)
	//	//Gradient mask
	//	if (agent->Net_Priority[j]->Net_Flag&Net_Flag_Gradient_Mask) {
	//		if (_step + 1 == agent->RNN_Max_Steps) {
	//			agent->Net_Priority[j]->Pre_Net_Head.back()->GetOutPut().f(Device_Func::Assignment, 1.0);
	//		}
	//		else agent->Net_Priority[j]->Pre_Net_Head.back()->GetOutPut()._ZeroMemory();
	//	}
	return true;
}
struct ParamServer {
	Agent*agent;// , **shardServer;
	Worker**workers;
	int workers_Num, paramNum, trainNum;
	Base_Param**train_DataSet;
	static const int Max_Thr = 128;
	thread*thr[Max_Thr];
	HyperParamSearcher param;
	//template<typename NetFunc>
	void Init() {
		workers = new Worker*[workers_Num] { 0 };
		//shardServer = new Agent*[workers_Num] { 0 };
		for (int i = 0; i < workers_Num; i++) {
			//shardServer[i] = new Agent(Agent::RNN, 1, Net(param), 1, false, time(0), param["Max_srand_row"]);
			workers[i] = new Worker(i, workers_Num, agent);
		}
	}
	ParamServer(int workers_Num, Agent*InitAgent, HyperParamSearcher&param) :workers_Num(workers_Num) {
		this->param = param;
		agent = InitAgent;
		agent->Online_Init(param);
		this->param._param = agent->Net_Param;
		Init();
	}
	ParamServer(int workers_Num, NetFunc Net, string Param_Path) :workers_Num(workers_Num) {
		param.SetPath(Param_Path, Param_Path);
		param.Read_Param();
		agent = new Agent(Agent::RNN, param["Batch"], Net(param), param["Max_Step"], true, time(0), param["Max_srand_row"]);
		agent->Online_Init(param);
		Init();
	}
	~ParamServer() {
		//delete agent;
		for (int i = 0; i < workers_Num; i++)delete workers[i];// , delete shardServer[i];
		delete[] workers;
	}
	void InitializeData(Base_Param**param, int paramNum, int trainNum, Agent::DataWrite2Device data_proc_func = Agent::Default_Data2Device) {
		data_proc_func(param, paramNum, agent, false);
		this->paramNum = paramNum;
		this->trainNum = trainNum;
		train_DataSet = param;
	}
	template<typename Train_Func, typename NetFunc>
	void run(Train_Func Func, NetFunc Net) {
		for (int i = 0; i < workers_Num; i++)
			thr[i] = new thread(&Worker::run<Train_Func, NetFunc>, workers[i], Net, ref(param), trainNum, Func, trainNum / workers_Num / param["Batch"]);
		while (true) {
			int finish = 0;
			for (int i = 0; i < workers_Num; i++) {
				//send lastest parameter
				if (workers[i]->state == Worker::FETCH) {
					/*lock_guard<mutex>locker(workers[i]->mux);
					workers[i]->state = Worker::READY;*/
				}
				else if (workers[i]->state == Worker::PUSH) {
					{
						lock_guard<mutex>locker(workers[i]->mux);
						workers[i]->state = Worker::Update;
					}
					while (workers[i]->state == Worker::Update)this_thread::sleep_for(std::chrono::milliseconds(1));
				}
				else if (workers[i]->state == Worker::FINISH) {
					finish++;
				}
			}
			if (finish == workers_Num)break;
		}
		for (int i = 0; i < workers_Num; i++) {
			thr[i]->join();delete thr[i];
		}
		//get BN parameter 
		workers[0]->worker = agent;
		workers[0]->worker->Net_Param["Max_Epoch"] = 1;
		workers[0]->train(trainNum, Func, _Net_::G_Net::TrainUnit::BN_avg_miniBatch_Num * 2, true);
		//workers[0]->worker->Data_Assignment(agent);
		//agent->Write_to_File("SL_Net");
	}
	template<typename Train_Func>
	void DataTest(int trainNum,int testNum, Train_Func train_func) {
		agent->Test(testNum, 0, trainNum, testNum / param["Batch"], train_func);
	}

};
ParamServer*Server;
void Server_Init(int workers_Num, NetFunc Net, string Param_Path) {
	Server = new ParamServer(workers_Num, Net, Param_Path);
}
void Server_Init(int workers_Num, Agent*InitAgent, HyperParamSearcher&param) {
	Server = new ParamServer(workers_Num, InitAgent, param);
}
void Server_InitData(Base_Param**param, int paramNum, int trainNum) {
	Server->InitializeData(param, paramNum, trainNum);
}
template<typename Train_Func, typename NetFunc>
void Server_StartTrain(NetFunc NetFunc, Train_Func train_func) {
	Server->run(train_func, NetFunc);
}
template<typename Train_Func>
void Server_Test(int trainNum, int testNum, Train_Func train_func) {
	Server->DataTest(trainNum, testNum, train_func);
}
void Server_Disponse(bool freeAgent) {
	if (freeAgent)delete Server->agent;
	delete Server;
}


