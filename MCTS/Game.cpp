// Catch_Ball.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
#define MCTS_EXPORTS

#include"League.h"
#include"Gomoku.h"
#include"Go.h"
using namespace Base_Net;

#define XML_Error(s) return printf(s),false

bool move();
void Header_Proc(const char*key,const char*val);

char name[1000] = "", xml_data[1000] = "";
bool XML_Header_Parse() {
	char key[1000], value[1000];
	while (true) {
		key[0] = value[0] = '\0';
		if (scanf("%[a-z0-9A-Z]=\"%[^\"]\"", key, value) != 2) {
			break;
		}
		Header_Proc(key,value);
		scanf("%*[ ]");
	}
	return scanf("%[^>]", xml_data) != 1;
}
bool XML_Reader() {
	char _name[1000];
	//get <
	scanf("%*[^<]");
	if (getchar() != '<') {
		XML_Error("error:not find '<'\n");
	}
	//get name
	name[0] = '\0';
	if (scanf("%[^>^ ]", name) != 1) {
		printf("error:can't read 'name'\n");
		return false;
	}
	char ch = getchar();
	xml_data[0] = '\0';
	if (ch == ' ') {
		//get info
		if (XML_Header_Parse()) {
			//XML_Error("error:can't read 'xml_data'\n");
		}
		if (getchar() != '>') {
			XML_Error("error:not find '>'\n");
		}
	}
	else if (ch == '>') {}
	else XML_Error("error:unexpected name end'\n");

	//process xml_data if exist



	//name invalid
	if (strlen(name) <= 0) {
		XML_Error("error:name error\n");
	}
	// <! ignore
	if (name[0] == '!') {
		return true;
	}
	//xml version&XML Start Read
	else if (name[0] == '?') {
		XML_Reader();
		return true;
	}
	//return end name
	//otherwise new child
	if (name[0] == '/') {
		return true;
	}
	// '/>' end
	if (name[strlen(name) - 1] == '/' || (strlen(xml_data) > 0 && xml_data[strlen(xml_data) - 1] == '/')) {
		return true;
	}

	//process child
	if (strcmp(name, "game") == 0) {}
	else if (strcmp(name, "move") == 0) {
		if (!move()) {
			XML_Error("error:move error\n");
		}
	}
	else if (strcmp(name, "info") == 0) {}
	else {
		printf("not process '%s'\n", name);
	}
	strcpy(_name, name);
	do {
		//find end
		XML_Reader();
	} while (name[0] != '/');
	if (strcmp(_name, &name[1]) != 0) {
		XML_Error("error:end name not match start name\n");
	}
	name[0] = '\0';
	return true;
}
const ui MaxGames = 100000;
const ui MaxStep = 15 * 15;
int cnt = 0, tot = 0;
pi Game[MaxGames][MaxStep];
int GameStep[MaxGames];
int GameResult[MaxGames];
bool move() {
	char X[10], Y[10];
	int sz = 0;
	int step = 0;
	while ((sz = scanf("%[a-o]%[0-9]", X, Y)) != -1) {
		if (sz != 2)break;
		assert(0 <= X[0] - 'a'&&X[0] - 'a' <= 14);
		int y = atoi(Y);
		assert(1 <= y && y <= 15);
		Game[cnt][step++] = { X[0] - 'a',y - 1 };
		scanf("%*[^<^a-o]");
	}
	GameStep[cnt++] = step;
	//Max_Step = max(Max_Step, step);
	return true;
}
void Header_Proc(const char*key, const char*value) {
	if (strcmp(name, "game") == 0) {
		if (strcmp(key, "id") == 0) {
			int id = atoi(value);
			/*if (id != cnt + 1)
				printf("%d not found\n", id), cnt = id - 1;*/
		}
		else if (strcmp(key, "bresult") == 0) {
			double res = atof(value);
			GameResult[cnt] = (res - 0.5) * 2;
		}
	}
}

int Gomoku_Data_Parse(const char*Path) {
	freopen(Path, "r", stdin);
	XML_Reader();
	freopen("CON", "r", stdin);
	return cnt;
}
void Gomoku_User_Param(HyperParamSearcher&param) {
	param["Screen_Width"] = 15;
	param["Scale_Num"] = 1;
	param["Scale_Width_Heighth"] = 7;
	param["Max_Step"] = 1;

	param["paramNum"] = 2673103;// tot;// / 10;
	param["trainNum"] = param["paramNum"] / 5 * 4;
	param["ExtraTrainFactorPerEpoch"] = 0.5;
	param["ExtraTestFactorPerEpoch"] = 1;
}


Param**game_data[Cuda_Max_Stream];// = new GomokuData*[128]{ 0 };
void agentResponse(int flag, int idx, int rotation, int**position, int step, double*OutPut, double*Value, Agent*agent) {
	//int tid = get_stm_id;
	//int Batch = agent->Net_Param["Batch"];
	//if (flag == 0) {
	//	*game_data[tid][idx] = GomokuData(idx, W, H, position, step, W*H + 1, rotation);
	//}
	//else if (flag == 1) {
	//	agent->Online_New_Turn();
	//	agent->Online_Next_Step();
	//	Agent::Default_Data2Device((Base_Param**)game_data[tid], Batch, agent, true);
	//	Gomoku::MCTS_Gomoku_Rollout(agent, NULL, 0, 0);
	//	agent->Online_OutPut((Base_Param**)game_data[tid], false, GomokuData::encoder, false);
	//}
	//else if (flag == 2)game_data[tid][idx]->get_Policy_Value(agent->OutPut_Net, OutPut, *Value, rotation);
	////fill blank
	//else if (flag == 3) {
	//	*game_data[tid][idx] = GomokuData(W, H, W*H + 1);
	//}
	//else assert(false);
}
//void agentResponse(int flag, int idx, Mat* Hidden_State, const ui Action, double* OutPut, double* Value, Mat* Next_State, double* Reward, Agent* agent, Environment* e, Param** data){}
int Environment::HiddenStateSize;
void agentResponse(int flag, int idx, Mat* Hidden_State, const ui Action, double* OutPut, double* Value, Mat* Next_State, double* Reward, int step, Agent* agent, Environment* e, Param** data) {
	//int tid = get_stm_id;
	int Batch = agent->Net_Param["Batch"];
	//InPut&&fill blank
	if (flag == 0 || flag == 3) {
		//dynamic function
		if (Hidden_State)
			e->DynamicEncode(agent->InPut_Net, data[idx], idx, Hidden_State, Action, flag == 3);
		//represent function
		else if (Action == -1)
			e->RepresentEncode(data[idx], flag == 3);
		//prediction function
		//else if (Action == -2)
			//e->PredictionEncode(data[idx], Next_State, flag == 3);
		else assert(false);
		//*game_data[tid][idx] = GomokuData(idx, W, H, position, step, W*H + 1, rotation);
	}
	else if (flag == 1) {
		agent->Online_New_Turn();
		agent->Online_Next_Step();
		Agent::Default_Data2Device((Base_Param**)data, Batch, agent, true);
		if (Hidden_State)
			e->DynamicDecode(agent, step);
		else if (Action == -1)
			e->RepresentDecode(agent);
		else assert(false);
		//else Gomoku::MCTS_Direct(agent, NULL, 0, 0);
		agent->Online_OutPut((Base_Param**)data, false, Environment::encoder, false);
	}
	else if (flag == 2) {
		//dynamic function
		if (Hidden_State)
			e->Get_NextState_Reward_And_Policy_Value(agent->OutPut_Net, data[idx], idx, Next_State, Reward, OutPut, Value);
		//represent function
		else if (Action == -1)
			e->Get_Initial_State_And_Policy_Value(agent->OutPut_Net, data[idx], idx, Next_State, OutPut, Value);
		//prediction function
		//else if (Action == -2)
			//e->Get_Policy_And_Value(agent->OutPut_Net, data[idx], OutPut, Value);
		else assert(false);
	}
	//End operator
	else if (flag == 4) {
		e->End_Reset(agent);
	}
	else assert(false);
}
int GomokuJudger(int**Position, int step, const pi&InsStone) {
	//win
	if (Game_Judger(Position, step % 2 == 1, { InsStone.X,InsStone.Y }))
		return 1;
	//draw
	else if (step >= 200)return 2;
	else return 0;
}
double Sum = 0;
void sleepy() {
	std::chrono::milliseconds dura(20000);
	this_thread::sleep_for(dura);
	for (int j = 0; j < 100000000; j++) {
		Sum += j;
	}
}

condition_variable cv;
bool flag = false;
mutex m;
void _sleepy() {
	std::unique_lock<std::mutex> _lock(m);
	std::chrono::milliseconds dura(1000);
	this_thread::sleep_for(dura);
	cv.wait(_lock);// , []() { return flag; });
	printf("Wake\n");
	//std::chrono::milliseconds dura(6000);
	this_thread::sleep_for(dura);
}
#include <random>
#include <iomanip>
#include<Windows.h>
int main()
{
	/*::ShowWindow(::GetConsoleWindow(), SW_HIDE);
	while (!(_kbhit() && _getch() == 'p')) {
		this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	cout << "Show\n";
	this_thread::sleep_for(std::chrono::milliseconds(20000));
	return 0;*/
	//map<int, double>Q;
	//if (Q.find(1) != Q.end() && Q[1] == 0) { cout << "asdf\n"; }
	//mutex m; std::unique_lock<std::mutex> lock(m);
	//cv.wait_for(lock, std::chrono::milliseconds(1000), [edge]() { return edge->Ready(); });
	//std::random_device rd;
	//std::mt19937 gen(rd());

	//// A gamma distribution with alpha=1, and beta=2
	//// approximates an exponential distribution.
	//std::gamma_distribution<double> d(0.02, 1);

	//double hist[100]; double sum = 0;
	//for (int n = 0; n < 100; ++n) {
	//	sum += hist[n] = d(gen);
	//}
	//for (int n = 0; n < 100; ++n) {
	//	hist[n] /= sum;
	//}
	/*list<int>Q, Q1;
	for (int i = 0; i < 5; i++)Q.push_back(i), Q1.push_back(i);
	auto it = Q.begin(); advance(it, 3);
	Q1.splice(Q1.end(), Q, Q.begin(), it);*/
	//for (int i = 0; i < 128; i++) {
	//std::unique_lock<std::mutex> _lock(m);
	//new thread(_sleepy);
	////_lock.unlock();
	////cv.notify_one();
	//
	//	//std::unique_lock<std::mutex> _lock(m);
	//	this_thread::sleep_for(std::chrono::milliseconds(2000));
	//	//std::unique_lock<std::mutex> _lock(m);
	//	//_lock.lock();
	//	//std::lock_guard<std::mutex> lock(m);
	//	//cv.notify_one();
	//	
	//	{
	//		//std::lock_guard<std::mutex> lock(m);
	//	}
	//	std::chrono::milliseconds dura(1000);
	//	this_thread::sleep_for(dura);
	//	
	//	
	//	flag = true;
	//	cv.notify_one();
	//	this_thread::sleep_for(dura);
	//	this_thread::sleep_for(dura);
		//std::unique_lock<std::mutex> _lock(m);
	
	//_lock.unlock();
	//}
	//std::chrono::milliseconds dura(20000);
	//while (true)
	//this_thread::sleep_for(dura);
	//stmReset();
	srand(time(0));

	/*for (int t = 0; t < Cuda_Max_Stream; t++) {
		game_data[t] = new GomokuData*[128]{ 0 };
		for (int i = 0; i < 128; i++)game_data[t][i] = new GomokuData();
	}*/
	
	//Chess Data
	/*cout << "total Games: " << Gomoku_Data_Parse("renjunet_v10_20210120.rif") << endl;
	tot = 0;for (int i = 0; i < cnt; i++)tot += GameStep[i];
	DataSet ds; ds.trainSet_Save_Load(false, 450000, "dataset");*/
	//for (int t = 0; t < cnt; t++) {
	//	for (int i = 0; i < GameStep[t]; i++)
	//		ds.trainSet_Add_data(Gomoku(ds.trainSet_gameCount(), i));
	//	for (int j = 0; j < GameStep[t]; j++) {
	//		Gomoku::SetMove(ds.trainSet_Param(ds.trainSet_gameCount()), Game[t][j], W*H);
	//	}
	//	//Reward
	//	//Win:1 Loss:-1 Draw:0
	//	ds.trainSet_Add_Reward(GameResult[t]);
	//}
	//DataSet player_ds;
	//player_ds.trainSet_Save_Load(false, 2673103, "Player_DataSet");
	//Go go;
	//for (int k = 7; k <= 8; k++) {
	//	char Path[100]; sprintf(Path, "trainSet_data_%d", k);
	//	DataSet player_ds; player_ds.trainSet_Save_Load(false, 3073103, Path);
	//	DataSet ds; ds.trainSet_Save_Load(false, player_ds.dataCount + player_ds.gameCount + 3, (((string)"extend_") + Path).c_str());
	//	//add absorbing state
	//	ds.trainSet_Add_data(Go::Data(Go::W*Go::H, Go::W*Go::H, 1.0, go));
	//	ds.trainSet_Add_data(Go::Data(Go::W*Go::H, Go::W*Go::H, -1.0, go));
	//	ds.trainSet_Add_data(Go::Data(Go::W*Go::H, Go::W*Go::H, 0.0, go));
	//	int idx = 0;
	//	for (int t = 0; t < player_ds.gameCount; t++) {
	//		int gamestep = player_ds.trainSet_Param(t).Out(0).Count + 1;
	//		int MaxStep = ds.trainSet_dataCount() + gamestep;
	//		go.Reset();
	//		for (int i = 0; i < gamestep; i++) {
	//			ui move = (i == (gamestep - 1)) ? -1 : (player_ds.trainSet_Param(t).Out(0)[i] - 1);
	//			double out[225] = { 0 };
	//			if (i != gamestep - 1)
	//				memcpy(out, &player_ds.trainSet_Param(idx + i).In(0)[1], sizeof(double) * 225);
	//			//uniform policy at terminal state
	//			else fill(out, out + Go::W*Go::H, 1.0 / (Go::W*Go::H));
	//			int scr[225]; go.GetGameScreen(scr);
	//			ui lastmove = i == 0 ? 0 : (player_ds.trainSet_Param(t).Out(0)[i - 1] - 1);
	//			//repeat terminal state
	//			//1:black win(first hand) -1:white win 0:draw
	//			ds.trainSet_Add_data(Go::Data(ds.trainSet_dataCount(), MaxStep, Go::W*Go::H, scr, Go::W*Go::H, out, lastmove, i % 2 == 0 ? player_ds.Reward[t] : -player_ds.Reward[t], 0, &go));
	//			//Value direct from outcome with board games,otherwise bootstrap n=10
	//			if (i != gamestep - 1)go.Act(move, 0);
	//		}
	//		ds.gameCount++;
	//		idx += gamestep - 1;
	//	}
	//	//absorbing
	//	for (int i = 0; i < ds.trainSet_Param(0).Count; i++) {
	//		ds.trainSet_Param(0).In(i) = ds.trainSet_Param(3).In(i);
	//		ds.trainSet_Param(1).In(i) = ds.trainSet_Param(4).In(i);
	//		ds.trainSet_Param(2).In(i) = ds.trainSet_Param(5).In(i);
	//	}
	//	ds.trainSet_Save_Load(true, -1, (((string)"extend_") + Path).c_str());
	//}
	

	
	/*Gomoku*Go = new Gomoku[tot];
	tot = 0;
	for (int i = 0; i < cnt; i++) {
		for (int j = 0; j < GameStep[i]; j++) {
			Go[tot] = Gomoku(i, j, GameResult[i]);
			tot++;
			Gomoku::SetMove(Go[i], Game[i][j], MaxStep);
		}
	}*/
	//48.36%
	//32filters 12features 4conv 14epoch 50.55% 767mins
	//64filters 20features 6conv 9epoch 51.96% 1358mins
	//32filters 20features 6conv 28epoch 51.27% 2400mins
	//32filters 36features 6conv 23epoch 51.49% 52.64% 2800mins
	//32filters 36features 6conv 9epoch 48.50% 1100mins -SGD_no_momentum
	//32filters 36features 6conv 12epoch 49.22% 1500mins -SGD_no_momentum
	//32filters 32features 6conv epoch 51.73% mins -SGD_no_momentum
	AgentGenerator agents("Gomoku_param_Rec", "Gomoku_best_param");
	//player_ds.miniTrain_Start(agents, Gomoku_Policy_Value_ResNet, Gomoku_User_Param, Gomoku::MCTS_GomokuSimulation);
	//agents.MCTS_run(Gomoku_Policy_Value_ResNet, Gomoku_User_Param, GomokuJudger, agentResponse, NULL);// &ds);
	//Go Gomoku;
	//agents.MCTS_run_extend(Gomoku);
	return 0;



	//Reinforce learning
	//HyperParamSearcher param("Gomoku_param_Rec", "Gomoku_best_param");
	//param.Read_Param();
	//int _Agent = 2;
	//Agent**agent = new Agent*[_Agent] {new Agent("Agent_1", param["Batch"], param["Max_Step"], param["Max_srand_row"]), new Agent("Agent_0", param["Batch"], param["Max_Step"], -1, false)};
	//time_t seed = time(0);
	////Agent*sl = new Agent("SL_Net_51.27_Conv6_32filters_20Features", param["Batch"], param["Max_Step"], -1, false);
	////for (int i = 0; i < _Agent; i++) {
	////	//if (i % 2 == 0)seed = time(0);
	////	agent[i] = new Agent(Agent::RNN, param["Batch"], Gomoku_SL_PolicyNet(param), param["Max_Step"],false);
	////	//Agent from File
	////	//Agent*sl = new Agent("Gomoku_SL_Net_FC_39_Policy", param["Batch"], param["Max_Step"]);
	////	//RL Inital from SL
	////	sl->Data_Assignment(agent[i]);
	////	//agent Init Training param
	////	agent[i]->Online_Init(param);
	////	agent[i]->Write_to_File("ConvSL2RL_Net");
	////	Sleep(10);
	////}
	//Agent*RNet = new Agent(Agent::RNN, param["Batch"], Gomoku_SL_PolicyNet(param), param["Max_Step"], true, time(0), 225);
	//RNet->Online_Init(param);
	////AgentGenerator agents;
	//double win = 0;
	//trainSet_Init(param["Batch"] * 15 * 15 * 10);
	//Agent*agent[2] = { RNet,RNet };
	//FiveChess_StartGame(agent, 2, win, 10, true);
	//printf("Win: %lf%%\n", 100 * win);
	//agents.run(1, FiveChess_AgentNet, FiveChess_StartGame, FiveChess_User_Param, agent, 2, win);

	//return 0;





	//League league;
	//league.run();
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
