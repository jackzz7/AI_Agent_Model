#pragma once


#include"RL_MCTS.h"
#include"Game.h"
#include"Gomoku.h"
#include"Go.h"

#include<set>
#include<time.h>
#include<io.h>
#include<direct.h>
#include<random>
#include<experimental/filesystem>
#include<type_traits>
#include<algorithm>

#include<dist.h>

#define DEBUG(expression)(void)(                                                       \
            (!!(expression)) ||                                                              \
            (pipeOut("ERROR expr:%s file:%s line:%u\n",#expression, __FILE__, (unsigned)(__LINE__)), 0)) \

//extern GomokuData**game_data[Cuda_Max_Stream];
namespace multiMCTS_extend {

	static const int bits = 4;
	static const int Maximum_Cache = 1 << bits;
	//static const int BufferSize = 400;


	static const int Max_Action_Space = 100;
	static const int MaxGameLength = 256;

	//random number for stone
	//us Randomer[Max_Action_Space][2];

	/*template<class T>
	void _WR(T& var, char* target, ui& idx, bool Write) {
		if (Write)
			memcpy(&target[idx], (void*)&var, sizeof(T)), idx += sizeof(T);
		else memcpy((void*)&var, &target[idx], sizeof(T)), idx += sizeof(T);
	}
	template<class T>
	void _WR(T* var, char* target, ui Count, ui& idx, bool Write) {
		if (Write)
			memcpy(&target[idx], (void*)var, sizeof(T) * Count), idx += sizeof(T) * Count;
		else memcpy(var, &target[idx], sizeof(T) * Count), idx += sizeof(T) * Count;
	}*/

	static int Hidden_State_Size;
	static int Game_Action_Space;
	static int Screen_Size;

	static double Q_Maximum_Value = 1;
	static double Q_Minimum_Value = -1;

	struct PINFO;
	struct MCTS_Edge {
		const static int Edge_Size = sizeof(int) + 2 * sizeof(float);
		//volatile char next_position_flag;
		volatile float Q_Value;
		//traverse Count
		volatile int Visit_Count;
		//control explore
		volatile float Prior_Probability;
		//link next position ID
		//Hidden State Position
		//int next_position_ID;
		//Real root Position
		//int next_root_ID;
		PINFO* volatile next_Node;


		MCTS_Edge() {
			Q_Value = 0;
			Visit_Count = 0;
			Prior_Probability = -1;
			//next_position_ID = -1;
			next_Node = NULL;
		}
		MCTS_Edge(const double& Priori) :MCTS_Edge() {
			Prior_Probability = Priori;
			//next_position_ID = next_ID;
		}

		//static const int Cpuct = 1.0;
		double Confidence_Bound(const double& sqrt_tot_Node_visit) const {
			assert(!isnan(Q_Value));
			return Q_Value + Prior_Probability * sqrt_tot_Node_visit / (1 + Visit_Count);
		}
		MCTS_Edge& operator=(const MCTS_Edge& e) {
			memcpy(this, &e, sizeof(MCTS_Edge));
			return *this;
		}
		void Update_Q_Value() {
			if (Visit_Count > 0);//Q_Value = tot_Value / Visit_Count;
			else Visit_Count = 0, Q_Value = 0;
		}
		void Virtual_Loss(const int& vl) {
			int tmp = 0;
			if ((tmp = Visit_Count + vl) > 0) {
				assert(tmp > 0);
				Q_Value = (1.0 * Q_Value * Visit_Count - vl) / tmp;
				Visit_Count = tmp;
			}
			else Visit_Count = Q_Value = 0;
		}
		void restore_Virutal_Loss(const int& vl, const std::pair<float, ui>& Visted) {
			int tmp = (Visit_Count + -vl + Visted.Y);
			if (tmp > 0) {
				Q_Value = (1.0 * Q_Value * Visit_Count + vl + Visted.X) / tmp;
				Visit_Count += -vl + Visted.Y;
			}
			else Visit_Count = Q_Value = 0;
		}
	};

	struct EDGE {
		vector<MCTS_Edge>* edge;
		int sz;
		EDGE() {
			sz = 0;
			edge = new vector<MCTS_Edge>();
			//pre-alloc memory
			edge->resize(Game_Action_Space);
		}
		~EDGE() {
			delete edge;
		}
		//get_Edge_with_Maximum_UCB
		MCTS_Edge& UCB(volatile ull& tot_Visit, mt19937& rng, ui& move) {
			++tot_Visit;
			assert(sz > 0);
			const double Cbase = 19652, Cinit = 1.25;
			double sqrt_tot_visit = sqrt(tot_Visit) * (log((1.0 + tot_Visit + Cbase) / Cbase) + Cinit);
			ui idx = rng() % sz, _i = -1;
			MCTS_Edge* best = &(*edge)[idx]; double mx = (*edge)[idx].Confidence_Bound(sqrt_tot_visit);
			best->Virtual_Loss(1);
			for (auto& k : *edge) {
				_i++;
				if (k.Prior_Probability == -1)continue;
				double tar = k.Confidence_Bound(sqrt_tot_visit);
				if (mx < tar || best->Prior_Probability == -1)k.Virtual_Loss(1), best->Virtual_Loss(-1), best = &k, mx = tar, idx = _i;
			}
			//assert(best->Prior_Probability >= 0);
			move = idx;
			return *best;
		}
		/*MCTS_Edge& Sample_Action(volatile ull& tot_Visit, mt19937& rng, ui& move) {
			++tot_Visit;
			assert(sz > 0);
			ui _i = -1;
			double sum = 0, val = 0; for (auto& k : *edge)if (k.Prior_Probability != -1)sum += k.Prior_Probability, assert(k.Prior_Probability >= 0);
			double sample_val = rng() % 100000 / 100000.0 * sum;
			MCTS_Edge* best = NULL;
			for (auto& k : *edge) {
				_i++;
				if (k.Prior_Probability == -1)continue;
				val += k.Prior_Probability;
				if (val > sample_val) {
					move = _i;
					return k;
				}
			}
			assert(false);
			return *best;
		}*/
		void push(const MCTS_Edge& e) {
			assert(sz < Game_Action_Space);
			(*edge)[sz] = e;
			(*edge)[sz].Update_Q_Value();
			//if (sz + 1 < Game_Action_Space)(*edge)[sz + 1].next_position_ID = -1;
			sz++;
		}
		//end flag
		//void push_end() {
		//	if (sz < Game_Action_Space) {
		//		assert(false);
		//		//(*edge)[sz] = MCTS_Edge(-1);
		//	}
		//}
		bool Ready() {
			return (sz == Game_Action_Space);// || (*edge)[sz].next_position_ID == -2;
		}
		bool empty() {
			return sz == 0;
		}
		EDGE& operator=(EDGE&& e) {
			swap(this->edge, e.edge);
			swap(this->sz, e.sz);
			return *this;
		}
	};
	static const int Max_Thr = 128;
	static const int Value_Num = 4;
	struct Position {
		//static size_t Position_Size;
		//ID+Feature=Identity
		int PID;
		Mat Hidden_State;

		float Value[Value_Num];
		float Action_Reward;
		//-1:PID==root_ID other:PID==virtual ID

		EDGE edges;

		Position() {
			PID = -1;
			//root_ID_exist = -1;
			Hidden_State.Reset(1, Hidden_State_Size, false, false, false);
			//Hidden_State = new float[Hidden_State_Size];
		}
		~Position() {
			Hidden_State.Disponse();
		}
	};
	//size_t Position::Position_Size;

	struct PINFO {
		Position p;
		ull totVisit;
		//bool Virtual_File_Flag;
	};
	typedef list<PINFO*> PList;
	struct trajectory_Data {
		static size_t traj_Output_Size;
		//Screen
		double* Scr;
		//action
		ui move = 0;
		//probability
		double* OutPut;
		int out_cnt;
		//four players Value
		double Value[Value_Num];
		//Reward
		double Action_Reward;
		//pass flag
		bool passTrain;
		trajectory_Data() {
			OutPut = new double[traj_Output_Size];
			Scr = new double[Screen_Size];
		}
		~trajectory_Data() {
			delete[] OutPut;
			delete[] Scr;
		}
		trajectory_Data& operator=(const trajectory_Data& right) {
			memcpy(Scr, right.Scr, sizeof(double) * Screen_Size);
			move = right.move;
			memcpy(OutPut, right.OutPut, sizeof(double) * traj_Output_Size);
			out_cnt = right.out_cnt;
			memcpy(Value, right.Value, sizeof(double) * Value_Num);
			Action_Reward = right.Action_Reward;
			passTrain = right.passTrain;
			return *this;
		}
	};
	size_t trajectory_Data::traj_Output_Size;

	//gobal variable
	void Init_MCTS(Environment& e, const char* dir = "") {
		/*fstream file; file.open((string)dir + "XOR_Random_Number", ios::in | ios::binary);
		if (file)file.read((char*)Randomer, Max_Action_Space * 2 * sizeof(us)), assert(file.gcount() == Max_Action_Space * 2 * sizeof(us)), file.close();
		else {
			for (int i = 0; i < Max_Action_Space; i++)
				for (int j = 0; j < 2; j++)
					Randomer[i][j] = rand_i();
			file.close();
			file.open((string)dir + "XOR_Random_Number", ios::out | ios::binary);
			file.write((char*)Randomer, Max_Action_Space * 2 * sizeof(us)), file.close();
		}*/
		Game_Action_Space = e.getRealGameActionSpace();
		Screen_Size = e.getScreen_Size();
		Hidden_State_Size = e.getHiddenStateSize();
		trajectory_Data::traj_Output_Size = 30 * 2;//2 * Game_Action_Space;
		//Position::Position_Size = sizeof(float) * 2 + Game_Action_Space * MCTS_Edge::Edge_Size;
	}


	struct WR_Manager {
		//string DirPath;
		char dir[100];
		WR_Manager(const char* dir_path, const char* Name) {
			if (_access(dir_path, 0) != 0) _mkdir(dir_path);
			sprintf_s(dir, "%s%s\\", dir_path, Name);
			if (_access(dir, 0) != 0) _mkdir(dir);
		}
	};
	struct Evalution {
		int ID, PolicyID;
		//PINFO*Node;
		Mat* H_State;
		ui Action;
		int step;

		double* OutPut;
		//mulit value
		double Value[Value_Num];
		Mat* Next_State;
		double Reward;
		double PlayerID[4];

		Evalution() { Next_State = NULL; OutPut = NULL; }
		void Init(int tid, int Policy_ID, Mat* State, Mat* Next_State, int step, ui action) {
			ID = tid; PolicyID = Policy_ID;
			this->H_State = State; this->step = step;
			Action = action;
			if (!OutPut)OutPut = new double[Game_Action_Space];
			fill(OutPut, OutPut + Game_Action_Space, -1.0);
			//if (!Value)Value = new double[Value_Num] {0};
			fill(Value, Value + Value_Num, -2);
			this->Next_State = Next_State;
			Reward = 0;
			fill(PlayerID, PlayerID + Value_Num, -1);
		}
		~Evalution() {
			delete[] OutPut;
		}
	};

	static bool StochasticPolicy, extra_flag;
	static int extra_data;
	class MCTS_Policy_Extend {

		//temporary stay in memory,interface with file
		WR_Manager* File_Buffer;
		//PositionLookUp* LookUpTable;

		//resign
		double V_resign, Enable_resign;
		double Min_V_resign[2];
		static const int resign_MaxSize = 2000;
		static const int resign_MinSize = 100;
		list<double>recent_resign;
		multiset<double>resign_Sort_Set;
	public:
		int PolicyID;
		bool* ActionIsValid;
		Environment* environment_API;
		condition_variable* Eva_CV = NULL;
		mutex* Eva_m = NULL;

		MCTS_Policy_Extend(int ID, Environment* e, int Thr = 1, int Sim = 500, const char* Name = "0", const char* dir = "MCTS_Policy\\") :PolicyID(ID), Thr_Num(Thr), SimulationPerMCTS(Sim), environment_API(e) {
			File_Buffer = new WR_Manager(dir, Name);
			//LookUpTable = new PositionLookUp();
			ActionIsValid = new bool[Game_Action_Space];
			dir_noise = new double[Game_Action_Space];

			trainData = new trajectory_Data[MaxGameLength];
			for (int i = 0; i < Maximum_Round_Num; i++)
				GameData[i] = new trajectory_Data[MaxGameLength];
			fill(Response, Response + Max_Thr, true);

			Finish_Thr = 0;
			for (int t = 0; t < Thr_Num; t++)
				this->Thr[t] = new thread(&MCTS_Policy_Extend::Thread_Start, this, t, std::ref(Node[0]), (Mat*)NULL, -1);
			std::unique_lock<std::mutex> lock(Thr_lock);
			Main_CV.wait(lock, [this]() {return Finish_Thr == Thr_Num; });
			//Finish_Thr = 0;

			V_resign = -1e9;
			recent_resign.clear();
			resign_WR(false);
			resign_Sort_Set.clear();
			for (auto& k : recent_resign)resign_Sort_Set.insert(k);
			//init random
			std::vector<std::uint32_t> seeds;
			seeds.resize(Thr_Num + 1);
			std::seed_seq seq{ time(0),(ll)PolicyID };
			seq.generate(seeds.begin(), seeds.end());
			for (int t = 0; t < Thr_Num + 1; t++)
				rng[t].seed(seeds[t]);
		}
		~MCTS_Policy_Extend() {
			Finish_Thr = -1;
			for (int i = 0; i < Thr_Num; i++) {
				{
					lock_guard<mutex>locker(Thr_m[i]);
					Thr_CV[i].notify_one();
				}
				Thr[i]->join();
				delete Thr[i];
			}
			delete File_Buffer;
			//delete LookUpTable;
			delete[] ActionIsValid;
			delete[] dir_noise;
			delete[] trainData;
			for (int i = 0; i < Maximum_Round_Num; i++)delete[] GameData[i];
			for (int j = 0; j < Max_Thr; j++) {
				pre_alloc[j].remove_if([](auto const& e) {delete e; return true; });
			}
		}
		void remove_dir() {
			char path[100]; strcpy(path, File_Buffer->dir);
			delete File_Buffer; File_Buffer = NULL;
			std::experimental::filesystem::remove_all(path);
		}
		void resign_WR(bool Write) {
			string path = File_Buffer->dir; path += "resign";
			fstream file; file.open(path.c_str(), ios::binary | (Write ? ios::out : ios::in));
			if (file) {
				int sz = recent_resign.size();
				File_WR(file, &sz, sizeof(sz), Write);
				assert(sz <= resign_MaxSize);
				if (!Write)recent_resign.resize(sz);
				auto it = recent_resign.begin();
				for (int i = 0; i < sz; i++) {
					double& val = *(it++);
					File_WR(file, &val, sizeof(val), Write);
				}
			}
			file.close();
		}
		void Update_and_Clear() {
			for (int i = 0; i < Thr_Num; i++) {
				for (auto& k : Path[i]) {
					if (k._new)
						pre_alloc[i].push_back(k._new);
					k.node = NULL;
				}
				Path[i].clear();
			}
		}
		pair<PINFO*, int> newPosition(list<PINFO*>& alloc) {
			//assert(ID == -1);
			//Create or Initial from file
			auto p = alloc.front();
			p->p.PID = -1;// p->p.root_ID_exist = -1;
			p->p.edges.sz = 0; p->totVisit = 0; //p->Virtual_File_Flag = false;
			return { p,0 };
			//return File_Buffer.ReadPosition(ID, Feature, alloc);
		}
		
		//write all data to file
		void Write2File() {
			//Eva_Loop_end();
			resign_WR(true);
		}
		struct Node_Info {
			PINFO* volatile& node, *_new;
			Node_Info(PINFO* volatile& Node, PINFO* __new) :node(Node), _new(__new){}
		};
		volatile int Response[Max_Thr];
		Evalution Eva_Stack[Max_Thr];
		condition_variable Response_CV[Max_Thr];
		mutex Response_mux[Max_Thr];
		mt19937 rng[Max_Thr + 1];
		vector<Node_Info>Path[Max_Thr];
		list<PINFO*> pre_alloc[Max_Thr];
		struct Result {
			double Values[Value_Num];
			int InsPlayerID;
		}result[Max_Thr];
		void MC_Tree_Search(PINFO* volatile& Node, const int step, Mat* Hidden_State, const ui action,int PlayerID, int tid) {
			int type = -1; PINFO* _new = NULL;
			//Create new Node
			if (!Node) {
				auto p = newPosition(pre_alloc[tid]);
				if (!Node) {
					type = p.Y, Node = p.X;
					_new = p.X; assert(_new == pre_alloc[tid].front());
					pre_alloc[tid].pop_front();
				}
			}
			//leaf Node,expanded by Agent
			if (type == 0) {
				Eva_Stack[tid].Init(tid, PolicyID, Hidden_State, &Node->p.Hidden_State, step, action);
				Response[tid] = 0;
				//wait for evalution
				{
					unique_lock<std::mutex> lock(Response_mux[tid]);
					Response_CV[tid].wait(lock, [this, tid]() { return Response[tid] == 1; });
				}
				EDGE* edge = &Node->p.edges;
				Path[tid].emplace_back(Node_Info(Node, _new));
				assert(_new != NULL);
				//node already evalution
				assert(abs(Eva_Stack[tid].Value[0]) <= 1 + 1e-8);
				//copy return values
				memcpy(result[tid].Values, Eva_Stack[tid].Value, sizeof(double) * Value_Num);
				result[tid].InsPlayerID = PlayerID;

				if (Node != _new) {
					//return -Eva_Stack[tid].Value;
					return;
				}
				double* Action_Prior = Eva_Stack[tid].OutPut;// , Value = Eva_Stack[tid].Value;
				double Reward = Eva_Stack[tid].Reward;

				bool* ActionMask = NULL;
				//modify stochastic action mask
				//get current simulation action mask
				if (!Hidden_State) {
					ActionMask = new bool[Game_Action_Space];
					memcpy(ActionMask, ActionIsValid, sizeof(bool) * Game_Action_Space);
					if (StochasticPolicy)
						environment_API->ModifyStochasticActionMask(Action_Prior, ActionMask);
				}

				//add root action mask
				//no Action mask in inner Simulations
				for (int k = 0; k < Game_Action_Space; k++) {
					edge->push(MCTS_Edge((!Hidden_State && !ActionMask[k]) ? -1 : (environment_API->getSpecialActionMask(action, k, Action_Prior[k]) ? Action_Prior[k] : -1)));
					assert(Action_Prior[k] + 1e-8 >= 0);
				}
				delete[] ActionMask;
				//edge->push_end();
				//add extra P(s,a) noise
				if (IsAddNoise && !Hidden_State)generate_noise(Node, environment_API->legal_moves_Count, tid);
				assert(edge == &Node->p.edges);
				assert(edge->sz == Game_Action_Space);

				for (int i = 0; i < Value_Num; i++)Node->p.Value[i] = Eva_Stack[tid].Value[i];
				Node->p.Action_Reward = 0;// Reward;

				//wake up yield
				for (int i = 0; i < Thr_Num; i++) {
					if (Response[i] == 2) {
						lock_guard<mutex> lock(Response_mux[i]);
						Response_CV[i].notify_one();
					}
				}
				return;
			}
			//traverse UCB edges
			else {
				//wait all edges insert
				if (!Node->p.edges.Ready()) {
					Response[tid] = 2;
					std::unique_lock<std::mutex> lock(Response_mux[tid]);
					Response_CV[tid].wait(lock, [&Node]() { return Node->p.edges.Ready(); });
				}
				Response[tid] = 1;

				EDGE* edge = &Node->p.edges;
				assert(edge->sz == Game_Action_Space);
				ui move = -1;
				MCTS_Edge& best_act = edge->UCB(Node->totVisit, rng[tid], move);

				//simple determine next PlayerID
				int NextPlayerID = environment_API->GetNextPlayerID(move, PlayerID);

				////MCTS traverse subtree
				MC_Tree_Search(best_act.next_Node, step + 1, &Node->p.Hidden_State, move, NextPlayerID, tid);
				//discount Reward
				assert(abs(best_act.next_Node->p.Action_Reward) <= 1 + 1e-8);
				//result = best_act.next_Node->p.Action_Reward + environment_API->getRewardDiscount() * result;

				//Update best_act
				double val = result[tid].Values[(PlayerID - result[tid].InsPlayerID + Value_Num) % Value_Num];
				best_act.restore_Virutal_Loss(1, { val,1 });
				//assert(edge == &Node->p.edges);
				if (type == -1) {
					assert(Node != NULL&& !Node->p.edges.empty());
				}
				else Path[tid].emplace_back(Node_Info(Node, _new));
				return;
			}
		}
		//add dirichlet noise to P(s,a)
		const double dir_alpha = 0.75;
		double* dir_noise;
		void addDirNoise(PINFO* Node, double* dir_distribution) {
			for (int i = 0; i < Game_Action_Space; i++) {
				auto& e = (*Node->p.edges.edge)[i];
				if (e.Prior_Probability == -1)continue;
				e.Prior_Probability = dir_alpha * e.Prior_Probability + (1 - dir_alpha) * dir_distribution[i];
			}
		}
		void removeDirNoise(PINFO* Node, double* dir_distribution) {
			for (int i = 0; i < Game_Action_Space; i++) {
				auto& e = (*Node->p.edges.edge)[i];
				if (e.Prior_Probability == -1)continue;
				e.Prior_Probability = (e.Prior_Probability - (1 - dir_alpha) * dir_distribution[i]) / dir_alpha;
			}
		}
		void generate_noise(PINFO* NoisedNode, int legal_moves_Count, int tid) {
			//~10 random moves over all average legal moves 
			const double times = 10.0 / legal_moves_Count;// (Action_Space / 4 * 3);
			std::gamma_distribution<double> dist(times, 1);
			double sum = 0;
			for (int i = 0; i < Game_Action_Space; i++) {
				if ((*NoisedNode->p.edges.edge)[i].Prior_Probability == -1)continue;
				sum += dir_noise[i] = dist(rng[tid]), assert(dir_noise[i] > 0);
			}
			for (int i = 0; i < Game_Action_Space; i++) {
				if ((*NoisedNode->p.edges.edge)[i].Prior_Probability == -1)continue;
				dir_noise[i] /= sum;
			}
			addDirNoise(NoisedNode, dir_noise);
		}

		ui Sampling_Visit(PINFO* Node, MCTS_Edge*& best_act, int step, ull& tot, double* Max_Q = NULL, int Sampling_step = 0) {
			ui move = -1;
			for (auto& e : *Node->p.edges.edge) {
				move++;
				if (e.Prior_Probability == -1)continue;
				if (!e.next_Node)continue;
				if (!ActionIsValid[move])continue;
				tot += e.Visit_Count, assert(e.Visit_Count >= 0);
				assert(!isnan(e.Q_Value));
				if (Max_Q)*Max_Q = max(*Max_Q, e.Q_Value);
			}assert(tot > 0);
			//sampling available moves
			move = -1;
			if (step < Sampling_step) {
				ull chi = rng[0]() % tot, cnt = 0;
				for (auto& e : *Node->p.edges.edge) {
					move++;
					if (e.Prior_Probability == -1)continue;
					if (!e.next_Node)continue;
					if (!ActionIsValid[move])continue;
					cnt += e.Visit_Count;
					if (cnt > chi) {
						best_act = &e;
						return move;
					}
				}
				assert(false); return -1;
			}
			//max count
			else {
				vector<pair<MCTS_Edge*, ui>>Q; Q.clear();
				ull Visit_Count = 0;
				for (auto& e : *Node->p.edges.edge) {
					move++;
					if (e.Prior_Probability == -1)continue;
					if (!e.next_Node)continue;
					if (!ActionIsValid[move])continue;
					if (Visit_Count < e.Visit_Count)Visit_Count = e.Visit_Count, Q.clear(), Q.push_back({ &e,move });
					else if (Visit_Count == e.Visit_Count)Q.push_back({ &e,move });
				}
				int idx = rng[0]() % Q.size();
				best_act = Q[idx].first;
				assert(best_act->Visit_Count > 0);
				return Q[idx].second;
			}
		}
		volatile int Finish_Thr;
		condition_variable Thr_CV[Max_Thr];
		mutex Thr_m[Max_Thr], Thr_lock;
		bool IsAddNoise;
		int firstPlayerID;
		void Thread_Start(int tid, PINFO* volatile& Node, Mat* Hidden_State, const ui Action) {
			std::unique_lock<std::mutex> lock(Thr_m[tid]);
			{
				lock_guard<mutex>locker(Thr_lock);
				if (++Finish_Thr == Thr_Num)Main_CV.notify_one();
			}
			while (true) {
				Thr_CV[tid].wait(lock);
				//exit
				if (Finish_Thr == -1)break;
				for (int i = 0; i < SimulationPerMCTS / Thr_Num; i++) {
					int PID = -1; MC_Tree_Search(Node, Step, Hidden_State, Action, 0, tid);
				}
				lock_guard<mutex>locker(Thr_lock);
				if (++Finish_Thr == Thr_Num)Main_CV.notify_one();
			}
		}

		static const int Maximum_Round_Num = 64;
		trajectory_Data* GameData[Maximum_Round_Num];
		int GameOrder[Maximum_Round_Num][MaxGameLength], RoundStep[Maximum_Round_Num];

		trajectory_Data* trainData;
		int round_number;
		int GameResult[Value_Num], GameStep;
		int PlayerIDOrder[MaxGameLength];
		void Terminal_State(int step, int _) {

			//set only single win,if more than one player win
			for (int i = 1; i <= step; i++)
				if (trainData[i].move % (3 * 4 * 4 + 9) == 54) {
					step = i; break;
				}

			//PlayerIDOrder[step] = environment_API->GetNextPlayerID(trainData[step].move, PlayerIDOrder[step - 1]);//environment_API->getInsCheckPlayerID();
			PlayerIDOrder[step] = environment_API->GetNextPlayerID(trainData[step].move, PlayerIDOrder[step - 1]);

			GameStep = step;
			//add terminal state
			//environment_API->GetGameScreen(trainData[GameStep].Scr);
			//trainData[GameStep].move = move;
			fill(trainData[GameStep].Value, trainData[GameStep].Value + Value_Num, 0);
			//uniform policy at terminal state
			fill(trainData[GameStep].OutPut, trainData[GameStep].OutPut + trajectory_Data::traj_Output_Size, 0.0);
			int& idx = trainData[GameStep].out_cnt = 0;
			trainData[GameStep].OutPut[idx++] = -1;
			trainData[GameStep].Action_Reward = 0;
			trainData[GameStep].passTrain = true;
			GameStep++;

			//save round
			for (int i = 0; i < GameStep; i++) {
				GameData[round_number][i] = trainData[i];
				GameOrder[round_number][i] = PlayerIDOrder[i];
				RoundStep[round_number] = GameStep;
				assert(PlayerIDOrder[i] != -1);
				assert(i == 0 || (PlayerIDOrder[i] == environment_API->GetNextPlayerID(trainData[i].move, PlayerIDOrder[i - 1])));
			}
		}
		thread* Thr[Max_Thr];
		condition_variable Main_CV;
		//generate RL data
		struct Sorted_Policy {
			int move;
			int Count;
			ui random_number;
			bool operator<(const Sorted_Policy& R) {
				if (Count != R.Count)return Count < R.Count;
				else return random_number < R.random_number;
			}
		};
		PINFO* Test_Root;
		vector<int> Search_Policy(int step, int sampling_step, vector<int> action_mask) {
			PlayerIDOrder[step] = environment_API->getInsCheckPlayerID();
			//assert(step == 0 || (PlayerIDOrder[step] == environment_API->GetNextPlayerID(trainData[step].move, PlayerIDOrder[step - 1])));
			vector<Sorted_Policy>Policy;
			memset(ActionIsValid, 0, sizeof(bool) * Game_Action_Space);
			//fill(ActionIsValid, ActionIsValid + Game_Action_Space, true);
			int cnt = 0;
			for (auto& action : action_mask)ActionIsValid[environment_API->getAct(action)] = true, cnt++;
			//simplify Select Policy
			if (cnt == 0) {
				//meld failed(can't discard tile because tile banned relu)
				//have modify env,can't happen
				assert(false);
				//assert(36 <= trainData[step].move && trainData[step].move <= 39); 
				return {};
			}
			//check
			if (sampling_step != 0 && step > 0 && trainData[step].move % (3 * 4 * 4 + 9) < 3 * 4 * 4) {
				int check = trainData[step].move % (3 * 4 * 4 + 9) % (3 * 4) / 3, cc = 0;
				for (auto& act : action_mask) {
					if (abs(act / 34 - 50) <= 1)assert(check == 1);
					else if (act / 34 == 52 || act / 34 == 53)assert(check == 2);
					else if (act / 34 == 54)assert(check == 3 || check == 0);
					else cc++;
				}if (cc == action_mask.size())assert(check == 0);
			}


			//for (int i = 0; i < Game_Action_Space; i++)
			//	if (ActionIsValid[i])
			//		Policy.push_back({ (int)environment_API->getSimAct(i),0,rng[0]() });
			////fast Uniform
			//environment_API->GetGameScreen(trainData[step].Scr);
			////trainData[step + 1].move = environment_API->getAct(_move);
			//for (int i = 0; i < Value_Num; i++)trainData[step].Value[i] = 0;
			//int& idx = trainData[step].out_cnt = 0; double sum = 0;
			//fill(trainData[step].OutPut, trainData[step].OutPut + trajectory_Data::traj_Output_Size, 0.0);
			//for (int i = 0; i < Game_Action_Space; i++)
			//	if (ActionIsValid[i]) {
			//		trainData[step].OutPut[idx++] = i;//environment_API->getAct(i);
			//		sum += trainData[step].OutPut[idx++] = 1.0 / cnt;
			//	}
			//assert(idx <= trajectory_Data::traj_Output_Size);
			//assert(abs(sum - 1) <= 1e-8);
			//trainData[step].Action_Reward = 0;
			//trainData[step].passTrain = false;


			StochasticPolicy = false;
			extra_flag = PlayerIDOrder[step] == extra_data;
			//MCTS Policy
			{
				Finish_Thr = 0;
				Eva_CV->notify_one();
			}
			//resultID = PlayerID;
			Step = step; IsAddNoise = (sampling_step != 0);
			//X Simulation from root
			for (int t = 0; t < Thr_Num; t++) {
				lock_guard<mutex>locker(Thr_m[t]);
				Thr_CV[t].notify_one();
			}
			{
				std::unique_lock<std::mutex> lock(Thr_lock);
				Main_CV.wait(lock, [this]() {return Finish_Thr == Thr_Num; });
			}
			//remove noise
			if (IsAddNoise)
				removeDirNoise(Node[0], dir_noise);

			PINFO* Root = Node[0]; float* Value = Node[0]->p.Value;
			Test_Root = Root;
			//Agent sampling move
			MCTS_Edge* best_act = NULL; ull tot = 0; double Max_Q = -1e9;
			//sampling move over all steps
			//ui move = Sampling_Visit(Root, best_act, step, tot, &Max_Q, 15);
			//assert(Max_Q != -1e9 && move != -1);
			ui move = -1;
			for (auto& e : *Root->p.edges.edge) {
				move++;
				if (e.Prior_Probability == -1)continue;
				if (!e.next_Node)continue;
				if (!ActionIsValid[move])continue;
				tot += e.Visit_Count, assert(e.Visit_Count >= 0);
				assert(!isnan(e.Q_Value));
				//if (Max_Q)*Max_Q = max(*Max_Q, e.Q_Value);
			}assert(tot > 0);

			//add Train Data
			environment_API->GetGameScreen(trainData[step].Scr);
			//trainData[step + 1].move = environment_API->getAct(move);
			for (int i = 0; i < Value_Num; i++)trainData[step].Value[i] = Value[i];
			int& idx = trainData[step].out_cnt = 0;
			fill(trainData[step].OutPut, trainData[step].OutPut + trajectory_Data::traj_Output_Size, 0.0);
			double sum = 0; ui _i = -1;
			double best_P = -1, best_i = 0;
			for (auto& e : *Root->p.edges.edge) {
				_i++;
				if (e.Prior_Probability == -1)continue;
				if (!ActionIsValid[_i])continue;
				if (!e.next_Node) {
					Policy.push_back({ (int)environment_API->getSimAct(_i),0,rng[0]() });
					continue;
				}
				if (idx + 2 <= trajectory_Data::traj_Output_Size) {
					trainData[step].OutPut[idx++] = _i;// environment_API->getAct(_i);
					sum += trainData[step].OutPut[idx++] = 1.0 * e.Visit_Count / tot;
				}
				else assert(false);
				Policy.push_back({ (int)environment_API->getSimAct(_i),e.Visit_Count,rng[0]() });
				if (e.Prior_Probability > best_P)best_P = e.Prior_Probability, best_i = _i;
			}


			assert(idx <= trajectory_Data::traj_Output_Size);
			assert(abs(sum - 1) <= 1e-8);
			trainData[step].Action_Reward = 0;
			trainData[step].passTrain = false;

			Update_and_Clear();

			/*set<int>A, B;
			for (auto& k : Policy)A.insert(k.move);
			for (auto& k : action_mask)B.insert(k);
			assert(A == B);*/
			//sorted Policy move
			assert(Policy.size() == cnt);

			vector<int>result;
			//sampling move
			if (step < sampling_step) {
				ll tot = 0; for (auto& k : Policy) {
					if (k.Count == 0)k.Count = 1; tot += k.Count;
				}
				ui* ID = new ui[tot], * tmpOrd = new ui[tot];
				int* id2move = new int[tot]; map<int, bool>cap; int _i = 0;
				for (auto& k : Policy)
					for (int i = 0; i < k.Count; i++)id2move[_i++] = k.move;
				for (int i = 0; i < tot; i++)tmpOrd[i] = rng[0](), ID[i] = i;
				sort(ID, ID + tot, [&](const ui& a, const ui& b) {return tmpOrd[a] < tmpOrd[b]; });
				for (int i = 0; i < tot; i++) {
					int id = ID[i];
					if (!cap[id2move[id]]) {
						cap[id2move[id]] = true;
						result.push_back(id2move[id]);
					}
				}
				delete[] ID;
				delete[] tmpOrd;
				delete[] id2move;
			}
			//greedy
			else {
				sort(Policy.begin(), Policy.end());
				reverse(Policy.begin(), Policy.end());
				for (auto& k : Policy)result.push_back(k.move);
			}
			assert(result.size() == Policy.size());

			//speed up train process
			//enough to generate
			//not small,not big
			ui drop_win_percent = 10, flag = rng[0]() % 100;
			//ui meld_percent = 90, meld_flag = rng[0]() % 100;
			bool choiced = false;
			for (auto& k : Policy)
				//auto win
				if (k.move / 34 == 54) {
					result.assign(1, k.move);
					choiced = true;
				}
			    //auto draw
				else if (k.move / 34 == 55) {
					result.assign(1, k.move);
					choiced = true;
				}
			//auto loss
			//slow drop win tile percent
			//to ensure dateSet Size
				else if (k.move / 34 < 48 && k.move / 34 % 12 / 3 == 3 && flag < drop_win_percent) {
					result.assign(1, k.move);
				}
			//not chi/pon/kan
			//always pass if can

			//check Node visited choice if always pass
			//later,always sample action with chi/pon/kan special action
			//ignore when test
			//if(!choiced)
			//	//first increase meld,then always sample sepcial action
			//	for (auto& k : Policy)
			//		if (k.move / 34 < 48 && k.move / 34 % 3 == 1) {
			//			result.assign(1, k.move);
			//		}
			/*if (PlayerIDOrder[step] == 1) {
				int user_choose = -1;
				cin >> user_choose;
				result.assign(1, user_choose);
			}*/

			trainData[step + 1].move = environment_API->getAct(result[0]);

			return result;
		}
		vector<int> Search_Stochastic_Policy(int step, int sampling_step, vector<int> action_mask) {

			vector<Sorted_Policy>Policy;
			memset(ActionIsValid, 0, sizeof(bool) * Game_Action_Space);
			//fill(ActionIsValid, ActionIsValid + Game_Action_Space, true);
			int cnt = 0;for (auto& action : action_mask)ActionIsValid[environment_API->getAct(action)] = true, cnt++;
			environment_API->StochasticEnv();
			//generate traj data and move


			StochasticPolicy = true;
			//MCTS Policy
			{
				Finish_Thr = 0;
				Eva_CV->notify_one();
			}
			//resultID = PlayerID;
			Step = step; IsAddNoise = (sampling_step != 0);
			//X Simulation from root
			for (int t = 0; t < Thr_Num; t++) {
				lock_guard<mutex>locker(Thr_m[t]);
				Thr_CV[t].notify_one();
			}
			{
				std::unique_lock<std::mutex> lock(Thr_lock);
				Main_CV.wait(lock, [this]() {return Finish_Thr == Thr_Num; });
			}
			//remove noise
			if (IsAddNoise)
				removeDirNoise(Node[0], dir_noise);

			PINFO* Root = Node[0]; float* Value = Node[0]->p.Value;
			//Agent sampling move
			MCTS_Edge* best_act = NULL; ull tot = 0; double Max_Q = -1e9;
			//sampling move over all steps
			//ui move = Sampling_Visit(Root, best_act, step, tot, &Max_Q, 15);
			//assert(Max_Q != -1e9 && move != -1);
			ui move = -1;
			for (auto& e : *Root->p.edges.edge) {
				move++;
				if (e.Prior_Probability == -1)continue;
				if (!e.next_Node)continue;
				//if (!ActionIsValid[move])continue;
				tot += e.Visit_Count, assert(e.Visit_Count >= 0);
				assert(!isnan(e.Q_Value));
				//if (Max_Q)*Max_Q = max(*Max_Q, e.Q_Value);
			}assert(tot > 0);

			double sum = 0; ui _i = -1;
			double best_P = -1, best_i = 0;
			for (auto& e : *Root->p.edges.edge) {
				_i++;
				if (e.Prior_Probability == -1)continue;
				//if (!ActionIsValid[_i])continue;
				//find correspond real action
				int target = -1;
				if (_i % 57 < 48)
					for (auto& k : action_mask) {
						int act = environment_API->getAct(k);
						if (act % 57 >= 48)continue;
						if (act / 57 == _i / 57 && act % 57 % 3 == _i % 57 % 3)assert(target == -1), target = act;
					}
				else target = _i;
				assert(target != -1);
				target = environment_API->getSimAct(target);
				Policy.push_back({ target,e.next_Node ? e.Visit_Count : 0,rng[0]() });

				//if (e.Prior_Probability > best_P)best_P = e.Prior_Probability, best_i = _i;
			}

			Update_and_Clear();

			this_thread::sleep_for(std::chrono::milliseconds(rng[0]() % 2 * 1000));

			//sorted Policy move
			assert(Policy.size() == cnt);
			vector<int>result;
			//sampling move
			if (step < sampling_step) {
				ll tot = 0; for (auto& k : Policy) {
					if (k.Count == 0)k.Count = 1; tot += k.Count;
				}
				ui* ID = new ui[tot], * tmpOrd = new ui[tot];
				int* id2move = new int[tot]; map<int, bool>cap; int _i = 0;
				for (auto& k : Policy)
					for (int i = 0; i < k.Count; i++)id2move[_i++] = k.move;
				for (int i = 0; i < tot; i++)tmpOrd[i] = rng[0](), ID[i] = i;
				sort(ID, ID + tot, [&](const ui& a, const ui& b) {return tmpOrd[a] < tmpOrd[b]; });
				for (int i = 0; i < tot; i++) {
					int id = ID[i];
					if (!cap[id2move[id]]) {
						cap[id2move[id]] = true;
						result.push_back(id2move[id]);
					}
				}
				delete[] ID;
				delete[] tmpOrd;
				delete[] id2move;
			}
			//greedy
			else {
				sort(Policy.begin(), Policy.end());
				reverse(Policy.begin(), Policy.end());
				for (auto& k : Policy)result.push_back(k.move);
			}

			return result;
		}
		vector<int> Generate_Stochastic_Policy(int step, int sampling_step, vector<int> action_mask) {
			//generate traj data and move
			/*traj_data_Randomize(action_mask, step * 6);
			return { (int)environment_API->getSimAct(trainData[step * 6 + 1].move) };*/
		}
		//void traj_data_Randomize(vector<int> action_mask, int step) {
		//	//get action mask and transform
		//	//only using discard action in tree search for simplification
		//	PlayerIDOrder[step] = environment_API->getInsCheckPlayerID();
		//	int unroll_step = environment_API->getMaxUnrolledStep();
		//	vector<Sorted_Policy>Policy;
		//	memset(ActionIsValid, 0, sizeof(bool) * Game_Action_Space);
		//	//root Node
		//	if (step % unroll_step == 0) {
		//		//send Screen Data
		//		//environment_API->Reset(data.Scr);
		//		for (auto& action : action_mask)ActionIsValid[environment_API->getAct(action)] = true;
		//		//randomize tiles
		//		environment_API->StochasticEnv();
		//		//replace action
		//		environment_API->ModifyStochasticActionMask(NULL, ActionIsValid);
		//	}
		//	//using root randomize info
		//	//only can discard tile
		//	else {
		//		//terminal state
		//		if (environment_API->GetGameState(NULL)) {
		//			Terminal(step);
		//			return;
		//		}
		//		//only simple discard action
		//		environment_API->GetInsActionMask(ActionIsValid);
		//	}
		//	int cnt = 0;
		//	for (int i = 0; i < Game_Action_Space; i++)if (ActionIsValid[i])cnt++;
		//	assert(cnt > 0);

		//	for (int i = 0; i < Game_Action_Space; i++)
		//		if (ActionIsValid[i])
		//			Policy.push_back({ i,0,rng[0]() });
		//	//fast Uniform
		//	environment_API->GetGameScreen(trainData[step].Scr);
		//	int& idx = trainData[step].out_cnt = 0; double sum = 0;
		//	fill(trainData[step].OutPut, trainData[step].OutPut + trajectory_Data::traj_Output_Size, 0.0);
		//	for (int i = 0; i < Game_Action_Space; i++)
		//		if (ActionIsValid[i]) {
		//			trainData[step].OutPut[idx++] = i;
		//			sum += trainData[step].OutPut[idx++] = 1.0 / cnt;
		//		}
		//	assert(idx <= trajectory_Data::traj_Output_Size);
		//	assert(abs(sum - 1) <= 1e-8);
		//	trainData[step].Action_Reward = 0;
		//	trainData[step].passTrain = false;

		//	assert(Policy.size() == cnt);
		//	vector<int>result;
		//	//greedy
		//	{
		//		sort(Policy.begin(), Policy.end());
		//		reverse(Policy.begin(), Policy.end());
		//		for (auto& k : Policy)result.push_back(k.move);
		//	}
		//	trainData[step + 1].move = result[0];

		//	environment_API->Act(trainData[step + 1].move, NULL);

		//	traj_data_Randomize(action_mask, step + 1);
		//}
		inline Environment::Policy_Func getPolicyFunc() {
			return std::bind(&MCTS_Policy_Extend::Search_Policy, this, _1, _2, _3);
		}
		inline Environment::Policy_Func getStochasticPolicyFunc() {
			return std::bind(&MCTS_Policy_Extend::Search_Stochastic_Policy, this, _1, _2, _3);
		}
		inline Environment::Policy_Func getGeneratePolicyFunc() {
			return std::bind(&MCTS_Policy_Extend::Generate_Stochastic_Policy, this, _1, _2, _3);
		}
		inline Environment::State_Func getStateFunc() {
			return std::bind(&MCTS_Policy_Extend::One_State, this, _1, _2);
		}
		inline Environment::State_Func getMoveStateFunc() {
			return std::bind(&MCTS_Policy_Extend::Move_State, this, _1, _2);
		}
		inline Environment::State_Func getSetStateMoveFunc() {
			return std::bind(&MCTS_Policy_Extend::Set_State_Move, this, _1, _2);
		}
		inline Environment::State_Func getInitRoundFunc() {
			return std::bind(&MCTS_Policy_Extend::Init_Round, this, _1, _2);
		}
		inline Environment::State_Func getTerminalStateFunc() {
			return std::bind(&MCTS_Policy_Extend::Terminal_State, this, _1, _2);
		}
		inline Environment::GameResult_Func getSetGameResultFunc() {
			return std::bind(&MCTS_Policy_Extend::SetGameResult, this, _1);
		}
		void Set_State_Move(const int&step,const int&move) {
			assert(false);
			trainData[step].move = move;
			assert(step > 0);
		}
		void Move_State(const int&dst_step,const int&src_step) {
			assert(false);
			int _move = trainData[dst_step].move;
			int __move = trainData[src_step + 1].move;
			trainData[dst_step] = trainData[src_step];
			trainData[dst_step].move = _move;
			trainData[dst_step + 1].move = __move;
			assert(PlayerIDOrder[src_step] != -1);
			PlayerIDOrder[dst_step] = PlayerIDOrder[src_step];
		}
		void One_State(const int&step,const int&action) {
			assert(false);
			assert(step > 0);
			assert(PlayerIDOrder[step - 1] != -1);
			assert(action >= 34 || environment_API->getInsCheckPlayerID() == environment_API->GetNextPlayerID(trainData[step].move, PlayerIDOrder[step - 1]));
			assert(0 <= action < 34 || action == 43 || action == 45);
			//if (action < 34)
			PlayerIDOrder[step] = environment_API->GetNextPlayerID(trainData[step].move, PlayerIDOrder[step - 1]);
			//else PlayerIDOrder[step] = (PlayerIDOrder[step - 1] + 1) % 4;
			//add simple trainData
			//environment_API->GetGameScreen(trainData[step].Scr);
			trainData[step + 1].move = environment_API->getAct(action);
			//for (int i = 0; i < Value_Num; i++)trainData[step].Value[i] = Value[i];
			int& idx = trainData[step].out_cnt = 0;
			fill(trainData[step].OutPut, trainData[step].OutPut + trajectory_Data::traj_Output_Size, 0.0);
			trainData[step].OutPut[idx++] = environment_API->getAct(action);
			trainData[step].OutPut[idx++] = 1.0;
			trainData[step].Action_Reward = 0;
			trainData[step].passTrain = true;
		}
		void Init_Round(const int&_,const int&__) {
			//increase round number
			round_number++;
			fill(PlayerIDOrder, PlayerIDOrder + MaxGameLength, -1);
		}
		void SetGameResult(int*result) {
			for (int i = 0; i < 4; i++)
				GameResult[i] = result[i];
		}
		
		

		//Update Policy using lastest network
		void Data_Reanalyze(int step) {
			//environment_API->GetInsActionMask(ActionIsValid);
			if (trainData[step].OutPut[0] == -1)return;
			//memset(ActionIsValid, 0, sizeof(bool) * Game_Action_Space);
			for (int i = 0; i < trainData[step].out_cnt / 2; i++) {
				int idx = environment_API->getSimAct(trainData[step].OutPut[i * 2]);
				//ActionIsValid[idx] = true;
			}
			Finish_Thr = 0; Eva_CV->notify_one();
			Step = step; IsAddNoise = true;
			//send Screen Data
			environment_API->DirectRepresent(trainData[step].Scr);
			//X Simulation from root
			for (int t = 0; t < Thr_Num; t++) {
				lock_guard<mutex>locker(Thr_m[t]);
				Thr_CV[t].notify_one();
			}
			{
				std::unique_lock<std::mutex> lock(Thr_lock);
				Main_CV.wait(lock, [this]() {return Finish_Thr == Thr_Num; });
			}
			//remove noise
			removeDirNoise(Node[0], dir_noise);

			MCTS_Edge* best_act = NULL; ull tot = 0; double Max_Q = -1e9;
			Sampling_Visit(Node[0], best_act, step, tot, &Max_Q, 10);

			//Update data Policy
			int& idx = trainData[step].out_cnt = 0;
			fill(trainData[step].OutPut, trainData[step].OutPut + trajectory_Data::traj_Output_Size, 0.0);
			double sum = 0; ui _i = -1;
			for (auto& e : *Node[0]->p.edges.edge) {
				_i++;
				if (e.Prior_Probability == -1)continue;
				if (!e.next_Node)continue;
				//if (!ActionIsValid[_i])continue;
				trainData[step].OutPut[idx++] = environment_API->getAct(_i);
				sum += trainData[step].OutPut[idx++] = 1.0 * e.Visit_Count / tot;
			}assert(abs(sum - 1) <= 1e-8);

			Update_and_Clear();
			Data_Reanalyze(step + 1);
		}

		void MCTS_Start() {
			Init_State();
			//preAction(0);
			//Generate_Move(Step);
			//update resign value
			//if (!Enable_resign) {
			//	int _sz = resign_Sort_Set.size();
			//	if (GameResult == 1)resign_Sort_Set.insert(Min_V_resign[0]), recent_resign.push_back(Min_V_resign[0]);
			//	else if (GameResult == -1)resign_Sort_Set.insert(Min_V_resign[1]), recent_resign.push_back(Min_V_resign[1]);
			//	else if (GameResult == 0) {
			//		double val = min(Min_V_resign[0], Min_V_resign[1]);
			//		resign_Sort_Set.insert(val), recent_resign.push_back(val);
			//	}
			//	else printf("GameResult error\n"), assert(false);
			//	if (_sz != resign_Sort_Set.size() && resign_Sort_Set.size() >= resign_MinSize) {
			//		//pop
			//		if (recent_resign.size() > resign_MaxSize) {
			//			assert(resign_Sort_Set.find(recent_resign.front()) != resign_Sort_Set.end());
			//			resign_Sort_Set.erase(resign_Sort_Set.find(recent_resign.front())), recent_resign.pop_front();
			//		}
			//		const double resign_rate = 0.05;//0.049;
			//		int sz = resign_Sort_Set.size() * resign_rate;
			//		for (auto& k : resign_Sort_Set)if (sz == 0) {
			//			V_resign = k; break;
			//		}
			//		else sz--;
			//	}
			//}
		}



		//action API
		int Ins_PID, Step;// , resultID;
		us Ins_Feature;
		PINFO* volatile Node[MaxGameLength];
		int Thr_Num;
		int SimulationPerMCTS;

		//void addStone(const MCTS_Edge& best_act, const ui& move) {
		//	//addFeature(Ins_Feature, move, Step);
		//	Step++;
		//	//Ins_PID = best_act.next_Node == NULL ? -1 : best_act.next_Node->p.root_ID_exist;
		//	//Node[Step] = best_act.next_Node;
		//	environment_API->Act(move, NULL);
		//}

		void Init_State() {
			Ins_PID = -1;
			Step = 0; Ins_Feature = 0;
			memset((void*)(void**)Node, 0, sizeof(Node));
			AllocNode();
			Min_V_resign[0] = Min_V_resign[1] = 1e9;
			Enable_resign = recent_resign.size() < resign_MinSize ? false : ((rng[0]() % 10) < 7);
			environment_API->Reset(&rng[0]);
			round_number = -1;
			memset(RoundStep, 0, sizeof(RoundStep));
		}
		//do prePolicy action
		//void preAction(int step) {
		//	//select three abandon tile
		//	for (int i = 0; i < Value_Num; i++) {
		//		for (int j = 0; j < 3; j++) {
		//			environment_API->GetExtraPreActionMask(ActionIsValid, i, 0);
		//			int move = -1;
		//			do {
		//				move = rng[0]() % Game_Action_Space;
		//			} while (!ActionIsValid[move]);
		//			assert(move != -1);
		//			//ui move = Search_Policy(i * 5 + j);
		//			environment_API->PreAct(move, 0, 0);
		//		}
		//	}
		//	//swap tiles
		//	environment_API->PreAct(-1, 1, rng[0]() % 3 + 1);
		//	//select banned type
		//	for (int i = 0; i < Value_Num; i++) {
		//		environment_API->GetExtraPreActionMask(ActionIsValid, i, 1);
		//		int move = -1;
		//		do {
		//			move = rng[0]() % Game_Action_Space;
		//		} while (!ActionIsValid[move]);
		//		assert(move != -1);
		//		//ui move = Search_Policy(i * 5 + 3);
		//		environment_API->PreAct(move, 2, 0);
		//		//Terminal_State(i * 5 + 4);
		//	}
		//	//host draw&start game
		//	environment_API->PreAct(-1, 3, 0);

		//	//only using One Player ban Action
		//	//ok
		//	Step = 0;
		//}
		void AllocNode() {
			//for (int i = 1; i < Maximum_Cache; i++)
				//File_Buffer->free_Node[0].splice(File_Buffer->free_Node[0].end(), File_Buffer->free_Node[i]);
			for (int j = 0; j < Thr_Num; j++) {
				//Path[j].clear();
				int cnt = SimulationPerMCTS / Thr_Num + 1 - pre_alloc[j].size();
				//auto it = File_Buffer->free_Node[0].begin(); advance(it, min(cnt, File_Buffer->free_Node[0].size()));
				//pre_alloc[j].splice(pre_alloc[j].end(), File_Buffer->free_Node[0], File_Buffer->free_Node[0].begin(), it);
				//cnt = SimulationPerMCTS / Thr_Num + 1 - pre_alloc[j].size();
				for (int k = 0; k < cnt; k++) {
					auto p = new PINFO();
					pre_alloc[j].emplace_back(p);
				}
			}
		}

		//ui Select_Move(int* result, int CheckID, bool extraMask = false) {
		//	if (environment_API->GetGameState(result))return -1;
		//	if (!extraMask)
		//		environment_API->GetInsActionMask(ActionIsValid, CheckID, true);
		//	//simplify Select Policy
		//	int cnt = 0, _action = -1;
		//	for (int i = 0; i < Game_Action_Space; i++)
		//		if (ActionIsValid[i])_action = i, cnt++;
		//	//Policy make decision
		//	if (cnt > 1) {
		//		{
		//			Finish_Thr = 0;
		//			Eva_CV->notify_one();
		//		}
		//		Step = 0; IsAddNoise = false;
		//		//X Simulation from root
		//		for (int t = 0; t < Thr_Num; t++) {
		//			lock_guard<mutex>locker(Thr_m[t]);
		//			Thr_CV[t].notify_one();
		//		}
		//		{
		//			std::unique_lock<std::mutex> lock(Thr_lock);
		//			Main_CV.wait(lock, [this]() {return Finish_Thr == Thr_Num; });
		//		}
		//		PINFO* Root = Node[0]; float* Value = Node[0]->p.Value;
		//		//Agent sampling move
		//		MCTS_Edge* best_act = NULL; ull tot = 0; double Max_Q = -1e9;
		//		ui move = Sampling_Visit(Root, best_act, Step, tot, &Max_Q, 0);
		//		assert(Max_Q != -1e9 && move != -1);

 	//			Update_and_Clear();
		//		_action = move;
		//	}
		//	//auto default
		//	else assert(cnt == 1);

		//	//host action
		//	if (CheckID == -1) {
		//		environment_API->Act(_action, NULL);
		//		environment_API->PrintScr();
		//	}
		//	return _action;
		//}
		//void Opponent_Move(const int& opponent_move, int CheckID) {
		//	environment_API->GetInsActionMask(ActionIsValid, CheckID, true);
		//	//oppponent move
		//	environment_API->Act(opponent_move, NULL);
		//}
	};
	struct Agent_Group {

		MCTS_Policy_Extend** MP;
		Agent** rollout_Agent = NULL;
		int MP_Count;
		int Thr_Num;

		thread* Eva_Loop = NULL;
		Agent_Group(int MCTS_Policy_Count, MCTS_Policy_Extend** MCTS_Policy, int Thr_Num, Agent** rollout_Agent, int rollout_Num) :MP_Count(MCTS_Policy_Count), MP(MCTS_Policy),
			rollout_Agent(rollout_Agent), Thr_Num(Thr_Num), rollout_Num(rollout_Num) {}
		~Agent_Group() {
			Disponse();
		}
		void run() {
			if (Eva_Loop == NULL)
				Loop = true, Eva_Loop = new thread(&Agent_Group::Evaluation_Loop, this);
		}
		void Disponse() {
			Loop = false;
			{
				lock_guard<mutex> lock(Eva_m);
				Eva_CV.notify_one();
			}
			if (Eva_Loop)Eva_Loop->join(); delete Eva_Loop; Eva_Loop = NULL;
			//free MCTS Policy
			for (int i = 0; i < MP_Count; i++)delete MP[i], MP[i] = NULL;
			//free agent
			//for (int i = 0; i < rollout_Num; i++)delete rollout_Agent[i], rollout_Agent[i] = NULL;
			//delete[] rollout_Agent;
		}

		int rollout_Num;

		list<Evalution*>agent_Q[Max_Thr];
		volatile int agent_Q_Ready[Max_Thr]{ 0 };
		condition_variable agent_CV[Max_Thr], Eva_CV;
		mutex agent_mux[Max_Thr], Eva_m;
		void Agent_Eva(int id) {
			int rollout_Batch = rollout_Agent[id]->Net_Param["Batch"];
			unique_lock<std::mutex>cv_lock(agent_mux[id]);
			Param** data = new Param * [Max_Thr];
			for (int i = 0; i < Max_Thr; i++)data[i] = new Param();
			//int cnt = 0;
			while (true) {
				agent_Q_Ready[id] = -1;
				agent_CV[id].wait(cv_lock, [this, id]() {return agent_Q_Ready[id] != -1; });
				//if (++cnt % 10000 == 0)printf("agent:%d ", id);
				if (agent_Q_Ready[id] == -2)break;
				assert(!agent_Q[id].empty());// && agent_Q[id].size() <= rollout_Batch);
				while (!agent_Q[id].empty()) {
					auto it = agent_Q[id].begin();
					Evalution* tar = NULL; int flag = 0;
					for (int i = 0; i < rollout_Batch; i++) {
						if (it != agent_Q[id].end()) {
							tar = *it; flag = 0;
							it++;
						}//fullfill
						else flag = 3;
						agentResponse(flag, i, tar->H_State, tar->Action, tar->OutPut, tar->Value, tar->Next_State, &tar->Reward, tar->step, rollout_Agent[id], MP[tar->PolicyID]->environment_API, data);
					}
					//run agent
					agentResponse(1, 0, tar->H_State, tar->Action, tar->OutPut, tar->Value, tar->Next_State, &tar->Reward, tar->step, rollout_Agent[id], MP[tar->PolicyID]->environment_API, data);
					//OupPut
					it = agent_Q[id].begin();
					for (int i = 0; i < rollout_Batch; i++) {
						tar = *it;
						agentResponse(2, i, tar->H_State, tar->Action, tar->OutPut, tar->Value, tar->Next_State, &tar->Reward, tar->step, rollout_Agent[id], MP[tar->PolicyID]->environment_API, data);
						it++;
						if (it == agent_Q[id].end())break;
					}
					agentResponse(4, 0, tar->H_State, tar->Action, tar->OutPut, tar->Value, tar->Next_State, &tar->Reward, tar->step, rollout_Agent[id], MP[tar->PolicyID]->environment_API, data);
					agent_Q[id].erase(agent_Q[id].begin(), it);
				}
			}
			//free cuda stream
			unBindStm();
			for (int i = 0; i < Max_Thr; i++)delete data[i];
			delete[] data;
		}

		bool Loop = true;
		void Evaluation_Loop() {
			thread* rollout_Loop[Max_Thr];
			int Thr_cnt = 0, yield_cnt = 0, remain_Thr = 0;
			list<Evalution*>Q; Q.clear();
			for (int i = 0; i < rollout_Num; i++)
				rollout_Loop[i] = new thread(&Agent_Group::Agent_Eva, this, i);
			std::unique_lock<std::mutex> Eva_lock(Eva_m);
			int sum_Node = 0, sum_Batch = 0;
			int cnt = 0;
			while (cnt < MP_Count) {
				cnt = 0; for (int i = 0; i < MP_Count; i++)if (MP[i] != NULL)cnt++, MP[i]->Eva_CV = &Eva_CV, MP[i]->Eva_m = &Eva_m;
				this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			//pipeOut("DEBUG EvaLoop Start");
			int Mini_Batch = rollout_Agent[rollout_Num - 1]->Net_Param["Batch"];
			while (Loop) {
				remain_Thr = yield_cnt = 0;
				for (int t = 0; t < MP_Count; t++) {
					auto tar = MP[t];
					for (int i = 0; i < Thr_Num; i++) {
						if (tar->Response[i] != 1) {
							if (tar->Response[i] == 0) {
								Q.push_back(&(tar->Eva_Stack[i]));
								tar->Response[i] = 3;
							}
							//yield
							else if (tar->Response[i] == 2) {
								yield_cnt++;
							}
							//return back value
							else if (tar->Response[i] == 3 && -2 < tar->Eva_Stack[i].Value[Value_Num-1]) {
								auto& it = tar->Eva_Stack[i];
								assert(it.ID == i);
								assert(abs(it.Value[0]) <= 1 + 1e-8);
								lock_guard<mutex> lock(tar->Response_mux[it.ID]);
								tar->Response[it.ID] = 1;
								tar->Response_CV[it.ID].notify_one();
							}
						}
					}
					remain_Thr += Thr_Num - tar->Finish_Thr;
				}
				//Agent evalution
				int Batch_Size = min(remain_Thr, Mini_Batch);
				if (!Q.empty() && Q.size() + yield_cnt >= Batch_Size) {
					auto it = Q.begin();
					int rep = -1, dyn = -1;
					bool rep_flag = false, dyn_flag = false;
					for (int j = 0; j < rollout_Num; j++)
						if (j >= 1 && agent_Q_Ready[j] == -1)dyn = j;
						else if (j < 1 && agent_Q_Ready[j] == -1)rep = j;
					for (int k = 0; k < Batch_Size; k++) {
						if (it == Q.end())break;
						//representation
						if ((*it)->H_State == NULL && rep != -1)
							agent_Q[rep].push_back(*it), rep_flag = true, it = Q.erase(it);
						//dynamic
						else if ((*it)->H_State && dyn != -1) {
							agent_Q[dyn].push_back(*it), dyn_flag = true, it = Q.erase(it);
						}
						//it++; sum_Node++;
					}
					//wake up agent
					for (int j = 0; j < rollout_Num; j++) {
						if ((rep_flag && j == rep) || (dyn_flag && j == dyn)) {
							lock_guard<mutex> lock(agent_mux[j]);
							agent_Q_Ready[j] = 1;
							agent_CV[j].notify_one();
							sum_Batch++;
						}
					}
				}
				//wait
				if (remain_Thr == 0) {
					Eva_CV.wait(Eva_lock);
				}
			}
			//End Loop
			for (int i = 0; i < rollout_Num; i++) {
				{
					lock_guard<mutex> lock(agent_mux[i]);
					agent_Q_Ready[i] = -2;
					agent_CV[i].notify_one();
				}
				rollout_Loop[i]->join(), delete rollout_Loop[i];
			}
		}
	};


	struct Agent_Extend :public Agent_API {
		MCTS_Policy_Extend* MCTS_Agent = NULL;
		Agent_Group* Group0 = NULL;
		HyperParamSearcher param;
		void MCTS_Agent_Init(const char* agent_Path, const char* agent_param_Path, const char* dir_path, Environment& e) {
			pipeOut("DEBUG MCTS Clear\n");
			std::experimental::filesystem::remove_all(((string)dir_path + "MCTS\\0\\").c_str());
			pipeOut("DEBUG MCTS Init\n");
			Init_MCTS(e, dir_path);
			param.Read_Param((string)agent_param_Path);
			if (!MCTS_Agent) {
				Agent* agent;
				if (param["Init_Agent"])
					agent = new Agent(Agent::RNN, param["Batch"], e.JointNet(param), param["Max_Step"], true, time(0), param["Max_srand_row"]);
				else agent = new Agent(agent_Path, param["Batch"], param["Max_Step"], param["Max_srand_row"], true);
				const int rollout_Num = 3, rollout_Batch = 8;
				Agent** rollout = new Agent * [rollout_Num] { 0 };
				param["Batch"] = rollout_Batch; param["Max_Step"] = 1;
				for (int i = 0; i < rollout_Num; i++) {
					Net* net = i == 0 ? e.RepresentationNet(param) : e.DynamicsNet(param);
					rollout[i] = new Agent(Agent::RNN, param["Batch"], net, param["Max_Step"], false, time(0), param["Max_srand_row"]);
					rollout[i]->Online_Init(param);
				}
				e.JoinetNet_Assignment(rollout, agent, rollout_Num);
				MCTS_Agent = new MCTS_Policy_Extend(0, &e, param["Thr_Num"], param["Simulations"], "0", ((string)dir_path + "MCTS\\").c_str());
				pipeOut("DEBUG MCTS Create success\n");
				//GPU
				Group0 = new Agent_Group(1, &MCTS_Agent, MCTS_Agent->Thr_Num, rollout, rollout_Num);
				Group0->run();
				pipeOut("DEBUG Agent is running successfully\n");
				delete agent;
			}
		}
		//bot run on tenhou.net
		void MCTS_Agent_Run() {
			MCTS_Agent->Init_State();
			MCTS_Policy_Extend* ag = MCTS_Agent;
			vector<Environment::State_Func>Funcs({ ag->getStateFunc(), ag->getMoveStateFunc(),ag->getSetStateMoveFunc() ,ag->getInitRoundFunc(),
				ag->getTerminalStateFunc(),ag->getTerminalStateFunc() });
			MCTS_Agent->environment_API->Environment_Loop({ ag->getStochasticPolicyFunc() }, ag->getSetGameResultFunc(), Funcs, NULL, NULL);
		}
		string move_track[MaxGameLength];
		//string MCTS_Agent_Response(string* moves, int moves_cnt) {
		//	pipeOut("DEBUG response");
		//	//detect takeback
		//	for (int i = 0; i < MCTS_Agent->Step; i++) {
		//		//change moves
		//		if (i >= moves_cnt || moves[i] != move_track[i]) {
		//			pipeOut("DEBUG take back");
		//			//MCTS_Agent->TakeBack(i, moves);
		//			MCTS_Agent_New_Game();
		//			break;
		//		}
		//	}
		//	DEBUG(moves_cnt >= MCTS_Agent->Step);
		//	while (moves_cnt > MCTS_Agent->Step) {
		//		pipeOut("DEBUG do %s", moves[MCTS_Agent->Step]);
		//		int mov = MCTS_Agent->environment_API->parse_action(moves[MCTS_Agent->Step]);
		//		move_track[MCTS_Agent->Step] = moves[MCTS_Agent->Step];
		//		MCTS_Agent->Opponent_Move_without_IO(mov);
		//	}

		//	int move = -1, result = 2;
		//	MCTS_Agent->Select_Move_without_IO(move, result);
		//	move_track[MCTS_Agent->Step - 1] = MCTS_Agent->environment_API->Encode_action(move);
		//	return move_track[MCTS_Agent->Step - 1];
		//}
		void MCTS_Agent_New_Game() {
			pipeOut("DEBUG MCTS New Game");
			//wait for modification
			//MCTS_Agent->Update_and_Write();
			MCTS_Agent->Init_State();
			pipeOut("DEBUG MCTS End Write");
		}
		void MCTS_Agent_End() {
			pipeOut("DEBUG MCTS End\n");
			//MCTS_Agent->Update_and_Write();
			//MCTS_Agent->Write2File();
			MCTS_Agent->remove_dir();
			//free agent
			for (int i = 0; i < Group0->rollout_Num; i++)delete Group0->rollout_Agent[i], Group0->rollout_Agent[i] = NULL;
			delete[] Group0->rollout_Agent;

			delete Group0;
			//delete MCTS_Agent;
		}
	};

	int tot_data_set_Count, tot_train_set_Count;
	void Go_Param(HyperParamSearcher& param) {
		param["paramNum"] = tot_data_set_Count;
		param["trainNum"] = tot_train_set_Count;
		param["ExtraTrainFactorPerEpoch"] = 1;
		param["ExtraTestFactorPerEpoch"] = 1;
		param["Stochastic"] = 1;
		param["EpochUpdate"] = 0;
		param["Max_Epoch"] = 1;
		param["EarlyStop"] = 0;
	}
	mutex ds_lock, rollout_lock;
	const int Max_Data_Size = 8 * 8 * 15 * 15 * 40;// *2;
	//bool Write_New_Data = false;
	void MCTS_generate_data_extend(int tid, MCTS_Policy_Extend* ag, DataSet& ds, const char* data_Path, int GamesPerAgent, int* com) {
		ll tot_step = 0, game_count = 0;
		clock_t start_time = clock();
		int Count = 0;
		while (true) {
			//execute MCTS,generate train data
			ag->MCTS_Start();
			
			//check valid
			bool pass = false;
			for (int i = 0; i < 4; i++)
				if (ag->GameResult[i] != 0)pass = true;

			if (pass || 1.0 * Count / (ds.gameCount + 1) >= 0.0) {
				if (pass)Count++;
				//train RL agent
				int MaxStep = ds.trainSet_dataCount() + ag->GameStep;
				int lastID = -1, StartID = ds.trainSet_dataCount();
				for (int i = 0; i < ag->GameStep; i++) {
					if (!ag->trainData[i].passTrain)
						lastID = ds.trainSet_dataCount();
					//repeat terminal state
					ds.trainSet_Add_data(Environment::Data(lastID, ag->PlayerIDOrder, StartID, MaxStep, Screen_Size, ag->trainData[i].Scr, ag->environment_API->getSimplifyActionSpace(), ag->trainData[i].OutPut, ag->trainData[i].out_cnt, ag->trainData[i].move, /*ag->GameResult*/NULL, Value_Num, ag->trainData[i].Action_Reward, *ag->environment_API));
				}
				ds.gameCount++;
				tot_step += ag->GameStep, game_count++;
				//if (!((ag->GameResult == 1 && ag->GameStep % 2 == 0) || (ag->GameResult == -1 && ag->GameStep % 2 == 1) || ag->GameResult == 0))printf("error:result not match step\n"), assert(false);
				printf("tid:%d step:%d result:%d %d %d %d average_step:%.02lf tot_step:%d Count:%d time:%d min(s)\n", tid, ag->GameStep, ag->GameResult[0], ag->GameResult[1], ag->GameResult[2], ag->GameResult[3],
					1.0 * tot_step / game_count, tot_step, ds.trainSet_gameCount(), (clock() - start_time) / CLOCKS_PER_SEC / 60);

				//Batch
				char ch = 0; bool save = false; int idx = 0;
				while (com[idx++] != -1) {
					if (com[idx - 1] > com[tid])save = true;
				}
				if (ds.trainSet_gameCount() % GamesPerAgent == 0 || (_kbhit() && (ch = _getch()) == 'l') || save) {
					ag->Write2File();
					//absorbing
					if (ds.trainSet_Param(0).In(0).MaxCount == 0) {
						ds.complete_absorbing_Head();
					}
					ds.trainSet_Save_Load(true, -1, data_Path);
					if (ch == 'l' || save)com[tid]++;
					break;
				}
			}
		}
	}
	//Python environment call MCTS Policy API to generate action
	void MCTS_generate_data_called_by_Environment(int tid, MCTS_Policy_Extend* ag, DataSet& ds, DataSet& reward_ds,const char* data_Path, int GamesPerAgent, int* com) {
		//env ==Game Info==>MCTS Policy
		//env <==Action Policy==MCTS Policy
		//env ==final select Action==>MCTS Policy==>Record action

		//env ==result==>MCTS Policy
		multiset<double>plus_rewardSet, minus_rewardSet;
		ll tot_step = 0, game_count = 0;
		clock_t start_time = clock();
		int Count = 0, ds_Count = 0, tot_ds_Count = 0;

		double pass_percent = 0.8;
		Count = ds.gameCount * pass_percent;
		ds_Count = ds.gameCount * pass_percent;
		tot_ds_Count = ds.gameCount;


		auto* env = ag->environment_API;
		while (true) {
			int reward_ds_old_count = reward_ds.dataCount;
			//execute MCTS,generate train data
			ag->Init_State();
			vector<Environment::State_Func>Funcs({ ag->getStateFunc(), ag->getMoveStateFunc(),ag->getSetStateMoveFunc() ,ag->getInitRoundFunc(),
				ag->getTerminalStateFunc(),ag->getTerminalStateFunc()});
			env->Environment_Loop({ ag->getPolicyFunc() }, ag->getSetGameResultFunc(), Funcs, NULL, &reward_ds);

			assert(ag->round_number + 1 == reward_ds.dataCount - reward_ds_old_count);
			//check valid
			

			//modify each round Reward regard to reward predictor
			auto rounds_reward = env->getRoundsReward();
			//round score to [-1,1]

			bool pass = false;
			/*for (int i = 0; i < 4; i++)
				if (ag->GameResult[i] != 25000)pass = true;*/
			if (!rounds_reward[0].Y)pass = true;

			if (pass || 1.0 * Count / (ds.gameCount + 1) >= 0.8) {
				if (pass)Count++;
				//train RL agent
				for (int t = 0; t < ag->round_number + 1; t++) {
					double round_reward[4] = { 0,0,0,0 };
					//bootstrapped to n=10 step
					for (int i = 0; i < 4; i++) {
						//(-1,1)
						/*round_reward[i] = rounds_reward[t].X[i * 2 + 1] - (t == 0 ? 0.5 : rounds_reward[t - 1].X[i * 2 + 1]);
						round_reward[i] /= 0.4;
						round_reward[i] = min(1.0, round_reward[i]);
						round_reward[i] = max(-1.0, round_reward[i]);*/
						if (ag->GameResult[i] > 25000)round_reward[i] = 1;
						else if (ag->GameResult[i] < 25000)round_reward[i] = -1;

						//ignore draw round
						//all round_reward set to 0

						//auto win action in the future?
					}
					//filter meanless round
					pass = false; int cnt = 0;
					//for (int i = 0; i < 4; i++) {
					//	if (abs(round_reward[i]) > 0.5)cnt++;
					//	//round_reward[i] = ag->GameResult[i];
					//}
					if (!rounds_reward[t].Y)pass = true;

					if (pass || 1.0 * ds_Count / (tot_ds_Count + 1) >= 0.8) {
						if (pass)ds_Count++;
						tot_ds_Count++;
						int MaxStep = ds.trainSet_dataCount() + ag->RoundStep[t];
						int lastID = -1, StartID = ds.trainSet_dataCount();
						for (int i = 0; i < ag->RoundStep[t]; i++) {
							if (!ag->GameData[t][i].passTrain)
								lastID = ds.trainSet_dataCount();
							//repeat terminal state
							ds.trainSet_Add_data(Environment::Data(lastID, ag->GameOrder[t], StartID, MaxStep, Screen_Size, ag->GameData[t][i].Scr, trajectory_Data::traj_Output_Size, ag->GameData[t][i].OutPut, ag->GameData[t][i].out_cnt, ag->GameData[t][i].move, round_reward, Value_Num, ag->GameData[t][i].Action_Reward, *ag->environment_API));
						}
					}
				}

				ds.gameCount++;
				tot_step += ag->round_number + 1, game_count++;
				//if (!((ag->GameResult == 1 && ag->GameStep % 2 == 0) || (ag->GameResult == -1 && ag->GameStep % 2 == 1) || ag->GameResult == 0))printf("error:result not match step\n"), assert(false);
				printf("tid:%d round_number:%d result:%d %d %d %d step_r0:%d average_rounds:%.02lf tot_rounds:%d Count:%d time:%d min(s)\n", tid, ag->round_number + 1, ag->GameResult[0], ag->GameResult[1], ag->GameResult[2], ag->GameResult[3],
					ag->RoundStep[0],1.0 * tot_step / game_count, tot_step, ds.trainSet_gameCount(), (clock() - start_time) / CLOCKS_PER_SEC / 60);

				//Batch
				char ch = 0; bool save = false; int idx = 0;
				while (com[idx++] != -1) {
					if (com[idx - 1] > com[tid])save = true;
				}
				if (ds.trainSet_gameCount() % GamesPerAgent == 0 || (_kbhit() && (ch = _getch()) == 'l') || save) {
					ag->Write2File();
					//absorbing
					if (ds.trainSet_Param(0).In(0).MaxCount == 0) {
						ds.complete_absorbing_Head();
					}
					ds.trainSet_Save_Load(true, -1, data_Path);
					reward_ds.trainSet_Save_Load(true, -1, ((string)data_Path + "_reward").c_str());
					if (ch == 'l' || save)com[tid]++;
					break;
				}
			}
		}
	}
	//void MCTS_gererate_randomize_data(int tid, MCTS_Policy_Extend* ag, DataSet& ds, DataSet& new_ds, int thread_cnt) {
	//	clock_t start_time = clock();
	//	int ds_idx = 3, game_id = 0;
	//	trajectory_Data tra;
	//	while (ds_idx < ds.dataCount) {
	//		//get traj data
	//		int data_len = 0, start_idx = ds_idx;
	//		{
	//			//Screen
	//			memcpy(tra.Scr, &ds.trainSet_Param(ds_idx).Out(0)[0], sizeof(double) * Screen_Size);
	//			//Policy
	//			int cnt = ds.trainSet_Param(ds_idx).Out(0).Count - Screen_Size - 2;
	//			assert(cnt > 0);
	//			memcpy(tra.OutPut, &ds.trainSet_Param(ds_idx).Out(0)[Screen_Size + 2], sizeof(double) * cnt);
	//			tra.out_cnt = cnt;
	//			//terminal state
	//			if (cnt == 1) {
	//				assert(tra.OutPut[0] == -1);
	//				data_len = -2;
	//			}
	//			//move
	//			else tra.move = ds.trainSet_Param(ds_idx + 1).Out(0)[Screen_Size];

	//			data_len++; ds_idx++;
	//		}
	//		if (game_id++ % thread_cnt == tid && data_len != -1) {
	//			//run lastest agent
	//			ag->Init_State();
	//			ag->traj_data_Randomize(tra, 0);

	//			//int MaxStep = ds.trainSet_dataCount() + ag->RoundStep[t];
	//			//int lastID = -1, StartID = ds.trainSet_dataCount();
	//			//for (int i = 0; i < ag->RoundStep[t]; i++) {
	//			//	if (!ag->GameData[t][i].passTrain)
	//			//		lastID = ds.trainSet_dataCount();
	//			//	//repeat terminal state
	//			//	ds.trainSet_Add_data(Environment::Data(lastID, ag->GameOrder[t], StartID, MaxStep, Screen_Size, ag->GameData[t][i].Scr, trajectory_Data::traj_Output_Size, ag->GameData[t][i].OutPut, ag->GameData[t][i].out_cnt, ag->GameData[t][i].move, round_reward, Value_Num, ag->GameData[t][i].Action_Reward, *ag->environment_API));
	//			//}

	//			//update Policy
	//			/*for (int i = 0; i < data_len; i++) {
	//				trajectory_Data& tra = ag->trainData[i];
	//				memcpy(&ds.trainSet_Param(start_idx + i).Out(0)[Screen_Size + 3], tra.OutPut, sizeof(double) * ag->environment_API->getSimplifyActionSpace());
	//			}
	//			printf("tid:%d game_id:%d time:%d min(s)\n", tid, game_id - 1, (clock() - start_time) / CLOCKS_PER_SEC / 60);*/
	//		}
	//	}
	//}
	void MCTS_Reanalyze_Data(int tid, MCTS_Policy_Extend* ag, DataSet& ds, int gen_cnt) {
		clock_t start_time = clock();
		int ds_idx = 3, game_id = 0;
		while (ds_idx < ds.dataCount) {
			int data_len = 0, start_idx = ds_idx;
			while (true) {
				trajectory_Data& tra = ag->trainData[data_len];
				//Screen
				memcpy(tra.Scr, &ds.trainSet_Param(ds_idx).Out(0)[0], sizeof(double) * Screen_Size);
				//Policy
				int cnt = ds.trainSet_Param(ds_idx).Out(0).Count - Screen_Size - 3;
				assert(cnt > 0);
				memcpy(tra.OutPut, &ds.trainSet_Param(ds_idx).Out(0)[Screen_Size + 3], sizeof(double) * cnt);
				tra.out_cnt = cnt;

				data_len++; ds_idx++;
				if (cnt == 1)break;
			}
			if (game_id++ % gen_cnt == tid) {
				//run lastest agent
				ag->Init_State();
				ag->Data_Reanalyze(0);

				//update Policy
				for (int i = 0; i < data_len; i++) {
					trajectory_Data& tra = ag->trainData[i];
					memcpy(&ds.trainSet_Param(start_idx + i).Out(0)[Screen_Size + 3], tra.OutPut, sizeof(double) * ag->environment_API->getSimplifyActionSpace());
				}
				printf("tid:%d game_id:%d time:%d min(s)\n", tid, game_id - 1, (clock() - start_time) / CLOCKS_PER_SEC / 60);
			}
		}
	}

	/*void Opponents_Move(MCTS_Policy_Extend** P, MCTS_Policy_Extend* InsPlayer, const ui& move) {
		for (int i = 0; i < 4; i++)
			if (P[i] != InsPlayer)
				P[i]->Opponent_Move(move, -1);
			else P[i]->environment_API->GetInsActionMask(P[i]->ActionIsValid, 0, true);
	}*/


	//void MCTS_Match(MCTS_Policy_Extend**P, int* result) {
	//	time_t seed = time(0);
	//	//reset state
	//	for (int i = 0; i < 4; i++) {
	//		P[i]->Init_State();
	//		//all same environment state
	//		mt19937 rng(seed);
	//		P[i]->environment_API->Reset(&rng);
	//	}
	//	mt19937 rng(time(0));
	//	//select three abandon tile
	//	for (int i = 0; i < Value_Num; i++) {
	//		for (int k = 0; k < 3; k++) {
	//			P[i]->environment_API->GetExtraPreActionMask(P[i]->ActionIsValid, i, 0);
	//			int move = -1;
	//			do {
	//				move = rng() % Game_Action_Space;
	//			} while (!P[i]->ActionIsValid[move]);
	//			assert(move != -1);
	//			//ui move = P[i]->Select_Move(result, 0, true);
	//			P[i]->environment_API->PreAct(move, 0, 0);
	//			for (int j = 0; j < 4; j++)
	//				if (j != i) {
	//					P[j]->environment_API->GetExtraPreActionMask(P[j]->ActionIsValid, i, 0);
	//					P[j]->environment_API->PreAct(move, 0, 0);
	//				}
	//		}
	//	}
	//	//swap tiles
	//	int shift = rng() % 3 + 1;
	//	for (int i = 0; i < 4; i++)
	//		P[i]->environment_API->PreAct(-1, 1, shift);
	//	//select banned type
	//	for (int i = 0; i < 4; i++) {
	//		P[i]->environment_API->GetExtraPreActionMask(P[i]->ActionIsValid, i, 1);
	//		//random choice banned to increase explore??
	//		int move = -1;
	//		do {
	//			move = rng() % Game_Action_Space;
	//		} while (!P[i]->ActionIsValid[move]);
	//		assert(move != -1);
	//		//ui move = P[i]->Select_Move(result, 0, true);
	//		P[i]->environment_API->PreAct(move, 2, 0);
	//		for(int  j=0;j<4;j++)
	//			if (j != i) {
	//				P[j]->environment_API->GetExtraPreActionMask(P[j]->ActionIsValid, i, 1);
	//				P[j]->environment_API->PreAct(move, 2, 0);
	//			}
	//	}
	//	//host draw&start game
	//	for (int i = 0; i < 4; i++)
	//		P[i]->environment_API->PreAct(-1, 3, 0);

	//	int pid = 0;
	//	while (true) {
	//		auto* ply = P[pid];
	//		//InsPlayer Action
	//		ui move = ply->Select_Move(result, -1);
	//		if (move == -1)return;

	//		Opponents_Move(P, ply, move);

	//		int opponent_move[3] = { 0 }, otherAllPass = 0;
	//		for (int i = 0; i < 3; i++) {
	//			auto* oppo = P[(pid + 1 + i) % 4];
	//			opponent_move[i] = oppo->Select_Move(result, i);
	//			//not pass
	//			if (opponent_move[i] != Game_Action_Space - 1)otherAllPass = max(otherAllPass, opponent_move[i]);
	//		}

	//		for (int i = 0; i < 4; i++)
	//			pid = P[i]->environment_API->OpponentsAct(opponent_move);
	//		ply->environment_API->PrintScr();
	//	}
	//}
	void Sampling_DataSet(DataSet& train_ds, int data_Count, int recent_Max_data, int dataSet_Max_ID, Environment* e) {
		train_ds.trainSet_Init(data_Count * e->getMaxUnrolledStep());
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
				ds.trainSet_Save_Load(false, Max_Data_Size, Path);
				Start += ds.dataCount;
				dataSet_Max_ID--;
			}
			for (int j = 0; j < ds.trainSet_Param(Start).Count; j++) {
				train_ds.trainSet_Add_data(Environment::Data(e));
				train_ds.trainSet_Param(train_ds.gameCount).In(j) = ds.trainSet_Param(Start).In(j);
				train_ds.trainSet_Param(train_ds.gameCount).In(j)[0] = j + train_ds.gameCount * e->getMaxUnrolledStep();
				train_ds.trainSet_Param(train_ds.dataCount - 1).Out(0) = ds.trainSet_Param(ds.trainSet_Param(Start).In(j)[0]).Out(0);
			}
			train_ds.gameCount++;
		}
		tot_data_set_Count = train_ds.trainSet_dataCount();
		tot_train_set_Count = train_ds.gameCount;
		//train_ds.trainSet_Save_Load(true, -1, "trainSet_data_sampling");
	}
}
using namespace multiMCTS_extend;
Agent_API* Get_multiMCTS_Extend_API() {
	return new Agent_Extend();
}

//Player data test
	//DataSet _player_ds;
	//_player_ds.trainSet_Save_Load(false, 2673103, "Player_DataSet");
	////test player data
	//if (&_player_ds)
	//	cout << _player_ds.Test(agent, _player_ds.trainSet_dataCount()/5, Gomoku::MCTS_GomokuSimulation, TestScore) << endl;
const double D = 100, K = 50;
void update_Elo(double* rating,double*expected,double*score) {
	for (int i = 0; i < 4; i++)
		rating[i] += K * (score[i] - expected[i]);
}
void estimated_scores(double*result,double*rating) {
	double sum = 0;
	for (int i = 0; i < 4; i++) {
		result[i] = 0;
		for (int j = 0; j < 4; j++) {
			if (i == j)continue;
			result[i] += 1.0 / (1.0 + pow(10, (rating[j] - rating[i]) / D));
		}
		sum += result[i] /= (4 * 3 / 2);
	}
	assert(sum == 1);
}
void score(int*result,double*score) {
	int a = 0, b = 0;
	for (int i = 0; i < 4; i++)if (result[i] > 0)a += result[i];
	for (int i = 0; i < 4; i++)if (result[i] < 0)b += result[i];
	for (int i = 0; i < 4; i++)if (result[i] == 0)score[i];
}
struct PolicyAgent {
	int PolicyID;
	Environment* environment_API;

	PolicyAgent(int ID,Environment*env):PolicyID(ID),environment_API(env) {}
	inline Environment::Policy_Func getPolicyFunc() {
		return std::bind(&PolicyAgent::Search_Policy, this, _1, _2, _3);
	}
	vector<int> Search_Policy(int step, int sampling_step, vector<int> action_mask) {
		auto policy = environment_API->getAgentPolicy(PolicyID)[0];
		double mx = -1; int idx = -1;
		map<int, int>mask;
		for (auto& action : action_mask) {
			int type = action / 34, tile = action % 34;
			if (type < 48)type %= 3;
			else type = type - 48 + 3;
			int act = tile * 12 + type;
			mask[act] = action;
		}
		assert(mask.size() == action_mask.size());
		for (int i = 0; i < policy.size(); i++) {
			if (mask.find(i) != mask.end() && policy[i] > mx) {
				mx = policy[i], idx = i;
			}
		}

		for (auto& k : action_mask)
			//auto win
			if (k / 34 == 54) {
				return { k };
			}
		    //auto draw
			else if (k / 34 == 55) {
				return { k };
			}

		for (auto& k : action_mask)
			if (k / 34 < 48 && k / 34 % 3 == 1) {
				return { k };
			}

		return { mask[idx] };
	}
};
void multiMCTS_Evaluation(int rollout_Num, Agent*** rollout_agent, Environment**e) {
	Init_MCTS(*e[0], "");
	std::experimental::filesystem::remove_all(((string)"MCTS_Policy\\1\\").c_str());
	std::experimental::filesystem::remove_all(((string)"MCTS_Policy\\2\\").c_str());
	std::experimental::filesystem::remove_all(((string)"MCTS_Policy\\3\\").c_str());
	std::experimental::filesystem::remove_all(((string)"MCTS_Policy\\4\\").c_str());
	MCTS_Policy_Extend* Player[4];
	Agent_Group* Group[4];
	for (int i = 0; i < 4; i++) {
		//CPU
		//only one real env in match 
		Player[i] = new MCTS_Policy_Extend(0, e[0], 25, 500, string(1, '1' + i).c_str());
		//GPU
		Group[i] = new Agent_Group(1, &Player[i], Player[i]->Thr_Num, rollout_agent[i], rollout_Num);
		Group[i]->run();
	}

	//Policy Agent
	PolicyAgent /*PA(1, e[0]), */PA(0, e[0]);


	
	mt19937 rng(time(0));
	int result[4], finalResult[4] = { 0 };
	int tosumo[4] = { 0 }, loss[4] = { 0 }, riichi[4] = { 0 }, win[4] = { 0 };
	double Rating[4] = { 100,100,100,100 };
	for (int t = 0; t < 100; t++) {
		for (int i = 0; i < 4; i++)Player[i]->Init_State();
		//random player sitting
		vector<Environment::Policy_Func>Policys;
		int id[4] = { 0,1,2,3 }, ord[4];
		for (int i = 0; i < 4; i++)ord[i] = rng();
		sort(id, id + 4, [&](const int& a, const int& b) {return ord[a] < ord[b]; });

		//for (int i = 0; i < 4; i++)if (id[i] == 1)extra_data = i;

		for (int i = 0; i < 4; i++)Policys.push_back/*(id[i] == 1 ? NULL : PA.getPolicyFunc());*/(/*id[i] == 1 ? PA1.getPolicyFunc() : */Player[id[i]]->getPolicyFunc());
		e[0]->Environment_Loop(Policys, NULL, {}, result, NULL);
		for (int i = 0; i < 4; i++)if (result[i] & 1)riichi[id[i]]++, result[i] -= 1;
		int mx = -1, mi = 25000, loser = 0;
		for (int i = 0; i < 4; i++)result[i] -= 25000, mx = max(result[i], mx), mi = min(result[i], mi);
		for (int i = 0; i < 4; i++)if (result[i] < 0)loser++;
		for (int i = 0; i < 4; i++)if (result[i] < 0 && result[i] == mi && loser != 3)loss[id[i]]++;
		for (int i = 0; i < 4; i++)if (result[i] > 0 && loser == 3)tosumo[id[i]]++;
		assert(mx >= 0 && mi <= 0);
		//MCTS_Match(Player, result);
		printf("result: ");
		for (int i = 0; i < 4; i++)printf("%d ", result[i]), finalResult[id[i]] += result[i] == mx ? 1 : 0, win[id[i]] += (result[i] == mx&& loser>0) ? 1 : 0;
		for (int i = 0; i < 4; i++)finalResult[id[i]] += result[i] == mi ? -1 : 0;
		printf("\nt:%d ",t);
		for (int i = 0; i < 4; i++)printf("%d ", finalResult[i]);
		printf("\n");
		for (int i = 0; i < 4; i++)printf("%d ", win[i]);
		printf("\n");
		for (int i = 0; i < 4; i++)printf("%d ", tosumo[i]);
		printf("\n");
		for (int i = 0; i < 4; i++)printf("%d ", loss[i]);
		printf("\n");
		for (int i = 0; i < 4; i++)printf("%d ", riichi[i]);
		printf("\n");
		double estimated[4];
		//estimated_scores(estimated, Rating);
	}

	for (int i = 0; i < 4; i++)delete Group[i];
	getchar();
	return;
}
const int Generator_Thr = 25;
void multiGenerator_Proc(int tid, int Maximum_DataSet_Number, Environment* e, MCTS_Policy_Extend*& Main_ag, int* com) {
	DataSet ds, reward_ds;
	char path[100]; sprintf(path, "extend_trainSet_data_thr%d", tid);
	ds.trainSet_Save_Load(false, Max_Data_Size, path);
	//add absorbing state first
	if (ds.dataCount == 0) {
		ds.add_absorbing_Head(*e, trajectory_Data::traj_Output_Size, 0);
	}
	//reward ds
	reward_ds.trainSet_Save_Load(false, Max_Data_Size, ((string)path + "_reward").c_str());
	
	//thread*MCTS = new thread(&MCTS_generate_data, Main_ag, ref(ds), ref(train));
	char name[50]; sprintf(name, "thr_%d", tid);
	Main_ag = new MCTS_Policy_Extend(tid, e, Generator_Thr, 500, name, "D:\\multiMCTS_Policy\\");
	while (!Main_ag->Eva_CV)this_thread::sleep_for(std::chrono::milliseconds(100));
	for (int t = 0; t < 1000; t++) {
		MCTS_generate_data_called_by_Environment(tid, Main_ag, ds, reward_ds, path, 50, com);
		//MCTS_generate_data_extend(tid, Main_ag, ds, path, 500, com);
	}
	ds.Disponse();
	//free cuda stream
	//unBindStm();
}
//void Data_Reanalyze_Proc(int tid, DataSet& ds, Environment* e, MCTS_Policy_Extend*& Main_ag, int gen_cnt) {
//	char name[50]; sprintf(name, "thr_%d", tid);
//	Main_ag = new MCTS_Policy_Extend(tid, e, Generator_Thr, 800, name, "D:\\MCTS_Policy\\");
//	while (!Main_ag->Eva_CV)this_thread::sleep_for(std::chrono::milliseconds(100));
//	MCTS_Reanalyze_Data(tid, Main_ag, ds, gen_cnt);
//}
//void Combine_MP_ds(DataSet& ds, DataSet& ds1, Environment& e) {
//	int offset = ds.dataCount;
//	assert(ds.dataCount >= 3 && ds1.dataCount >= 3);
//	for (int i = 3; i < ds1.dataCount; i++) {
//		ds.trainSet_Add_data(Environment::Data(&e));
//		for (int j = 0; j < e.getMaxUnrolledStep(); j++)
//			if (ds1.trainSet_Param(i).In(j)[0] >= 3)ds1.trainSet_Param(i).In(j)[0] += offset - 3;
//		ds.trainSet_Param(ds.dataCount - 1) = ds1.trainSet_Param(i);
//	}
//	ds.gameCount += ds1.gameCount;
//}
//DataSet* Combine(int l, int r, Environment& e) {
//	int M = l + r >> 1;
//	if (l >= r)return NULL;
//	if (l + 1 == r) {
//		char path[100]; sprintf(path, "extend_trainSet_data_thr%d", l);
//		DataSet* ds = new DataSet(); ds->trainSet_Save_Load(false, Max_Data_Size, path);
//		return ds;
//	}
//	auto* left = Combine(l, M, e);
//	auto* right = Combine(M, r, e);
//	if (!left) {
//		assert(right); return right;
//	}
//	else {
//		assert(left && right);
//		if (left->dataCount > 3 && right->dataCount > 3)
//			Combine_MP_ds(*left, *right, e);
//		delete right;
//		return left;
//	}
//}
void RL_SelfPlay_with_multiMCTS_extend(int rollout_Num, Agent** rollout_agent, int agent_id, int Maximum_DataSet_Number, Environment** e, int Generator_Count, bool Reanalyze) {

	Init_MCTS(*e[0], "");
	int Agents_Count = agent_id;
	while (true)
	{
		//multi CPU
		thread* thr[Max_Thr];
		MCTS_Policy_Extend** MP = new MCTS_Policy_Extend * [Generator_Count] {NULL};
		int communicate[Max_Thr + 1]; fill(communicate, communicate + Max_Thr + 1, -1);
		//GPU
		Agent_Group* Group = new Agent_Group(Generator_Count, MP, Generator_Thr, rollout_agent, rollout_Num);
		Group->run();
		if (!Reanalyze)
			for (int t = 0; t < Generator_Count; t++) {
				communicate[t] = 0;
				thr[t] = new thread(&multiGenerator_Proc, t, Maximum_DataSet_Number, ref(e[t]), ref(MP[t]), communicate);
			}
		else {
			DataSet ds;
			char path[100]; sprintf(path, "extend_trainSet_data_%d", Maximum_DataSet_Number);
			ds.trainSet_Save_Load(false, Max_Data_Size, path);
			/*for (int t = 0; t < Generator_Count; t++)
				thr[t] = new thread(&Data_Reanalyze_Proc, t, ref(ds), ref(e[t]), ref(MP[t]), Generator_Count);*/
			for (int i = 0; i < Generator_Count; i++) {
				thr[i]->join(); delete thr[i];
			}
			ds.trainSet_Save_Load(true, -1, path);
		}

		for (int i = 0; i < Generator_Count; i++) {
			thr[i]->join();
			delete thr[i];
		}
		delete Group;
		delete[] MP;
	}
}
void multiMCTS_Train_extend(Agent* agent, int agent_id, int Maximum_DataSet_Number, Environment& e, int Generator_Count, int TrainDataNum) {

	agent_id++;

	/*auto res = Combine(0, Generator_Count, e);
	if (res->dataCount) {
		char path[100]; sprintf(path, "extend_trainSet_data_%d", Maximum_DataSet_Number + 1);
		res->trainSet_Save_Load(true, -1, path);
		Maximum_DataSet_Number++;
		delete res;
	}*/

	int train_cnt = 0;
	double ori_Speed = agent->Net_Param["Speed"];
	while (true) {
		{
			DataSet train_ds;
			//agent->Net_Param["Speed"] = max(ori_Speed * pow(0.1, floor(train_cnt / agent->Net_Param["Speed_decay_time"])), 2e-5);
			int cnt = TrainDataNum / 300000;
			while (cnt-- > 0) {
				Sampling_DataSet(train_ds, 300000, TrainDataNum, Maximum_DataSet_Number, &e);
				//async_Train(e.getJointNetFun(), agent, "Chess_best_param", Go_Param, train_ds, e.getTrainFun());
				train_ds.miniTrain_Start(agent, NULL, Go_Param, e.getTrainFun());
				agent->Write_to_File("SL_Net");
			}
		}

		//Update MCTS rollout agent
		char path[50]; sprintf(path, "SL_Net_%d", train_cnt);
		agent->Write_to_File(path);

		if (train_cnt == 20) {
			/*lock_guard<mutex>rollout_locker(rollout_lock);
			printf("Update Agent\n");

			e.JoinetNet_Assignment(rollout_agent, agent, rollout_Num);

			char path[50]; sprintf(path, "ExAgent_#%d", Agents_Count);
			agent->Write_to_File(path);
			fstream file; file.open("Ex_lastest_Agent_id", ios::out);
			if (file.is_open()) {
				char buf[50]; int sz = sprintf(buf, "%d", Agents_Count);
				file.write(buf, sz);
			}file.close();
			extra_games_count = GamesPerAgent * GamesPerAgent_Factor;*/

			break;
		}
		//control train time
		train_cnt++;
	}
}


