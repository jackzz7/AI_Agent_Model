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

DataSet* Combine_MP_ds(DataSet* ds, DataSet* ds1, Environment& e);
//extern GomokuData**game_data[Cuda_Max_Stream];
namespace MCTS_extend {

	static const int bits = 4;
	static const int Maximum_Cache = 1 << bits;
	static const int BufferSize = 400;


	static const int Max_Action_Space = 5000;
	static const int MaxGameLength = 513;

	//random number for stone
	us Randomer[Max_Action_Space][2];
	enum Colour {
		Black, White
	};

	template<class T>
	void _WR(T&var, char*target, ui&idx, bool Write) {
		if (Write)
			memcpy(&target[idx], (void*)&var, sizeof(T)), idx += sizeof(T);
		else memcpy((void*)&var, &target[idx], sizeof(T)), idx += sizeof(T);
	}
	template<class T>
	void _WR(T*var, char*target, ui Count, ui&idx, bool Write) {
		if (Write)
			memcpy(&target[idx], (void*)var, sizeof(T)*Count), idx += sizeof(T)*Count;
		else memcpy(var, &target[idx], sizeof(T)*Count), idx += sizeof(T)*Count;
	}


	static int Hidden_State_Size;
	static int Game_Action_Space;
	static int Screen_Size;
	struct PINFO;
	struct MCTS_Edge {

		const static int Edge_Size = 2 * sizeof(int) + 2 * sizeof(float);
		//ui move;
		//volatile char next_position_flag;
		volatile float Q_Value;
		//traverse Count
		volatile int Visit_Count;
		//control explore
		volatile float Prior_Probability;
		//link next position ID
		//Hidden State Position
		int next_position_ID;
		//Real root Position
		//int next_root_ID;

		PINFO *volatile next_Node;


		MCTS_Edge() {
			Q_Value = 0;
			Visit_Count = 0;
			Prior_Probability = -1;
			next_position_ID = -1;
			next_Node = NULL;
		}
		MCTS_Edge(const double&Priori, int next_ID) :MCTS_Edge() {
			Prior_Probability = Priori;
			next_position_ID = next_ID;
		}

		//static const int Cpuct = 1.0;
		double Confidence_Bound(const double&sqrt_tot_Node_visit) const {
			assert(!isnan(Q_Value));
			return Q_Value + Prior_Probability * sqrt_tot_Node_visit / (1 + Visit_Count);
		}
		MCTS_Edge&operator=(const MCTS_Edge&e) {
			memcpy(this, &e, sizeof(MCTS_Edge));
			return *this;
		}
		void Update_Q_Value() {
			if (Visit_Count > 0);//Q_Value = tot_Value / Visit_Count;
			else Visit_Count = 0, Q_Value = 0;
		}
		void Virtual_Loss(const int& vl) {
			//tot_Value -= vl;
			int tmp = 0;
			if ((tmp = Visit_Count + vl) > 0) {
				assert(tmp > 0);
				Q_Value = (1.0 * Q_Value * Visit_Count - vl) / tmp;
				Visit_Count = tmp;
			}
			else Visit_Count = Q_Value = 0;
		}
		void restore_Virutal_Loss(const int&vl, const std::pair<float, ui>&Visted) {
			int tmp = (Visit_Count + -vl + Visted.Y);
			if (tmp > 0) {
				Q_Value = (1.0 * Q_Value * Visit_Count + vl + Visted.X) / tmp;
				Visit_Count += -vl + Visted.Y;
			}
			else Visit_Count = Q_Value = 0;
		}


		void WR(char*target, bool Write) {
			ui _i = 0;
			_WR(Q_Value, target, _i, Write);
			_WR(Visit_Count, target, _i, Write);
			_WR(Prior_Probability, target, _i, Write);
			_WR(next_position_ID, target, _i, Write);
			//_WR(next_root_ID, target, _i, Write);
			assert(_i == Edge_Size);
		}
	};

	struct EDGE {
		vector<MCTS_Edge>*edge;
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
		MCTS_Edge&UCB(volatile ull&tot_Visit, mt19937&rng, ui&move) {
			++tot_Visit;
			assert(sz > 0);
			const double Cbase = 19652, Cinit = 1.25;
			double sqrt_tot_visit = sqrt(tot_Visit)*(log((1.0 + tot_Visit + Cbase) / Cbase) + Cinit);
			ui idx = rng() % sz, _i = -1;
			MCTS_Edge*best = &(*edge)[idx]; double mx = (*edge)[idx].Confidence_Bound(sqrt_tot_visit);
			best->Virtual_Loss(1);
			for (auto&k : *edge) {
				_i++;
				if (k.Prior_Probability == -1)continue;
				double tar = k.Confidence_Bound(sqrt_tot_visit);
				if (mx < tar || best->Prior_Probability == -1)k.Virtual_Loss(1), best->Virtual_Loss(-1), best = &k, mx = tar, idx = _i;
			}assert(best->Prior_Probability >= 0);
			move = idx;
			return *best;
		}
		void push(const MCTS_Edge&e) {
			assert(sz < Game_Action_Space);
			(*edge)[sz] = e;
			(*edge)[sz].Update_Q_Value();
			if (sz + 1 < Game_Action_Space)(*edge)[sz + 1].next_position_ID = -1;
			sz++;
		}
		//end flag
		void push_end() {
			if (sz < Game_Action_Space) {
				assert(false);
				(*edge)[sz] = MCTS_Edge(-1, -2);
			}
		}
		bool Ready() {
			return (sz == Game_Action_Space);// || (*edge)[sz].next_position_ID == -2;
		}
		bool empty() {
			return sz == 0;
		}
		EDGE&operator=(EDGE&&e) {
			swap(this->edge, e.edge);
			swap(this->sz, e.sz);
			return *this;
		}
	};

	struct Position {
		static size_t Position_Size;
		//ID+Feature=Identity
		int PID;
		Mat Hidden_State;

		float Value;
		float Action_Reward;
		//-1:PID==root_ID other:PID==virtual ID
		//int root_ID_exist;

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
		static void addStone(us&Feature, const ui&pos, const Colour&colour) {
			Feature = (Feature ^ Randomer[pos][colour]) % Maximum_Cache;
		}
		void WR(char*target, bool Write) {
			ui _i = 0;
			for (int i = 0; i < Game_Action_Space; i++) {
				MCTS_Edge e = (*edges.edge)[i];
				e.WR(&target[_i], Write);
				_i += MCTS_Edge::Edge_Size;
				if (!Write)edges.push(e);
			}
			//_WR(root_ID_exist, target, _i, Write);
			_WR(Value, target, _i, Write);
			_WR(Action_Reward, target, _i, Write);
			assert(_i == Position_Size);
			//_WR(edges.sz, target, _i, Write);
			//assert(0 < edges.sz&&edges.sz <= Action_Space);
		}
	};
	size_t Position::Position_Size;

	struct PINFO {
		Position p;
		ull totVisit;
		bool Virtual_File_Flag;
	};
	typedef list<PINFO*> PList;
	struct trajectory_Data {
		//Screen
		double*Scr;
		//action
		ui move = 0;
		//probability
		double*OutPut;
		int out_cnt;
		//Value
		double Value;
		//Reward
		double Action_Reward;
		trajectory_Data() {
			OutPut = new double[Game_Action_Space];
			Scr = new double[Screen_Size];
		}
		~trajectory_Data() {
			delete[] OutPut;
			delete[] Scr;
		}
	};


	//gobal variable
	void Init_MCTS(Environment&e, const char*dir = "") {
		fstream file; file.open((string)dir + "XOR_Random_Number", ios::in | ios::binary);
		if (file)file.read((char*)Randomer, Max_Action_Space * 2 * sizeof(us)), assert(file.gcount() == Max_Action_Space * 2 * sizeof(us)), file.close();
		else {
			for (int i = 0; i < Max_Action_Space; i++)
				for (int j = 0; j < 2; j++)
					Randomer[i][j] = rand_i();
			file.close();
			file.open((string)dir + "XOR_Random_Number", ios::out | ios::binary);
			file.write((char*)Randomer, Max_Action_Space * 2 * sizeof(us)), file.close();
		}
		Game_Action_Space = e.getRealGameActionSpace();
		Screen_Size = e.getScreen_Size();
		Hidden_State_Size = e.getHiddenStateSize();
		Position::Position_Size = sizeof(float) * 2 + Game_Action_Space * MCTS_Edge::Edge_Size;
	}


	struct WR_Manager {
		int BufferMaxSize = BufferSize;
		//recent Position buffer
		//Position,rawPosition,VisitCount
		PList UpdateBuff[Maximum_Cache];
		//new Position Cache
		PList NewPositionBuff[Maximum_Cache];

		map<ui, PList::iterator>PositionCache[Maximum_Cache];

		int IDCount[Maximum_Cache];


		//string DirPath;
		char dir[100];
		WR_Manager(const char*dir_path, const char*Name) {
			if (_access(dir_path, 0) != 0) _mkdir(dir_path);
			sprintf_s(dir, "%s%s\\", dir_path, Name);
			if (_access(dir, 0) != 0) _mkdir(dir);
			for (int t = 0; t < Maximum_Cache; t++) {
				PositionCache[t].clear();
				UpdateBuff[t].clear();
				NewPositionBuff[t].clear();
				IDCount[t] = 0;
				ifstream file;
				char Name[100]; sprintf_s(Name, "%s%d", dir, t);
				file.open(Name, ios::binary | ios::in);
				if (file) {
					file.seekg(0, ios::end);
					IDCount[t] = file.tellg() / Position::Position_Size;
					assert(file.tellg() % Position::Position_Size == 0);
				}
				file.close();
			}
		}
		~WR_Manager() {
			//File_Thread_End();
			for (int i = 0; i < Maximum_Cache; i++) {
				End_File_Write(i);
				free_Node[i].remove_if([](auto const&e) {delete e; return true; });
			}
			for (int i = 0; i < Maximum_Cache; i++) {
				for (auto&k : UpdateBuff[i])delete k; UpdateBuff[i].clear();
				for (auto&k : NewPositionBuff[i])delete k; NewPositionBuff[i].clear();
				PositionCache[i].clear();
			}
		}
		//return ID
		int newPosition(const us&Feature) {
			return IDCount[Feature]++;
		}
		pair<PINFO*, int> InitPosition(const int&ID, ifstream&file, PList&alloc) {
			char*buff = new char[Position::Position_Size];
			auto _new = alloc.front();
			_new->p.PID = ID; _new->p.edges.sz = 0; _new->totVisit = 0; _new->Virtual_File_Flag = false;
			if (file.is_open()) {
				if (file.read(buff, Position::Position_Size).gcount() != Position::Position_Size) {
					assert(false); file.close();
					PINFO*volatile ret = NULL; return { ret, -1 };
				}
				_new->p.WR(buff, false);
				for (auto&k : *_new->p.edges.edge) {
					//assert(k.Prior_Probability != -1 || k.next_position_ID == -1);
					if (k.next_position_ID != -1)
						_new->totVisit += k.Visit_Count;
				}
				//_new->p.edges.push_end();
			}
			else assert(false);
			file.close();
			delete[] buff;
			return { _new,1 };
		}
		pair<PINFO*, int> ReadPosition(int&ID, const us&Feature, PList&alloc, PINFO* volatile&Node) {
			//assert(ID < IDCount[Feature]);
			//if (ID != -1) {
			//	//check virtual file
			//	//NewPosition
			//	if (!Virtual_File_New[Feature].empty() && ID >= Virtual_File_New[Feature].front()->p.PID) {
			//		if (ID <= Virtual_File_New[Feature].back()->p.PID)
			//			return { Virtual_File_New[Feature][ID - Virtual_File_New[Feature].front()->p.PID] ,5 };
			//	}

			//	//locate file content
			//	ifstream file; //ui ReadSize = 0; ull pos[2];
			//	if (!Node) alloc.front()->p.edges.sz = 0, Node = alloc.front();
			//	else return { Node,-1 };
			//	int FilePosCount = IDCount[Feature] - NewPositionBuff[Feature].size();
			//	assert(ID < FilePosCount);
			//	char Name[100]; sprintf_s(Name, "%s%u", dir, Feature);
			//	file.open(Name, ios::binary | ios::in);
			//	if (file) {
			//		file.seekg(ID * Position::Position_Size);
			//	}
			//	else assert(false), file.close();
			//	//Initial from File
			//	return InitPosition(ID, file, alloc);
			//}
			////create new Position
			//else {
			auto p = alloc.front();
			p->p.PID = -1;// p->p.root_ID_exist = -1;
			p->p.edges.sz = 0; p->totVisit = 0; p->Virtual_File_Flag = false;
			return { p,0 };
			//}
		}
		class Buffer {
			char*data;
			ui cnt, rcnt;
			fstream file;
			enum Mode :int;
			Mode mode;
			static const ui MaxSize = 1e7;
			void w2file() {
				if (cnt > 0)if (!file.write(data, cnt))assert(false);
				cnt = 0;
			}
			void w2mem() {
				assert(rcnt - cnt >= 0);
				memcpy(data, &data[cnt], rcnt - cnt);
				cnt = rcnt - cnt;
				if (file.read(&data[cnt], MaxSize - cnt).gcount() > 0) {
					rcnt = file.gcount() + cnt;
					cnt = 0;
				}
				else rcnt = cnt, cnt = 0, assert(false);
			}
		public:
			enum Mode {
				Read, Write
			};
			Buffer() {
				data = new char[MaxSize];
			}
			~Buffer() {
				delete[]data;
			}
			void Init(const char*Path, Mode mode, ios_base::openmode extraFlag = (ios_base::openmode)0) {
				rcnt = cnt = 0; this->mode = mode;
				if (file.is_open())file.close();
				if (mode == Read)file.open(Path, ios::binary | ios::in | extraFlag);
				else file.open(Path, ios::binary | ios::out | extraFlag);
			}
			void write(const void*obj, size_t size) {
				assert(size <= MaxSize && mode == Write);
				if (cnt + size > MaxSize)w2file();
				memcpy(&data[cnt], obj, size);
				cnt += size;
			}
			void read(void*obj, size_t size) {
				assert(size <= MaxSize && mode == Read);
				if (cnt + size > rcnt)w2mem();
				memcpy(obj, &data[cnt], size);
				cnt += size;
			}
			void seekg(ull pos) {
				assert(mode == Read);
				file.seekg(pos);
				rcnt = cnt = 0;
			}
			void seekp(ull pos, ios_base::seekdir way = ios::beg) {
				assert(mode == Write);
				w2file();
				file.seekp(pos, way);
			}
			ull tellp() {
				assert(mode == Write);
				return file.tellp();
			}
			void writeblank(size_t size) {
				assert(size <= MaxSize && mode == Write);
				if (cnt + size > MaxSize)w2file();
				cnt += size;
			}
			void close() {
				if (mode == Write && cnt > 0)w2file();
				file.flush();
				file.close();
			}
		};

		Buffer _file[Maximum_Cache];
		void WriteNewPosition(const us&Feature, PList&NewPositionBuff) {
			if (NewPositionBuff.empty())return;
			char Name[100]; sprintf_s(Name, "%s%u", dir, Feature);
			//string path = (string)Name + "_idx";
			Buffer&file = _file[Feature];
			file.Init(Name, Buffer::Write, ios::app);// , idx_file.Init(path.c_str(), Buffer::Write, ios::app);
			file.seekp(0, ios::end);
			//ull pos_idx = file.tellp();
			char*buff = new char[Position::Position_Size];
			for (const auto&it : NewPositionBuff) {
				it->p.WR(buff, true);
				file.write(buff, Position::Position_Size);
				assert(it->p.PID != -1);
			}
			delete[] buff;
			file.close();
			//idx_file.close();
		}
		void UpdatePosition(const us&Feature, PList&UpdateBuff) {
			if (UpdateBuff.empty())return;
			char Name[100]; sprintf_s(Name, "%s%u", dir, Feature);
			//string path = (string)Name + "_idx";
			Buffer&file = _file[Feature];
			file.Init(Name, Buffer::Write, ios::in);
			//ifstream idx_file(path.c_str(), ios::in | ios::binary);
			char buff[MCTS_Edge::Edge_Size];
			for (const auto&it : UpdateBuff) {
				//idx_file.seekg(it->p.PID * sizeof(ull));
				//ull pos; idx_file.read((char*)&pos, sizeof(ull));
				file.seekp(it->p.PID*Position::Position_Size);
				ui _num = 0;
				//file.write(&num, sizeof(uc));
				//assert(0 < num <= W * H);
				for (auto&k : *it->p.edges.edge) {
					//if (k.Prior_Probability == -1) { assert(false); break; }
					k.WR(buff, true);// Write(buff);
					file.write(buff, MCTS_Edge::Edge_Size);
					_num++;
				}
				assert(Game_Action_Space == _num);
				assert(it->p.PID != -1);
				//_num = 0; _WR(it->p.root_ID_exist, buff, _num, true);
				//file.write(buff, _num);
			}
			file.close();
			//idx_file.close();
		}

		PList free_Node[Maximum_Cache];
		thread*File_thread[Maximum_Cache] = { NULL };
		PList _UpdateBuff[Maximum_Cache], _NewPositionBuff[Maximum_Cache];
		vector<PINFO*> Virtual_File_New[Maximum_Cache];

		void File_Thread_Start() {
			for (int i = 0; i < Maximum_Cache; i++) {
				File_thread[i] = new thread([this, i]() {
					UpdatePosition(i, _UpdateBuff[i]); //_UpdateBuff[i].clear();
					WriteNewPosition(i, _NewPositionBuff[i]); //_NewPositionBuff[i].clear();
					});
			}
			/*for (int i = 0; i < Maximum_Cache; i++) {
				for (auto&e : _UpdateBuff[i])
					for (auto&k : *e->p.edges.edge)assert(k.next_root_ID == -1 || k.next_position_ID != -1);
				for (auto&e : _NewPositionBuff[i])
					for (auto&k : *e->p.edges.edge)assert(k.next_root_ID == -1 || k.next_position_ID != -1);
			}*/
		}
		void End_File_Write(int i) {
			//for (int i = 0; i < Maximum_Cache; i++)
			if (File_thread[i]) {
				File_thread[i]->join(); delete File_thread[i]; File_thread[i] = NULL;

				for (auto it = _UpdateBuff[i].begin(); it != _UpdateBuff[i].end(); it++)
					if (!(*it)->Virtual_File_Flag) {
						if (PositionCache[i].find((*it)->p.PID) != PositionCache[i].end())assert(PositionCache[i][(*it)->p.PID]._Getcont() != it._Getcont());
					}
					else assert(PositionCache[i][(*it)->p.PID] == it);
				_UpdateBuff[i].remove_if([](auto const&e) {return !e->Virtual_File_Flag; });
				_NewPositionBuff[i].remove_if([](auto const&e) {return !e->Virtual_File_Flag; });
				for (auto&k : _UpdateBuff[i])PositionCache[i].erase(k->p.PID);
				assert(PositionCache[i].size() == UpdateBuff[i].size());
				free_Node[i].splice(free_Node[i].end(), _UpdateBuff[i]);
				free_Node[i].splice(free_Node[i].end(), _NewPositionBuff[i]);
				//Virtual_File_UB[i].clear();
				Virtual_File_New[i].clear();
			}
		}
		void RecycleBuffer(int i) {
			End_File_Write(i);
			//move Node
			//for (int i = 0; i < Maximum_Cache; i++) {
			auto it = UpdateBuff[i].begin();
			assert(_UpdateBuff[i].empty());
			advance(it, max((int)UpdateBuff[i].size() - BufferMaxSize * 2, 0));// virtual_file_UB_Count[i]); //if (it != UpdateBuff[i].begin())_debug_ub_end[i] = prev(it);
			//_UpdateBuff[i].assign(UpdateBuff[i].begin(), it);
			_UpdateBuff[i].splice(_UpdateBuff[i].end(), UpdateBuff[i], UpdateBuff[i].begin(), it);
			for (auto&it : _UpdateBuff[i]) {
				assert(PositionCache[i].find(it->p.PID) != PositionCache[i].end());// && Virtual_File_UB[i].find(it->p.PID) == Virtual_File_UB[i].end());
				//PositionCache[i].erase(it->p.PID);
				//virtual file flag
				it->Virtual_File_Flag = true;
				//Virtual_File_UB[i][it->p.PID] = it;
			}

			//it = NewPositionBuff[i].begin();
			assert(_NewPositionBuff[i].empty());
			//virtual_file_new_Count[i] = ;
			//advance(it, max((int)NewPositionBuff[i].size() - 0, 0));// virtual_file_new_Count[i]); if (it != NewPositionBuff[i].begin())_debug_new_end[i] = prev(it);
			//_NewPositionBuff[i].assign(NewPositionBuff[i].begin(), it);
			_NewPositionBuff[i].splice(_NewPositionBuff[i].end(), NewPositionBuff[i]);// , NewPositionBuff[i].begin(), it);
			for (auto&it : _NewPositionBuff[i]) {
				//PositionCache[i].erase((it)->X.X.PID);
				//Virtual_File[i][(it)->X.X.PID] = it;
				//virtual file flag
				it->Virtual_File_Flag = true;
				//Virtual_File_New[i].push_back(it);
				//it->X.Y = true;
			}Virtual_File_New[i].assign(_NewPositionBuff[i].begin(), _NewPositionBuff[i].end());
			//assert(Virtual_File[i].size() == _UpdateBuff[i].size() + _NewPositionBuff[i].size());
			//}
		}
	};
	struct PositionLookUp {
		//position,raw Board
		PList MostViste[Maximum_Cache];

		PList RecentFrequent[Maximum_Cache];
		map<ui, PList::iterator>RF_map[Maximum_Cache];


		PositionLookUp() {
			clear();
		}
		~PositionLookUp() {
			clear();
		}
		void clear() {
			for (int i = 0; i < Maximum_Cache; i++) {
				for (auto&e : MostViste[i])delete e; MostViste[i].clear();
				for (auto&e : RecentFrequent[i])delete e; RecentFrequent[i].clear();
				RF_map[i].clear();
			}
		}
		//ID+Feature,read only
		pair<PINFO*, int> getPosition(WR_Manager&File_Buffer, int&ID, const us&Feature, PList&alloc, PINFO* volatile&Node) {
			//if (ID != -1) {
			//	//find MostVisit
			//	for (auto&it : MostViste[Feature]) {
			//		if (it->p.PID == ID) {
			//			assert(it->Virtual_File_Flag == false);
			//			return { it,4 };
			//		}
			//	}
			//	//find RecentFreq
			//	map<ui, PList::iterator>::const_iterator it;
			//	if ((it = RF_map[Feature].find(ID)) != RF_map[Feature].end()) {
			//		assert((*it->Y)->Virtual_File_Flag == false);
			//		return { *it->Y,3 };
			//	}

			//	//find Update buffer and newPosition buffer
			//	if ((it = File_Buffer.PositionCache[Feature].find(ID)) != File_Buffer.PositionCache[Feature].end()) {
			//		//assert((*it->Y)->Virtual_File_Flag == false);
			//		//virtual file
			//		if ((*it->Y)->Virtual_File_Flag)
			//			return { *it->Y,5 };
			//		else return { *it->Y,2 };
			//	}
			//}
			assert(ID == -1);
			//Create or Initial from file
			return File_Buffer.ReadPosition(ID, Feature, alloc, Node);
		}

		bool getMax(PList&ls, PList::const_iterator&_it) {
			ull mx = 0;
			//advance(it, _advance);
			for (auto it = ls.begin(); it != ls.end(); it++) {
				if ((*it)->totVisit > mx) {
					mx = (*it)->totVisit, _it = it;
				}
			}
			return mx > 0;
		}
		bool getMin(PList&ls, PList::const_iterator&_it) {
			ull mx = 1e18;
			for (auto it = ls.begin(); it != ls.end(); it++) {
				if ((*it)->totVisit < mx) {
					mx = (*it)->totVisit, _it = it;
				}
			}
			return mx != 1e18;
		}
		//UpdateBuffer to MostVisit
		void UB2MV(WR_Manager&File_Buffer, int i) {
			//for (int i = 0; i < Maximum_Cache; i++) {
			PList::const_iterator it;
			//push back most visit
			if (getMax(File_Buffer.UpdateBuff[i], it)) {
				assert(File_Buffer.PositionCache[i].find((*it)->p.PID) != File_Buffer.PositionCache[i].end());
				assert((*it)->Virtual_File_Flag == false);
				MostViste[i].push_back(*it), File_Buffer.PositionCache[i].erase((*it)->p.PID), File_Buffer.UpdateBuff[i].erase(it);
			}
			//}
		}
		//RF,MV move to UpdateCache
		void Recycle(WR_Manager&File_Buffer, int i) {
			//for (int i = 0; i < Maximum_Cache; i++) {
				//RF move to Update
			int cnt = RecentFrequent[i].size() - File_Buffer.BufferMaxSize;
			for (int j = 0; j < cnt; j++) {
				auto p = RecentFrequent[i].front(); RecentFrequent[i].pop_front(); RF_map[i].erase(p->p.PID);
				assert(p->p.PID != -1 && File_Buffer.PositionCache[i].find(p->p.PID) == File_Buffer.PositionCache[i].end());
				assert(p->Virtual_File_Flag == false);
				File_Buffer.UpdateBuff[i].push_back(p);
				File_Buffer.PositionCache[i][p->p.PID] = prev(File_Buffer.UpdateBuff[i].end());
			}
			//least recent frequent move to Update
			if (MostViste[i].size() > File_Buffer.BufferMaxSize / 2) {
				auto p = MostViste[i].front(); MostViste[i].pop_front();
				assert(p->p.PID != -1 && File_Buffer.PositionCache[i].find(p->p.PID) == File_Buffer.PositionCache[i].end());
				assert(p->Virtual_File_Flag == false);
				File_Buffer.UpdateBuff[i].push_back(p);
				File_Buffer.PositionCache[i][p->p.PID] = prev(File_Buffer.UpdateBuff[i].end());
			}
			//}
		}
	};
	static const int Max_Thr = 256;
	struct Evalution {
		int ID, PolicyID;
		//PINFO*Node;
		Mat*H_State;
		ui Action;
		int step;

		double*OutPut;
		double Value;
		Mat*Next_State;
		double Reward;

		Evalution() { Next_State = NULL; OutPut = NULL; }
		void Init(int tid, int Policy_ID, Mat*State, Mat*Next_State, int step, ui action) {
			ID = tid; PolicyID = Policy_ID;
			this->H_State = State; this->step = step;
			Action = action;
			if (!OutPut)OutPut = new double[Game_Action_Space];
			fill(OutPut, OutPut + Game_Action_Space, -1.0);
			Value = -2;
			this->Next_State = Next_State;
			Reward = 0;
		}
		~Evalution() {
			delete[] OutPut;
		}
	};
	
	static bool IsTest;

	class MCTS_Policy_Extend {
		int PolicyID;
		bool*ActionIsValid;

		//temporary stay in memory,interface with file
		WR_Manager*File_Buffer;
		PositionLookUp*LookUpTable;

		//resign
		double V_resign, Enable_resign;
		double Min_V_resign[2];
		static const int resign_MaxSize = 2000;
		static const int resign_MinSize = 100;
		list<double>recent_resign;
		multiset<double>resign_Sort_Set;
	public:
		Environment*environment_API;
		condition_variable*Eva_CV = NULL;
		mutex*Eva_m = NULL;

		MCTS_Policy_Extend(int ID, Environment*e, int Thr = 1, int Sim = 500, const char*Name = "0", const char*dir = "MCTS_Policy\\") :PolicyID(ID), Thr_Num(Thr), SimulationPerMCTS(Sim), environment_API(e) {
			File_Buffer = new WR_Manager(dir, Name);
			LookUpTable = new PositionLookUp();
			ActionIsValid = new bool[Game_Action_Space];
			dir_noise = new double[Game_Action_Space];

			trainData = new trajectory_Data[MaxGameLength];
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
			for (auto&k : recent_resign)resign_Sort_Set.insert(k);
			//init random
			std::vector<std::uint32_t> seeds;
			seeds.resize(Thr_Num + 1);
			std::seed_seq seq{ time(0),(ll)File_Buffer };
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
			delete LookUpTable;
			delete[] ActionIsValid;
			delete[] dir_noise;
			delete[] trainData;
			for (int j = 0; j < Max_Thr; j++) {
				pre_alloc[j].remove_if([](auto const&e) {delete e; return true; });
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
					double&val = *(it++);
					File_WR(file, &val, sizeof(val), Write);
				}
			}
			file.close();
		}
		void Update_and_Clear() {
			for (int i = 0; i < Thr_Num; i++) {
				for (auto&k : Path[i]) {
					if (k._new)
						pre_alloc[i].push_back(k._new);
					k.node = NULL;
				}
				Path[i].clear();
			}
		}
		//void Update_and_Write() {
		//	//recycle alloc
		//	int sum = 0;
		//	for (int i = 0; i < Thr_Num; i++) {
		//		for (auto&k : Path[i]) {
		//			if (k._new&&k.node != k._new)
		//				pre_alloc[i].push_back(k._new);
		//			sum++;
		//		}
		//	}
		//	for (int i = 0; i < Thr_Num; i++) {
		//		for (auto&k : Path[i]) {
		//			_Path[k.feature].emplace_back(k);
		//		}Path[i].clear();
		//	}
		//	for (int i = 0; i < Maximum_Cache; i++)sum -= _Path[i].size();
		//	assert(sum == 0);
		//	//File_Buffer->File_Thread_End();
		//	thread*io[Maximum_Cache];
		//	for (int i = 0; i < Maximum_Cache; i++) {
		//		io[i] = new thread([this, i]() {
		//			UpdatePath(i);
		//			//most visited Update move to MV
		//			//every game once
		//			if (environment_API->step == 0)
		//				LookUpTable->UB2MV(*File_Buffer, i);
		//			//async write to File
		//			File_Buffer->RecycleBuffer(i);
		//			});
		//	}
		//	for (int i = 0; i < Maximum_Cache; i++)
		//		io[i]->join(), delete io[i];
		//	File_Buffer->File_Thread_Start();
		//}
		//write all data to file
		void Write2File() {
			//Eva_Loop_end();
			resign_WR(true);
			//File_Buffer->File_Thread_End();
			//for (int i = 0; i < Maximum_Cache; i++)
			//	File_Buffer->End_File_Write(i);
			////File_Buffer->BufferMaxSize = 0;
			//for (int i = 0; i < Maximum_Cache; i++) {
			//	//move all to buffer
			//	File_Buffer->UpdateBuff[i].splice(File_Buffer->UpdateBuff[i].end(), LookUpTable->MostViste[i]);
			//	File_Buffer->UpdateBuff[i].splice(File_Buffer->UpdateBuff[i].end(), LookUpTable->RecentFrequent[i]);
			//	LookUpTable->RF_map[i].clear();

			//	File_Buffer->UpdatePosition(i, File_Buffer->UpdateBuff[i]);
			//	File_Buffer->WriteNewPosition(i, File_Buffer->NewPositionBuff[i]);
			//	for (auto&e : File_Buffer->UpdateBuff[i]) assert(!e->Virtual_File_Flag), File_Buffer->free_Node[i].push_back(e);
			//	for (auto&e : File_Buffer->NewPositionBuff[i]) assert(!e->Virtual_File_Flag), File_Buffer->free_Node[i].push_back(e);
			//	File_Buffer->UpdateBuff[i].clear();
			//	File_Buffer->NewPositionBuff[i].clear();
			//	File_Buffer->PositionCache[i].clear();
			//}
		}
		void addFeature(us&Feature, const ui&move, int step) {
			//position[move%W][move / W] = step % 2 + 1;
			Position::addStone(Feature, move, (Colour)(step % 2));
			//Position::addStone(rawPosition, move, (Colour)(step % 2));
		}
		void removeFeature(us&Feature, const ui&move, int step) {
			//position[move%W][move / W] = 0;
			Position::addStone(Feature, move, (Colour)(step % 2));
			//Position::removeStone(rawPosition, move);
		}
		
		struct Node_Info {
			PINFO*volatile&node, *_new;
			int type;
			us feature;
			int&pid;
			Node_Info(PINFO*volatile&Node, int ty, us fea, int&PID, PINFO*__new) :node(Node), pid(PID) {
				type = ty;
				feature = fea;
				_new = __new;
			}
		};
		volatile int Response[Max_Thr];
		Evalution Eva_Stack[Max_Thr];
		condition_variable Response_CV[Max_Thr];
		mutex Response_mux[Max_Thr];
		mt19937 rng[Max_Thr + 1];
		vector<Node_Info>Path[Max_Thr];
		PList pre_alloc[Max_Thr];
		double MC_Tree_Search(PINFO* volatile&Node, int&PID, us Feature, const int step, Mat*Hidden_State, const ui action, int tid) {
			int type = -1; PINFO*_new = NULL;
			//find Cache,Create or read Node
			if (!Node) {
				auto p = LookUpTable->getPosition(*File_Buffer, PID, Feature, pre_alloc[tid], Node);
				if (!Node || (p.Y == 1 && p.X == Node)) {
					type = p.Y, Node = p.X;
					if (type == 5)assert(Node->Virtual_File_Flag&&Node->p.PID == PID);
					if (type < 2) {
						_new = p.X; assert(_new == pre_alloc[tid].front());
						pre_alloc[tid].pop_front();
					}
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
				EDGE*edge = &Node->p.edges;
				Path[tid].emplace_back(Node_Info(Node, type, Feature, PID, _new));
				assert(_new != NULL);
				//node already evalution
				assert(abs(Eva_Stack[tid].Value) <= 1 + 1e-8);
				if (Node != _new) {
					return -Eva_Stack[tid].Value;
				}
				double*Action_Prior = Eva_Stack[tid].OutPut, Value = Eva_Stack[tid].Value;
				double Reward = Eva_Stack[tid].Reward;
				//add root action mask
				//no Action mask in inner Simulations
				for (int k = 0; k < Game_Action_Space; k++) {
					double cancel;
					/*if (IsTest) {
						cancel = ((!Hidden_State && !ActionIsValid[k]) || (Hidden_State && Action_Prior[k] < 5e-5)) ? -1 : Action_Prior[k];
					}
					else {*/
					cancel = (!Hidden_State && !ActionIsValid[k]) ? -1 : Action_Prior[k];
					//}
					edge->push(MCTS_Edge(cancel, -1));
					assert(Action_Prior[k] + 1e-8 >= 0);
				}
				edge->push_end();
				//add extra P(s,a) noise
				if (IsAddNoise && !Hidden_State)generate_noise(Node, environment_API->legal_moves_Count, tid);
				assert(edge == &Node->p.edges);
				assert(edge->sz == Game_Action_Space);

				Node->p.Value = Value;
				Node->p.Action_Reward = 0;// Reward;

				//wake up yield
				for (int i = 0; i < Thr_Num; i++) {
					if (Response[i] == 2) {
						lock_guard<mutex> lock(Response_mux[i]);
						Response_CV[i].notify_one();
					}
				}
				return -Value;
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
				ui move = -1; MCTS_Edge&best_act = edge->UCB(Node->totVisit, rng[tid], move);

				//add stone
				addFeature(Feature, move, step);

				////MCTS traverse subtree
				double result = MC_Tree_Search(best_act.next_Node, best_act.next_position_ID, Feature, step + 1, &Node->p.Hidden_State, move, tid);
				//discount Reward
				assert(abs(best_act.next_Node->p.Action_Reward) <= 1 + 1e-8);
				result = best_act.next_Node->p.Action_Reward + environment_API->getRewardDiscount()*result;
				//remove stone
				removeFeature(Feature, move, step);

				//Update best_act
				best_act.restore_Virutal_Loss(1, { result,1 });
				//assert(edge == &Node->p.edges);
				if (type == -1) {
					assert(Node != NULL);
					assert(!Node->p.edges.empty());
				}
				else Path[tid].emplace_back(Node_Info(Node, type, Feature, PID, _new));
				return -result;
			}
		}
		vector<Node_Info>_Path[Maximum_Cache];
		//Node write to Cache 
		//void UpdatePath(int tid) {
		//	//for (int i = 0; i < Thr_Num; i++) {
		//	for (auto&k : _Path[tid]) {
		//		assert(k.node != NULL);
		//		assert(!k.node->p.edges.empty());
		//		assert(k.feature == tid);
		//		//new Position
		//		assert(k.type == 5 || k.node->Virtual_File_Flag == false);
		//		if (k.type == 0) {
		//			if (k.node->p.PID == -1) {
		//				k.pid = k.node->p.PID = File_Buffer->newPosition(k.feature);
		//				//assert(File_Buffer->PositionCache[k.feature].find(k.node->X.X.PID) == File_Buffer->PositionCache[k.feature].end());
		//				File_Buffer->NewPositionBuff[k.feature].push_back(k.node);
		//				//File_Buffer->PositionCache[k.feature][k.node->X.X.PID] = prev(File_Buffer->NewPositionBuff[k.feature].end());
		//			}
		//		}
		//		//read from file/virtual file
		//		else if (k.type == 1 || k.type == 5) {
		//			assert(k.node->p.PID != -1);
		//			if (k.type == 5) {
		//				if (k.node->Virtual_File_Flag) {
		//					k.node->Virtual_File_Flag = false;
		//					//assert(File_Buffer->PositionCache[k.feature].find(k.node->p.PID) != File_Buffer->PositionCache[k.feature].end());
		//					File_Buffer->UpdateBuff[k.feature].push_back(k.node);
		//					File_Buffer->PositionCache[k.feature][k.node->p.PID] = prev(File_Buffer->UpdateBuff[k.feature].end());
		//				}
		//			}
		//			else if (File_Buffer->PositionCache[k.feature].find(k.node->p.PID) == File_Buffer->PositionCache[k.feature].end()) {
		//				File_Buffer->UpdateBuff[k.feature].push_back(k.node);
		//				File_Buffer->PositionCache[k.feature][k.node->p.PID] = prev(File_Buffer->UpdateBuff[k.feature].end());
		//				//if (k.type == 5)
		//					//assert(k.node->Virtual_File_Flag), k.node->Virtual_File_Flag = false;
		//					//File_Buffer->Virtual_File[k.feature].erase(k.node->X.X.PID);
		//			}
		//			//duplicate node
		//			else if (k.node == k._new&&*File_Buffer->PositionCache[k.feature][k.node->p.PID] != k.node) {
		//				File_Buffer->free_Node[k.feature].push_back(k.node);
		//			}
		//		}
		//		//hit PositionCache
		//		else if (k.type == 2) {
		//			//hit in UpdateCache,move to Recent Frequent
		//			//if (k.node->X.X.PID < File_Buffer->IDCount[k.feature] - File_Buffer->NewPositionBuff[k.feature].size()) {
		//			if (LookUpTable->RF_map[k.feature].find(k.node->p.PID) == LookUpTable->RF_map[k.feature].end()) {
		//				auto it = File_Buffer->PositionCache[k.feature][k.node->p.PID];
		//				assert((*it) == k.node);
		//				LookUpTable->RecentFrequent[k.feature].push_back(*it);
		//				LookUpTable->RF_map[k.feature][k.node->p.PID] = prev(LookUpTable->RecentFrequent[k.feature].end());
		//				File_Buffer->PositionCache[k.feature].erase(k.node->p.PID);
		//				File_Buffer->UpdateBuff[k.feature].erase(it);
		//			}
		//			else assert(k.node == (*LookUpTable->RF_map[k.feature][k.node->p.PID]));
		//			//}
		//			//newPosition virtual file,copy to Update
		//			/*else if (k.node->X.Y) {
		//				k.node->X.Y = false;
		//				File_Buffer->UpdateBuff[k.feature].push_back(k.node);
		//				File_Buffer->PositionCache[k.feature][k.node->X.X.PID] = prev(File_Buffer->UpdateBuff[k.feature].end());
		//			}*/
		//		}
		//		//find RecentFrequent
		//		else if (k.type == 3) {
		//			auto&it = LookUpTable->RF_map[k.feature][k.node->p.PID];
		//			assert(k.node == (*it));
		//			//move to list back
		//			if (it != prev(LookUpTable->RecentFrequent[k.feature].end())) {
		//				LookUpTable->RecentFrequent[k.feature].push_back(*it);
		//				LookUpTable->RecentFrequent[k.feature].erase(it);
		//				it = prev(LookUpTable->RecentFrequent[k.feature].end());
		//			}
		//		}//most visit
		//		else if (k.type == 4) {
		//			auto&ls = LookUpTable->MostViste[k.feature];
		//			if (ls.back() != k.node) {
		//				bool flag = false;
		//				for (auto it = ls.begin(); it != ls.end(); it++) {
		//					if (*it == k.node) {
		//						ls.push_back(*it); ls.erase(it);
		//						flag = true;
		//						break;
		//					}
		//				}assert(flag);
		//			}
		//		}
		//		else assert(false);
		//	}
		//	//}
		//	//reset and clear thread path
		//	for (auto&k : _Path[tid])
		//		//if (!(*k.node)->Virtual_File_Flag)
		//		k.node = NULL;
		//	_Path[tid].clear();
		//	//RF,MV to Update
		//	LookUpTable->Recycle(*File_Buffer, tid);
		//}
		//add dirichlet noise to P(s,a)
		const double dir_alpha = 0.75;
		double*dir_noise;
		void addDirNoise(PINFO*Node, double*dir_distribution) {
			for (int i = 0; i < Game_Action_Space; i++) {
				auto&e = (*Node->p.edges.edge)[i];
				if (e.Prior_Probability == -1)continue;
				e.Prior_Probability = dir_alpha * e.Prior_Probability + (1 - dir_alpha)*dir_distribution[i];
			}
		}
		void removeDirNoise(PINFO*Node, double*dir_distribution) {
			for (int i = 0; i < Game_Action_Space; i++) {
				auto&e = (*Node->p.edges.edge)[i];
				if (e.Prior_Probability == -1)continue;
				e.Prior_Probability = (e.Prior_Probability - (1 - dir_alpha)*dir_distribution[i]) / dir_alpha;
			}
		}
		void generate_noise(PINFO*NoisedNode, int legal_moves_Count,int tid) {
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

		ui Sampling_Visit(PINFO*Node, MCTS_Edge*&best_act, int step, ull&tot, double*Max_Q = NULL, int Sampling_step = 0) {
			ui move = -1;
			for (auto&e : *Node->p.edges.edge) {
				move++;
				if (e.Prior_Probability == -1)continue;
				if (e.next_position_ID == -1 && !e.next_Node)continue;
				if (!ActionIsValid[move])continue;
				tot += e.Visit_Count, assert(e.Visit_Count >= 0);
				assert(!isnan(e.Q_Value));
				if (Max_Q)*Max_Q = max(*Max_Q, e.Q_Value);
			}assert(tot > 0);
			//sampling available moves
			move = -1;
			if (step < Sampling_step) {
				ull chi = rng[0]() % tot, cnt = 0;
				for (auto&e : *Node->p.edges.edge) {
					move++;
					if (e.Prior_Probability == -1)continue;
					if (e.next_position_ID == -1 && !e.next_Node)continue;
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
				for (auto&e : *Node->p.edges.edge) {
					move++;
					if (e.Prior_Probability == -1)continue;
					if (e.next_position_ID == -1 && !e.next_Node)continue;
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
		mutex Thr_lock;
		volatile int Finish_Thr;
		condition_variable Thr_CV[Max_Thr];
		mutex Thr_m[Max_Thr];
		bool IsAddNoise;
		void Thread_Start(int tid, PINFO*volatile&Node, Mat*Hidden_State, const ui Action) {
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
					int PID = -1; MC_Tree_Search(Node, PID, 0, 0, Hidden_State, Action, tid);
				}
				lock_guard<mutex>locker(Thr_lock);
				if (++Finish_Thr == Thr_Num)Main_CV.notify_one();
			}
		}
		trajectory_Data*trainData;
		int GameResult, GameStep;
		void Terminal_State(int step) {
			GameStep = step;
			//add terminal state
			environment_API->GetGameScreen(trainData[GameStep].Scr);
			//trainData[GameStep].move = move;
			trainData[GameStep].Value = 0;
			//uniform policy at terminal state
			fill(trainData[GameStep].OutPut, trainData[GameStep].OutPut + Game_Action_Space, 0.0);
			int&idx = trainData[GameStep].out_cnt = 0;
			trainData[GameStep].OutPut[idx++] = -1;
			trainData[GameStep].Action_Reward = 0;
			GameStep++;
		}
		thread*Thr[Max_Thr];
		condition_variable Main_CV;
		//generate RL data
		void Generate_Move(int step) {
			AllocNode();
			environment_API->GetInsActionMask(ActionIsValid);
			if (environment_API->GetGameState(&GameResult)) {
				Terminal_State(step);
				return;
			}
			{
				Finish_Thr = 0;
				//lock_guard<mutex>locker(*Eva_m);
				Eva_CV->notify_one();
			}
			Step = step; IsAddNoise = true;
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

			PINFO*Root = Node[0]; double Value = Node[0]->p.Value;
			//Agent sampling move
			MCTS_Edge*best_act = NULL; ull tot = 0; double Max_Q = -1e9;
			ui move = Sampling_Visit(Root, best_act, step, tot, &Max_Q, 10);
			assert(Max_Q != -1e9&&move != -1);
			//Update_and_Write();


			//enable resign,90% games 
			if (Enable_resign) {
				if (Max_Q < V_resign&&Value < V_resign) {
					//clear Node
					Update_and_Clear();
					printf("resign: ");
					(GameResult = step % 2 ? 1 : -1);
					Terminal_State(step);
					return;
				}
			}
			else Min_V_resign[step % 2] = min(Min_V_resign[step % 2], Max_Q);

			//add Train Data
			environment_API->GetGameScreen(trainData[step].Scr);
			trainData[step + 1].move = environment_API->getAct(move);
			trainData[step].Value = Value;
			int&idx = trainData[step].out_cnt = 0;
			fill(trainData[step].OutPut, trainData[step].OutPut + Game_Action_Space, 0.0);
			double sum = 0; ui _i = -1;
			for (auto&e : *Root->p.edges.edge) {
				_i++;
				if (e.Prior_Probability == -1)continue;
				if (e.next_position_ID == -1 && !e.next_Node)continue;
				if (!ActionIsValid[_i])continue;
				trainData[step].OutPut[idx++] = environment_API->getAct(_i);
				sum += trainData[step].OutPut[idx++] = 1.0*e.Visit_Count / tot;
				if (idx > 200)printf("\nerror exceed maximum ActionSpace\n");
				assert(!isnan(e.Q_Value));
			}assert(abs(sum - 1) <= 1e-8);
			//add stone
			//addFeature(Feature, move, step);
			environment_API->Act(move, NULL);
			//get real Reward
			trainData[step].Action_Reward = 0;

			Update_and_Clear();

			//environment_API->PrintScr();

			assert(Ins_PID == -1);
			Generate_Move(step + 1);
		}
		//Update Policy using lastest network
		void Data_Reanalyze(int step) {
			//environment_API->GetInsActionMask(ActionIsValid);
			if (trainData[step].out_cnt == 1)return;
			memset(ActionIsValid, 0, sizeof(bool)*Game_Action_Space);
			for (int i = 0; i < trainData[step].out_cnt / 2; i++) {
				if (trainData[step].OutPut[i * 2 + 1] == 0)continue;
				int idx = environment_API->getSimAct(trainData[step].OutPut[i * 2]);
				ActionIsValid[idx] = true;
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

			MCTS_Edge*best_act = NULL; ull tot = 0; double Max_Q = -1e9;
			Sampling_Visit(Node[0], best_act, step, tot, &Max_Q, 0);

			//Update data Policy
			int&idx = trainData[step].out_cnt = 0;
			fill(trainData[step].OutPut, trainData[step].OutPut + Game_Action_Space, 0.0);
			double sum = 0; ui _i = -1;
			for (auto&e : *Node[0]->p.edges.edge) {
				_i++;
				if (e.Prior_Probability == -1)continue;
				if (e.next_position_ID == -1 && !e.next_Node)continue;
				if (!ActionIsValid[_i])continue;
				trainData[step].OutPut[idx++] = environment_API->getAct(_i);
				sum += trainData[step].OutPut[idx++] = 1.0*e.Visit_Count / tot;
			}assert(abs(sum - 1) <= 1e-8);

			Update_and_Clear();
			Data_Reanalyze(step + 1);
		}

		void MCTS_Start() {
			Init_State();
			Generate_Move(0);
			//update resign value
			if (!Enable_resign) {
				int _sz = resign_Sort_Set.size();
				if (GameResult == 1)resign_Sort_Set.insert(Min_V_resign[0]), recent_resign.push_back(Min_V_resign[0]);
				else if (GameResult == -1)resign_Sort_Set.insert(Min_V_resign[1]), recent_resign.push_back(Min_V_resign[1]);
				else if (GameResult == 0) {
					double val = min(Min_V_resign[0], Min_V_resign[1]);
					resign_Sort_Set.insert(val), recent_resign.push_back(val);
				}
				else printf("GameResult error\n"), assert(false);
				if (_sz != resign_Sort_Set.size() && resign_Sort_Set.size() >= resign_MinSize) {
					//pop
					if (recent_resign.size() > resign_MaxSize) {
						assert(resign_Sort_Set.find(recent_resign.front()) != resign_Sort_Set.end());
						resign_Sort_Set.erase(resign_Sort_Set.find(recent_resign.front())), recent_resign.pop_front();
					}
					const double resign_rate = 0.05;//0.049;
					int sz = resign_Sort_Set.size()*resign_rate;
					for (auto&k : resign_Sort_Set)if (sz == 0) {
						V_resign = k; break;
					}
					else sz--;
				}
			}
		}
		


		//action API
		int Ins_PID, Step;
		us Ins_Feature;
		PINFO*volatile Node[MaxGameLength];
		int Thr_Num;
		int SimulationPerMCTS;

		void addStone(const MCTS_Edge&best_act, const ui&move) {
			//addFeature(Ins_Feature, move, Step);
			Step++;
			//Ins_PID = best_act.next_Node == NULL ? -1 : best_act.next_Node->p.root_ID_exist;
			//Node[Step] = best_act.next_Node;
			environment_API->Act(move, NULL);
		}

		void Init_State() {
			Ins_PID = File_Buffer->IDCount[0] == 0 ? -1 : 0;
			Step = 0; Ins_Feature = 0;
			memset((void*)(void**)Node, 0, sizeof(Node));
			AllocNode();
			Min_V_resign[0] = Min_V_resign[1] = 1e9;
			Enable_resign = recent_resign.size() < resign_MinSize ? false : ((rng[0]() % 10) < 7);
			environment_API->Reset();
		}
		void AllocNode() {
			for (int i = 1; i < Maximum_Cache; i++)
				File_Buffer->free_Node[0].splice(File_Buffer->free_Node[0].end(), File_Buffer->free_Node[i]);
			for (int j = 0; j < Thr_Num; j++) {
				//Path[j].clear();
				int cnt = SimulationPerMCTS / Thr_Num + 1 - pre_alloc[j].size();
				auto it = File_Buffer->free_Node[0].begin(); advance(it, min(cnt, File_Buffer->free_Node[0].size()));
				pre_alloc[j].splice(pre_alloc[j].end(), File_Buffer->free_Node[0], File_Buffer->free_Node[0].begin(), it);
				cnt = SimulationPerMCTS / Thr_Num + 1 - pre_alloc[j].size();
				for (int k = 0; k < cnt; k++) {
					auto p = new PINFO();
					pre_alloc[j].emplace_back(p);
				}
			}
		}
		void Select_Move(int&SelectMove, int&result) {
			//AllocNode();
			//environment_API->GetInsActionMask(ActionIsValid);
			//Batch_Size = rollout_Batch, Finish_Thr = 0;
			//Eva_CV.notify_one();
			//for (int t = 0; t < Thr_Num; t++)
			//	Thr[t] = new thread(&MCTS_Policy_Extend::Thread_Start, this, t, std::ref(Node[Step]), std::ref(Ins_PID), Ins_Feature, Step, (Mat*)NULL, -1, false);
			//for (int i = 0; i < Thr_Num; i++) {
			//	Thr[i]->join(); delete Thr[i];
			//}
			//PINFO*Root = Node[Step];

			////select Maximum visit count
			//MCTS_Edge*best_act = NULL; ull tot = 0;
			//ui move = Sampling_Visit(Root, best_act, Step, tot, NULL, 0);
			//Update_and_Write();

			//addStone(*best_act, move);
			//SelectMove = move;

			//assert(best_act->next_position_ID != -1);
			////create root Node
			//if (!environment_API->GetGameState(result) && Ins_PID == -1) {
			//	Batch_Size = 1; Eva_CV.notify_one();
			//	assert(best_act->next_Node == NULL);
			//	if (best_act->next_Node == NULL) {
			//		MC_Tree_Search(best_act->next_Node, best_act->next_position_ID, Ins_Feature, Step, NULL, -1, 0);
			//		Ins_PID = best_act->next_Node->p.root_ID_exist;
			//	}
			//	if (Ins_PID == -1)
			//		MC_Tree_Search(Node[Step], best_act->next_Node->p.root_ID_exist, Ins_Feature, Step, NULL, -1, 0);
			//	else assert(Ins_PID != -1);
			//}
		}
		void Opponent_Move(const int&opponent_move) {
			//if (Step > 0 || File_Buffer->IDCount[0] > 0)assert(Node[Step] != NULL || Ins_PID != -1);
			//Batch_Size = 1;Eva_CV.notify_one();
			//if (Node[Step] == NULL)
			//	MC_Tree_Search(Node[Step], Ins_PID, Ins_Feature, Step, NULL, -1, 0);
			////oppponent move
			//MCTS_Edge*best_act = NULL;
			//best_act = &(*Node[Step]->p.edges.edge)[opponent_move];

			//addStone(*best_act, opponent_move);
			////expand new Node
			//if (Ins_PID == -1) {
			//	//assert(best_act->next_position_flag == 2 && best_act->next_position_ID == -1 && best_act->next_Node == NULL);
			//	if (best_act->next_Node == NULL) {
			//		///maybe lost dynamic Reward(=0) when treated as representation
			//		MC_Tree_Search(best_act->next_Node, best_act->next_position_ID, Ins_Feature, Step, &Node[Step - 1]->p.Hidden_State, opponent_move, 0);
			//		Ins_PID = best_act->next_Node->p.root_ID_exist;
			//	}
			//	if (Ins_PID == -1)
			//		MC_Tree_Search(Node[Step], best_act->next_Node->p.root_ID_exist, Ins_Feature, Step, NULL, -1, 0);
			//}
		}

		void Select_Move_without_IO(int&SelectMove, int&result) {
			environment_API->GetInsActionMask(ActionIsValid);
			if (environment_API->GetGameState(&result))return;
			{
				Finish_Thr = 0;
				Eva_CV->notify_one();
			}
			IsAddNoise = false;
			/*if (SimulationPerMCTS == 2000)
				IsTest = true;
			else IsTest = false;*/
			//X Simulation from root
			for (int t = 0; t < Thr_Num; t++) {
				lock_guard<mutex>locker(Thr_m[t]);
				Thr_CV[t].notify_one();
			}
			{
				std::unique_lock<std::mutex> lock(Thr_lock);
				Main_CV.wait(lock, [this]() {return Finish_Thr == Thr_Num; });
			}

			PINFO*Root = Node[0];
			//select Maximum visit count
			MCTS_Edge*best_act = NULL; ull tot = 0;
			ui move = Sampling_Visit(Root, best_act, Step, tot, NULL, 0);
			Update_and_Clear();

			addStone(*best_act, move);
			SelectMove = move;
		}
		void Opponent_Move_without_IO(const int&opponent_move) {
			assert(Node[Step] == NULL && Ins_PID == -1);
			environment_API->GetInsActionMask(ActionIsValid);
			//oppponent move
			addStone(MCTS_Edge(), opponent_move);
			assert(Ins_PID == -1);
		}
	};
	struct Agent_Group {

		MCTS_Policy_Extend**MP;
		Agent**rollout_Agent = NULL;
		int MP_Count;
		int Thr_Num;

		thread*Eva_Loop = NULL;
		Agent_Group(int MCTS_Policy_Count, MCTS_Policy_Extend**MCTS_Policy, int Thr_Num, Agent**rollout_Agent, int rollout_Num) :MP_Count(MCTS_Policy_Count), MP(MCTS_Policy),
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
			Param**data = new Param*[Max_Thr];
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
					Evalution*tar = NULL; int flag = 0;
					for (int i = 0; i < rollout_Batch; i++) {
						if (it != agent_Q[id].end()) {
							tar = *it; flag = 0;
							it++;
						}//fullfill
						else flag = 3;
						agentResponse(flag, i, tar->H_State, tar->Action, tar->OutPut, &tar->Value, tar->Next_State, &tar->Reward, tar->step, rollout_Agent[id], MP[tar->PolicyID]->environment_API, data);
					}
					//run agent
					agentResponse(1, 0, tar->H_State, tar->Action, tar->OutPut, &tar->Value, tar->Next_State, &tar->Reward, tar->step, rollout_Agent[id], MP[tar->PolicyID]->environment_API, data);
					//OupPut
					it = agent_Q[id].begin();
					for (int i = 0; i < rollout_Batch; i++) {
						tar = *it;
						agentResponse(2, i, tar->H_State, tar->Action, tar->OutPut, &tar->Value, tar->Next_State, &tar->Reward, tar->step, rollout_Agent[id], MP[tar->PolicyID]->environment_API, data);
						it++;
						if (it == agent_Q[id].end())break;
					}
					agentResponse(4, 0, tar->H_State, tar->Action, tar->OutPut, &tar->Value, tar->Next_State, &tar->Reward, tar->step, rollout_Agent[id], MP[tar->PolicyID]->environment_API, data);
					//assert(it == agent_Q[id].end());
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
			thread*rollout_Loop[Max_Thr];
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
							else if (tar->Response[i] == 3 && -2 < tar->Eva_Stack[i].Value) {
								auto&it = tar->Eva_Stack[i];
								assert(it.ID == i);
								assert(abs(it.Value) <= 1 + 1e-8);
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
						else if ((*it)->H_State&&dyn != -1) {
							agent_Q[dyn].push_back(*it), dyn_flag = true, it = Q.erase(it);
						}
						//it++; sum_Node++;
					}
					//wake up agent
					for (int j = 0; j < rollout_Num; j++) {
						if ((rep_flag&&j == rep) || (dyn_flag&&j == dyn)) {
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
		MCTS_Policy_Extend*MCTS_Agent = NULL;
		Agent_Group*Group0 = NULL;
		HyperParamSearcher param;
		void MCTS_Agent_Init(const char*agent_Path, const char*agent_param_Path, const char*dir_path, Environment&e) {
			pipeOut("DEBUG MCTS Clear\n");
			std::experimental::filesystem::remove_all(((string)dir_path + "MCTS\\0\\").c_str());
			pipeOut("DEBUG MCTS Init\n");
			Init_MCTS(e, dir_path);
			param.Read_Param((string)agent_param_Path);
			if (!MCTS_Agent) {
				Agent*agent = new Agent(agent_Path, param["Batch"], param["Max_Step"], param["Max_srand_row"], true);
				const int rollout_Num = 3, rollout_Batch = 8;
				Agent**rollout = new Agent*[rollout_Num] { 0 };
				param["Batch"] = rollout_Batch; param["Max_Step"] = 1;
				for (int i = 0; i < rollout_Num; i++) {
					Net*net = i == 0 ? e.RepresentationNet(param) : e.DynamicsNet(param);
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
		string move_track[MaxGameLength];
		string MCTS_Agent_Response(string*moves, int moves_cnt) {
			pipeOut("DEBUG response");
			//detect takeback
			for (int i = 0; i < MCTS_Agent->Step; i++) {
				//change moves
				if (i >= moves_cnt || moves[i] != move_track[i]) {
					pipeOut("DEBUG take back");
					//MCTS_Agent->TakeBack(i, moves);
					MCTS_Agent_New_Game();
					break;
				}
			}
			DEBUG(moves_cnt >= MCTS_Agent->Step);
			while (moves_cnt > MCTS_Agent->Step) {
				pipeOut("DEBUG do %s", moves[MCTS_Agent->Step]);
				int mov = MCTS_Agent->environment_API->parse_action(moves[MCTS_Agent->Step]);
				move_track[MCTS_Agent->Step] = moves[MCTS_Agent->Step];
				MCTS_Agent->Opponent_Move_without_IO(mov);
			}

			int move = -1, result = 2;
			MCTS_Agent->Select_Move_without_IO(move, result);
			move_track[MCTS_Agent->Step - 1] = MCTS_Agent->environment_API->Encode_action(move);
			return move_track[MCTS_Agent->Step - 1];
		}
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
	void Go_Param(HyperParamSearcher&param) {
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
	const int Max_Data_Size = 8 * 8 * 15 * 15 * 40;// *2;// *2;
	void MCTS_generate_data_extend(int tid, MCTS_Policy_Extend*ag, const char*data_Path, int GamesPerAgent,int*com,int&tot_Count) {
		ll tot_step = 0, game_count = 0;
		clock_t start_time = clock();

		DataSet new_ds;
		new_ds.trainSet_Init(Max_Data_Size);
		new_ds.add_absorbing_Head(*ag->environment_API, ag->environment_API->getSimplifyActionSpace(), 1);
		while (true) {
			//execute MCTS,generate train data
			ag->MCTS_Start();
			//train RL agent
			int MaxStep = new_ds.trainSet_dataCount() + ag->GameStep;
			for (int i = 0; i < ag->GameStep; i++) {
				//repeat terminal state
				//1:black win(first hand) -1:white win 0:draw
				new_ds.trainSet_Add_data(Environment::Data(new_ds.trainSet_dataCount(), MaxStep, Screen_Size, ag->trainData[i].Scr, ag->environment_API->getSimplifyActionSpace(), ag->trainData[i].OutPut, ag->trainData[i].out_cnt, ag->trainData[i].move, i % 2 == 0 ? ag->GameResult : -ag->GameResult, ag->trainData[i].Action_Reward, *ag->environment_API));
				//Value direct from outcome with board games,otherwise bootstrap n=10
			}
			new_ds.gameCount++; tot_Count++;
			tot_step += ag->GameStep, game_count++;
			if (!((ag->GameResult == 1 && ag->GameStep % 2 == 0) || (ag->GameResult == -1 && ag->GameStep % 2 == 1)|| ag->GameResult==0))printf("error:result not match step\n"), assert(false);
			printf("tid:%d step:%d result:%d average_step:%.02lf tot_step:%d Count:%d time:%d min(s)\n", tid, ag->GameStep, ag->GameResult, 1.0 * tot_step / game_count, tot_step, tot_Count, (clock() - start_time) / CLOCKS_PER_SEC / 60);
			//Batch
			char ch = 0; bool save = false; int idx = 0;
			while (com[idx++] != -1) {
				if (com[idx - 1] > com[tid])save = true;
			}
			if (new_ds.trainSet_gameCount() % GamesPerAgent == 0 || (_kbhit() && (ch = _getch()) == 'l') || save) {
				ag->Write2File();
				//combine new_ds to ds
				DataSet* ds = new DataSet(); ds->trainSet_Save_Load(false, Max_Data_Size, data_Path);
				//add absorbing state first
				if (ds->dataCount == 0) {
					ds->add_absorbing_Head(*ag->environment_API, ag->environment_API->getSimplifyActionSpace(), 1);
				}
				
				auto* Comb = Combine_MP_ds(ds, &new_ds, *ag->environment_API);
				//absorbing
				if (Comb->trainSet_Param(0).In(0).MaxCount == 0) {
					Comb->complete_absorbing_Head();
				}
				Comb->trainSet_Save_Load(true, -1, data_Path);
				delete Comb;

				if (ch == 'l' || save)com[tid]++;
				break;
			}
		}
	}
	void MCTS_Reanalyze_Data(int tid, MCTS_Policy_Extend*ag, DataSet&ds, int gen_cnt) {
		clock_t start_time = clock();
		int ds_idx = 3, game_id = 0;
		while (ds_idx < ds.dataCount) {
			int data_len = 0, start_idx = ds_idx;
			while (true) {
				trajectory_Data&tra = ag->trainData[data_len];
				//Screen
				memcpy(tra.Scr, &ds.trainSet_Param(ds_idx).Out(0)[0], sizeof(double)*Screen_Size);
				//Policy
				int cnt = ds.trainSet_Param(ds_idx).Out(0).Count - Screen_Size - 3;
				assert(cnt > 0);
				memcpy(tra.OutPut, &ds.trainSet_Param(ds_idx).Out(0)[Screen_Size + 3], sizeof(double)*cnt);
				tra.out_cnt = cnt;

				data_len++; ds_idx++;
				if (cnt == 1)break;
			}
			if (game_id++%gen_cnt == tid) {
				//run lastest agent
				ag->Init_State();
				ag->Data_Reanalyze(0);

				//update Policy
				for (int i = 0; i < data_len; i++) {
					trajectory_Data&tra = ag->trainData[i];
					memcpy(&ds.trainSet_Param(start_idx + i).Out(0)[Screen_Size + 3], tra.OutPut, sizeof(double)*ag->environment_API->getSimplifyActionSpace());
				}
				printf("tid:%d game_id:%d time:%d min(s)\n", tid, game_id - 1, (clock() - start_time) / CLOCKS_PER_SEC / 60);
			}
		}
	}

	
	double MCTS_Match(MCTS_Policy_Extend*P1, MCTS_Policy_Extend*P2) {
		int move = -1, result = 2;

		P1->Init_State();
		P2->Init_State();
		while (true) {
			P1->Select_Move_without_IO(move, result);
			P1->environment_API->PrintScr();
			if (result != 2)break;

			P2->Opponent_Move_without_IO(move);
			P2->Select_Move_without_IO(move, result);
			P2->environment_API->PrintScr();
			if (result != 2)break;

			P1->Opponent_Move_without_IO(move);
		}
		return result;
	}
	void Sampling_DataSet(DataSet&train_ds, int data_Count, int recent_Max_data, int dataSet_Max_ID, Environment*e) {
		train_ds.trainSet_Init(data_Count*e->getMaxUnrolledStep());
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
				ds.trainSet_Save_Load(false, Max_Data_Size, Path);
				Start += ds.dataCount;
				dataSet_Max_ID--;
			}
			for (int j = 0; j < ds.trainSet_Param(Start).Count; j++) {
				train_ds.trainSet_Add_data(Environment::Data(e));
				train_ds.trainSet_Param(train_ds.gameCount).In(j) = ds.trainSet_Param(Start).In(j);
				train_ds.trainSet_Param(train_ds.gameCount).In(j)[0] = j + train_ds.gameCount*e->getMaxUnrolledStep();
				train_ds.trainSet_Param(train_ds.dataCount - 1).Out(0) = ds.trainSet_Param(ds.trainSet_Param(Start).In(j)[0]).Out(0);
			}
			train_ds.gameCount++;
		}
		tot_data_set_Count = train_ds.trainSet_dataCount();
		tot_train_set_Count = train_ds.gameCount;
		//train_ds.trainSet_Save_Load(true, -1, "trainSet_data_sampling");
	}
}
using namespace MCTS_extend;
Agent_API*Get_MCTS_Extend_API() {
	return new Agent_Extend();
}

//Player data test
	//DataSet _player_ds;
	//_player_ds.trainSet_Save_Load(false, 2673103, "Player_DataSet");
	////test player data
	//if (&_player_ds)
	//	cout << _player_ds.Test(agent, _player_ds.trainSet_dataCount()/5, Gomoku::MCTS_GomokuSimulation, TestScore) << endl;
void MCTS_Evaluation(int rollout_Num, Agent**rollout_agent, Agent**rollout_agent1, Environment&e1, Environment&e2) {
	Init_MCTS(e1, "");
	std::experimental::filesystem::remove_all(((string)"MCTS_Policy\\1\\").c_str());
	std::experimental::filesystem::remove_all(((string)"MCTS_Policy\\2\\").c_str());
	//CPU
	auto Player0 = new MCTS_Policy_Extend(0, &e1, 25, 1000, "1");
	auto Player1 = new MCTS_Policy_Extend(0, &e2, 25, 1000, "2");
	//GPU
	Agent_Group Group0(1, &Player0, Player0->Thr_Num, rollout_agent, rollout_Num);
	Agent_Group Group1(1, &Player1, Player1->Thr_Num, rollout_agent1, rollout_Num);
	Group0.run();
	Group1.run();

	int p0 = 0, p1 = 0;
	for (int t = 0; t < 100; t++) {
		double res = MCTS_Match(t % 2 == 0 ? Player0 : Player1, t % 2 == 0 ? Player1 : Player0);
		if ((res == 1 && t % 2 == 0) || (res == -1 && t % 2 == 1))printf("P0 win\n"), p0++;
		else if (res != 0)printf("P1 win\n"), p1++;
		else printf("Draw\n");
		printf("\nt:%02d res: %02d-%02d margin of %.02lf%%\n", t, p0, p1, 100.0*(p1 - p0) / p1);
	}
	
	//delete dir
	Player0->remove_dir();
	Player1->remove_dir();
	getchar();
	return;
}
const int Generator_Thr = 25;
void Generator_Proc(int tid, int Maximum_DataSet_Number, Environment*e, MCTS_Policy_Extend* &Main_ag, int*com) {
	//DataSet ds;
	char path[100]; sprintf(path, "extend_trainSet_data_thr%d", tid);
	//ds.trainSet_Save_Load(false, Max_Data_Size, path);
	//add absorbing state first
	//if (ds.dataCount == 0) {
		//ds.add_absorbing_Head(*e, 1);
	//}
	//thread*MCTS = new thread(&MCTS_generate_data, Main_ag, ref(ds), ref(train));
	char name[50]; sprintf(name, "thr_%d", tid);
	Main_ag = new MCTS_Policy_Extend(tid, e, Generator_Thr, 800, name, "D:\\MCTS_Policy\\");
	while (!Main_ag->Eva_CV)this_thread::sleep_for(std::chrono::milliseconds(100));
	int tot = 0;
	for (int t = 0; t < 1000; t++) {
		MCTS_generate_data_extend(tid, Main_ag, path, 800, com, tot);
	}
	//ds.Disponse();
	//free cuda stream
	//unBindStm();
}
void Data_Reanalyze_Proc(int tid, DataSet&ds, Environment*e, MCTS_Policy_Extend* &Main_ag, int gen_cnt) {
	char name[50]; sprintf(name, "thr_%d", tid);
	Main_ag = new MCTS_Policy_Extend(tid, e, Generator_Thr, 800, name, "D:\\MCTS_Policy\\");
	while (!Main_ag->Eva_CV)this_thread::sleep_for(std::chrono::milliseconds(100));
	MCTS_Reanalyze_Data(tid, Main_ag, ds, gen_cnt);
}
DataSet* Combine_MP_ds(DataSet*ds, DataSet*ds1, Environment&e) {
	DataSet* comb = ds;
	int offset = ds->dataCount;
	assert(ds->dataCount >= 3 && ds1->dataCount >= 3);
	//new dynamic DataSet size
	if (comb->MaxCount < ds->dataCount + ds1->dataCount) {
		comb = new DataSet();
		comb->trainSet_Init(ds->dataCount + ds1->dataCount - 3);
		for (int i = 0; i < ds->dataCount; i++) {
			comb->trainSet_Add_data(Environment::Data(&e));
			comb->trainSet_Param(comb->dataCount - 1) = ds->trainSet_Param(i);
		}
		comb->gameCount += ds->gameCount;
		delete ds;
	}
	for (int i = 3; i < ds1->dataCount; i++) {
		comb->trainSet_Add_data(Environment::Data(&e));
		for (int j = 0; j < e.getMaxUnrolledStep(); j++)
			if (ds1->trainSet_Param(i).In(j)[0] >= 3)ds1->trainSet_Param(i).In(j)[0] += offset - 3;
		comb->trainSet_Param(comb->dataCount - 1) = ds1->trainSet_Param(i);
	}
	comb->gameCount += ds1->gameCount;
	return comb;
}
DataSet* Combine(int l, int r, Environment&e) {
	int M = l + r >> 1;
	if (l >= r)return NULL;
	if (l + 1 == r) {
		char path[100]; sprintf(path, "extend_trainSet_data_thr%d", l);
		DataSet*ds = new DataSet(); ds->trainSet_Save_Load(false, Max_Data_Size, path);
		return ds;
	}
	auto*left = Combine(l, M, e);
	auto*right = Combine(M, r, e);
	if (!left) {
		assert(right); return right;
	}
	else {
		assert(left&&right);
		if (left->dataCount > 3 && right->dataCount > 3)
			left = Combine_MP_ds(left, right, e);
		delete right;
		return left;
	}
}
void RL_SelfPlay_with_MCTS_extend(int rollout_Num, Agent**rollout_agent, int agent_id, int Maximum_DataSet_Number, Environment**e, int Generator_Count,bool Reanalyze) {

	Init_MCTS(*e[0], "");
	int Agents_Count = agent_id;
	//while (true)
	{
		//multi CPU
		thread*thr[Max_Thr];
		MCTS_Policy_Extend**MP = new MCTS_Policy_Extend*[Generator_Count] {NULL};
		int communicate[Max_Thr + 1]; fill(communicate, communicate + Max_Thr + 1, -1);
		//GPU
		Agent_Group*Group = new Agent_Group(Generator_Count, MP, Generator_Thr, rollout_agent, rollout_Num);
		Group->run();
		if (!Reanalyze)
			for (int t = 0; t < Generator_Count; t++) {
				communicate[t] = 0;
				thr[t] = new thread(&Generator_Proc, t, Maximum_DataSet_Number, ref(e[t]), ref(MP[t]), communicate);
			}
		else {
			DataSet ds;
			char path[100]; sprintf(path, "extend_trainSet_data_%d", Maximum_DataSet_Number);
			ds.trainSet_Save_Load(false, Max_Data_Size, path);
			for (int t = 0; t < Generator_Count; t++)
				thr[t] = new thread(&Data_Reanalyze_Proc, t, ref(ds), ref(e[t]), ref(MP[t]), Generator_Count);
			for (int i = 0; i < Generator_Count; i++) {
				thr[i]->join();delete thr[i];
			}
			sprintf(path, "extend_trainSet_data_%d_re", Maximum_DataSet_Number);
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
template<typename Train_Func, typename NetFunc>
void async_Train(NetFunc NetFunc,Agent*agent,string ParamPath, AgentGenerator::Hyperparam hyperparam_Func,DataSet&ds, Train_Func trainFunc) {
	HyperParamSearcher param(ParamPath, ParamPath);
	if (hyperparam_Func)hyperparam_Func(param);
	Server_Init(2, agent, param);
	Server_InitData((Base_Param**)ds.dataSet, param["paramNum"], param["trainNum"]);
	Server_StartTrain(NetFunc, trainFunc);
	//Server_Test<Agent::Train_Option>(Cnt, Test);
	Server_Disponse(false);
}
mutex train_mux, data_mux;
DataSet* sampling_ptr;
condition_variable train_CV, data_CV;
void async_Sampling(int TrainDataNum, int Maximum_DataSet_Number, Environment& e) {
	std::unique_lock<std::mutex> data_locker(data_mux);
	while (true) {
		Sampling_DataSet(*sampling_ptr, 200000, TrainDataNum, Maximum_DataSet_Number, &e);
		{
			lock_guard<mutex>locker(train_mux);
			printf("swap dataset\n");
			train_CV.notify_one();
		}
		//this_thread::sleep_for(std::chrono::milliseconds(2000));
		//wait data written to cuda
		data_CV.wait(data_locker);
	}
}
void MCTS_Train_extend(Agent*agent, int agent_id, int Maximum_DataSet_Number, Environment&e, int Generator_Count, int TrainDataNum) {

	agent_id++;

	auto res = Combine(0, Generator_Count, e);
	if (res->dataCount) {
		char path[100]; sprintf(path, "extend_trainSet_data_%d", Maximum_DataSet_Number + 1);
		res->trainSet_Save_Load(true, -1, path);
		Maximum_DataSet_Number++;
		delete res;
	}

	int train_cnt = 0;
	double ori_Speed = agent->Net_Param["Speed"];
	DataSet train_ds;// , train_ds_swap;
	std::unique_lock<std::mutex> train_locker(train_mux);
	sampling_ptr = &train_ds;


	//optimize CPU memory new and free,Speed Up Train???? 
	new thread(&async_Sampling, TrainDataNum, Maximum_DataSet_Number, ref(e));
	while (true) {
		{
			//DataSet train_ds;
			agent->Net_Param["Speed"] = max(ori_Speed * pow(0.1, floor(train_cnt / agent->Net_Param["Speed_decay_time"])), 2e-5);
			int cnt = max(TrainDataNum / 200000, 1);
			while (cnt-- > 0) {
				train_CV.wait(train_locker);
				//Sampling_DataSet(train_ds, 300000, TrainDataNum, Maximum_DataSet_Number, &e);
				//async_Train(e.getJointNetFun(), agent, "Chess_best_param", Go_Param, train_ds, e.getTrainFun());
				sampling_ptr->miniTrain_Start(agent, &data_CV, Go_Param, e.getTrainFun());
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


