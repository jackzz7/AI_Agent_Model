#pragma once


#include"RL_MCTS.h"
#include"Game.h"
#include"Gomoku.h"

#include<set>
#include<time.h>
#include<io.h>
#include<direct.h>
#include<random>
#include<experimental/filesystem>
#include<type_traits>

//AI match API
/** write a line to STDOUT */
int pipeOut(const char *fmt, ...)
{
	int i;
	va_list va;
	va_start(va, fmt);
	i = vprintf(fmt, va);
	putchar('\n');
	fflush(stdout);
	va_end(va);
	return i;
}
#define DEBUG(expression)(void)(                                                       \
            (!!(expression)) ||                                                              \
            (pipeOut("ERROR expr:%s file:%s line:%u\n",#expression, __FILE__, (unsigned)(__LINE__)), 0)) \

//constexpr int W = 15;
//constexpr int H = 15;

//static const int rawSize = 8 * sizeof(ull);
//random number for stone
us Randomer[W][H][2];
enum Colour {
	Black, White
};
AgentResponse Eva_Response = NULL;
Judger GameJudger = NULL;
//extern Param**game_data[Cuda_Max_Stream];
void Init_MCTS(const char*dir = "", AgentResponse ResponseFunc = agentResponse, Judger GameJudger = GomokuJudger) {
	if (!Eva_Response) {
		for (int t = 0; t < Cuda_Max_Stream; t++) {
			game_data[t] = new Param*[128]{ 0 };
			for (int i = 0; i < 128; i++)game_data[t][i] = new Param();
		}
		Eva_Response = ResponseFunc;
		::GameJudger = GameJudger;
		fstream file; file.open((string)dir + "XOR_Random_Number", ios::in | ios::binary);
		if (file)file.read((char*)Randomer, W*H * 2 * sizeof(us)), file.close();
		else {
			for (int i = 0; i < W*H; i++)
				for (int j = 0; j < 2; j++)
					Randomer[i%W][i / W][j] = rand_i();
			file.close();
			file.open((string)dir + "XOR_Random_Number", ios::out | ios::binary);
			file.write((char*)Randomer, W*H * 2 * sizeof(us)), file.close();
		}
	}
}
//get priority_queue's underlying vector Container
template<class T, class S, class C>
S& getContainer(priority_queue<T, S, C>&pq) {
	struct HackedQueue :private priority_queue<T, S, C> {
		static S& Container(priority_queue<T, S, C>&pq) {
			return pq.*&HackedQueue::c;
		}
	}; return HackedQueue::Container(pq);
}
struct PInfo;
static const int bits = 4;//16;
static const int Maximum_Cache = 1 << bits;
static const int BufferSize = 400;
//static const int PositionPerCache = 10;
//static const int MaximumRawCache = 1 << 10;
//static const int Inital_Edge_Num = W * H;

struct MCTS_Edge {
	uc move;
	volatile char next_position_flag;
	volatile float tot_Value;
	//traverse Count
	volatile int Visit_Count;
	//control explore
	volatile float Prior_Probability;
	//link next position ID
	int next_position_ID;

	volatile float Q_Value;
	PInfo*volatile next_Node;


	MCTS_Edge() {
		move = 0;
		Q_Value = 0;
		tot_Value = 0;
		Visit_Count = 0;
		Prior_Probability = -1;
		next_position_ID = -1;
		next_position_flag = 2;
		next_Node = NULL;
	}
	MCTS_Edge(const double&Priori, int move, int next_ID = -1) :MCTS_Edge() {
		Prior_Probability = Priori;
		this->move = move;
		next_position_ID = next_ID;
	}
	pi getMov()const {
		return { move%W,move / W };
	}
	int getX()const { return move % W; }
	int getY()const { return move / W; }

	static const double Cpuct;
	double Confidence_Bound(const double&sqrt_tot_Node_visit) const {
		return Q_Value + Cpuct * Prior_Probability*sqrt_tot_Node_visit / (1 + Visit_Count);
	}
	MCTS_Edge&operator=(const MCTS_Edge&e) {
		memcpy(this, &e, sizeof(MCTS_Edge));
		return *this;
	}
	void Update_Q_Value() {
		if (Visit_Count > 0)
			Q_Value = tot_Value / Visit_Count;
		else Visit_Count = 0;
	}
	void Virtual_Loss(const int&vl) {
		tot_Value -= vl;
		Visit_Count += vl;
		if (Visit_Count > 0)
			Q_Value = tot_Value / Visit_Count;
		else Visit_Count = Q_Value = 0;
	}
	void restore_Virutal_Loss(const int&vl, const std::pair<float, ui>&Visted) {
		tot_Value += vl + Visted.X;
		Visit_Count += -vl + Visted.Y;
		Q_Value = tot_Value / Visit_Count;
	}
	const static int Edge_Size = 2 * sizeof(uc) + 4 * sizeof(int);
	void Read(const char*src) {
		int _i = 0;
		memcpy(&move, &src[_i], sizeof(move)); _i += sizeof(move);
		memcpy((void*)&next_position_flag, &src[_i], sizeof(next_position_flag)); _i += sizeof(next_position_flag);
		memcpy((void*)&tot_Value, &src[_i], sizeof(tot_Value)); _i += sizeof(tot_Value);
		memcpy((void*)&Visit_Count, &src[_i], sizeof(Visit_Count)); _i += sizeof(Visit_Count);
		memcpy((void*)&Prior_Probability, &src[_i], sizeof(Prior_Probability)); _i += sizeof(Prior_Probability);
		memcpy((void*)&next_position_ID, &src[_i], sizeof(next_position_ID)); _i += sizeof(next_position_ID);
		assert(_i == Edge_Size);
	}
	void Write(char*dst) const {
		int _i = 0;
		memcpy(&dst[_i], &move, sizeof(move)); _i += sizeof(move);
		memcpy(&dst[_i], (void*)&next_position_flag, sizeof(next_position_flag)); _i += sizeof(next_position_flag);
		memcpy(&dst[_i], (void*)&tot_Value, sizeof(tot_Value)); _i += sizeof(tot_Value);
		memcpy(&dst[_i], (void*)&Visit_Count, sizeof(Visit_Count)); _i += sizeof(Visit_Count);
		memcpy(&dst[_i], (void*)&Prior_Probability, sizeof(Prior_Probability)); _i += sizeof(Prior_Probability);
		memcpy(&dst[_i], (void*)&next_position_ID, sizeof(next_position_ID)); _i += sizeof(next_position_ID);
		assert(_i == Edge_Size);
	}
};
//static const int Edge_Size = sizeof(MCTS_Edge) - sizeof(float) - sizeof(PInfo*);

struct EDGE {
	vector<MCTS_Edge>*edge;
	int sz;
	EDGE() {
		edge = NULL;
		sz = 0;
	}
	EDGE(int Size) :EDGE() {
		edge = new vector<MCTS_Edge>();
		//pre-alloc memory
		edge->resize(W*H);
	}
	~EDGE() {
		delete edge;
	}
	//get_Edge_with_Maximum_UCB
	MCTS_Edge&top(volatile ull&tot_Visit,mt19937&rng) {
		++tot_Visit;
		assert(sz > 0);
		const int Cbase = 19652, Cinit = 1;
		double sqrt_tot_visit = sqrt(tot_Visit)*(log((1.0 + tot_Visit + Cbase) / Cbase) + Cinit);
		int idx = rng() % sz;
		assert(0 <= idx);
		MCTS_Edge*best = &(*edge)[idx]; double mx = (*edge)[idx].Confidence_Bound(sqrt_tot_visit);
		best->Virtual_Loss(1);
		for (auto&k : *edge) {
			if (k.Prior_Probability == -1)break;
			double tar = k.Confidence_Bound(sqrt_tot_visit);
			if (mx < tar)k.Virtual_Loss(1), best->Virtual_Loss(-1), best = &k, mx = tar;
		}assert(best->Prior_Probability >= 0);
		return *best;
	}
	/*void clear() {
		edge->clear();
		sz = 0;
	}*/
	void push(const MCTS_Edge&e) {
		assert(sz < W*H);
		(*edge)[sz] = e;
		(*edge)[sz].Update_Q_Value();
		if (sz + 1 < W*H)(*edge)[sz + 1].next_position_ID = -1;
		sz++;
	}
	//end flag
	void push_end() {
		if (sz < W*H) {
			(*edge)[sz] = MCTS_Edge(-1, 0, -2);
		}
	}
	bool Ready() {
		return (sz == W * H) || (*edge)[sz].next_position_ID == -2;
	}
	void check(const MCTS_Edge&e) {
		for (auto&k : *edge) {
			if (k.Prior_Probability == -1)break;
			assert(k.move != e.move);
		}
	}
	bool empty() {
		return sz == 0;
	}
	EDGE&operator=(EDGE&&e) {
		swap(this->edge, e.edge);
		swap(this->sz, e.sz);
		return *this;
	}
	/*EDGE&operator=(EDGE&e) {
		swap(*this, e);
		return *this;
	}*/
};

struct Position {
	//ID+Feature=Identity
	int PID;
	//find UCB and traverse all edges
	//larger constant,only faster when edges>1000
	//std::priority_queue<MCTS_Edge>edges;
	EDGE edges;

	Position() {
		PID = -1;
	}
	Position(const int&ID) :edges(0) {
		PID = ID;
	}
	//add or remove stone
	static void addStone(us&Feature, const pi&pos, const Colour&colour) {
		Feature = (Feature ^ Randomer[pos.X][pos.Y][colour]) % Maximum_Cache;
	}
	static void addStone(us&Feature, int pos, const Colour&colour) {
		Feature = (Feature ^ Randomer[pos%W][pos / W][colour]) % Maximum_Cache;
	}
	static void addStone(ull*board, int pos, const Colour&colour) {
		board[pos * 2 / 64] |= (colour + 1LL) << (pos * 2 % 64);
	}
	static void removeStone(ull*board, int pos) {
		board[pos * 2 / 64] |= 3LL << (pos * 2 % 64);
		board[pos * 2 / 64] ^= 3LL << (pos * 2 % 64);
	}
};
struct PInfo {
	Position p;
	ull totVisit;
	bool Virtual_File_Flag;
};
//using PInfo = pair<pair<Position, bool>,volatile ull>;
//typedef list<pair<PInfo*volatile*, PInfo*volatile>> PList;
typedef list<PInfo*volatile> PList;
const double MCTS_Edge::Cpuct = 1.0;
static bool rawPositionIsEqual(const ull*board, const ull*board1) {
	return memcmp(board, board1, 8 * sizeof(ull)) == 0;
}
struct edgecmp {
	bool operator()(pair<MCTS_Edge, int>*const&a, pair<MCTS_Edge, int>*const&b)const {
		if (a->Y != b->Y)return a->Y < b->Y;
		else return a->X.move < b->X.move;
	}
};
//struct rawcmp {
//	//rawPosition
//	bool operator()(const ull* const&a, const ull* const&b)const {
//		int res = memcmp(a, b, 8 * sizeof(ull));
//		if (res < 0)return true;
//		else if (res > 0)return false;
//		else return false;
//	}
//};
struct Train_Data {
	int move;
	//probability
	double OutPut[W*H];
}trainData[W*H];
int GameResult, GameStep;


struct WR_Manager {
	int BufferMaxSize = BufferSize;
	//recent Position buffer
	//Position,rawPosition,VisitCount
	PList UpdateBuff[Maximum_Cache];
	//new Position Cache
	PList NewPositionBuff[Maximum_Cache];

	map<ui, PList::iterator>PositionCache[Maximum_Cache];
	//new edge buff,need rewrite whole file
	//ID,edge
	//map<int, list<MCTS_Edge>>NewEdgeBuff[Maximum_Cache];
	//rawPosition,ID
	//map<const ull*, int, rawcmp>rawPosBuff[Maximum_Cache];
	//binary search root cache
	//pair<const ull*, int>rawPosCache[Maximum_Cache][MaximumRawCache];
	int IDCount[Maximum_Cache];
	

	//string DirPath;
	char dir[100];
	WR_Manager(const char*dir_path,const char*Name) {
		if (_access(dir_path, 0) != 0) _mkdir(dir_path);
		sprintf(dir, "%s%s\\", dir_path, Name);
		if (_access(dir, 0) != 0) _mkdir(dir);
		for (int t = 0; t < Maximum_Cache; t++) {
			PositionCache[t].clear();
			UpdateBuff[t].clear();
			NewPositionBuff[t].clear();
			IDCount[t] = 0;
			ifstream file;
			char Name[100]; sprintf(Name, "%s%d_idx", dir, t);
			file.open(Name, ios::binary | ios::in);
			if (file) {
				file.seekg(0, ios::end);
				IDCount[t] = file.tellg() / sizeof(ull);
				assert(file.tellg() % sizeof(ull) == 0);
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
	pair<PInfo*volatile,int> InitPosition(const int&ID, const us&Feature, const ull*rawPosition, ifstream&file, ui ReadSize, list<PInfo*volatile>&alloc) {
		char buff[sizeof(uc) + W * H * MCTS_Edge::Edge_Size];
		auto _new = alloc.front();
		_new->p.PID = ID;_new->p.edges.sz = 0; _new->totVisit = 0; _new->Virtual_File_Flag = false;
		if (file.is_open()) {
			if (file.read(buff, ReadSize).gcount() <= 0) {
				assert(false); file.close(); 
				PInfo*volatile ret = NULL; return { ret, -1 };
			}
			ui cnt = 0; uc edge_Num = 0;
			memcpy(&edge_Num, &buff[cnt], sizeof(uc));
			assert(0 < edge_Num&&edge_Num <= W * H);
			cnt += sizeof(uc);
			for (int i = 0; i < edge_Num; i++) {
				MCTS_Edge e; e.Read(&buff[cnt]);//memcpy(&e, &buff[cnt], Edge_Size);
				cnt += MCTS_Edge::Edge_Size;
				_new->p.edges.push(e);
				if (e.next_position_flag != 2)
					_new->totVisit += e.Visit_Count;
				assert(e.next_position_flag != 3 || e.next_position_ID != -1);
			}
			_new->p.edges.push_end();
		}
		else assert(false);
		file.close();
		return { _new,1 };
	}
	pair<PInfo*, int> ReadPosition(int&ID, const ull*rawPosition, const us&Feature, list<PInfo*volatile>&alloc, PInfo*volatile&Node) {
		////check ID Buffer
		//if (ID == -1) {
		//	if (rawPosBuff[Feature].find(rawPosition) != rawPosBuff[Feature].end())
		//		ID = rawPosBuff[Feature][rawPosition];
		//}
		////find ID using binary search sorted rawPositions file
		//if (ID == -1) {
		//	ifstream rawfile; ull rawPos[8];
		//	char rawName[40]; sprintf(rawName, "%s%u_raw", dir, Feature);
		//	int l = 0, r = 0, res = -1, raw_idx = 0, _ID = -1;
		//	if ((r = IDCount[Feature] - rawPosBuff[Feature].size()) > 0) {
		//		const ui _rawSize = rawSize + sizeof(int); char buf[_rawSize];
		//		while (l < r) {
		//			int M = l + r >> 1; _ID = -1;
		//			//search cache
		//			if (raw_idx < MaximumRawCache) {
		//				auto&p = rawPosCache[Feature][raw_idx];
		//				if (p.X) {
		//					memcpy(rawPos, p.X, rawSize); _ID = p.Y;
		//				}
		//				assert(p.Y != -1);
		//			}
		//			else raw_idx = MaximumRawCache;
		//			//search file
		//			if (_ID == -1) {
		//				if (Node)return { Node,-1 };
		//				if (!rawfile.is_open())
		//					rawfile.open(rawName, ios::binary | ios::in);
		//				rawfile.seekg(M*_rawSize);
		//				rawfile.read((char*)buf, _rawSize);
		//				memcpy(rawPos, buf, rawSize);
		//				memcpy(&_ID, &buf[rawSize], sizeof(int));
		//			}
		//			if ((res = memcmp(rawPosition, rawPos, rawSize)) > 0)l = M + 1, raw_idx = (raw_idx << 1) + 2;
		//			else r = M, raw_idx = (raw_idx << 1) + 1;
		//			//find success
		//			if (res == 0) {
		//				ID = _ID; assert(ID != -1); break;
		//			}
		//		}
		//		rawfile.close();
		//	}
		//}
		assert(ID < IDCount[Feature]);

		if (ID != -1) {
			//check virtual file
			//NewPosition
			if (!Virtual_File_New[Feature].empty() && ID >= Virtual_File_New[Feature].front()->p.PID) {
				if (ID <= Virtual_File_New[Feature].back()->p.PID)
					return { Virtual_File_New[Feature][ID - Virtual_File_New[Feature].front()->p.PID] ,5 };
			}
			//UpdateBuffer
			//map<ui, PList::value_type>::const_iterator it;
			//if ((it = Virtual_File_UB[Feature].find(ID)) != Virtual_File_UB[Feature].end())
				//return { it->Y,5 };

			//locate file content
			ifstream file; ui ReadSize = 0; ull pos[2];
			if (!Node) alloc.front()->p.edges.sz = 0, Node = alloc.front();
			else return { Node,-1 };
			int FilePosCount = IDCount[Feature] - NewPositionBuff[Feature].size();
			assert(ID < FilePosCount);
			char Name[100]; sprintf(Name, "%s%u_idx", dir, Feature);
			file.open(Name, ios::binary | ios::in);
			if (file) {
				file.seekg(ID * sizeof(ull));
				ui _sz = file.read((char*)&pos, 2 * sizeof(ull)).gcount();
				file.close(); sprintf(Name, "%s%u", dir, Feature);
				file.open(Name, ios::binary | ios::in);
				file.seekg(pos[0]);
				//last one ID
				if (_sz == sizeof(ull))ReadSize = sizeof(uc) + W * H * MCTS_Edge::Edge_Size;
				else if (_sz == 2 * sizeof(ull))ReadSize = pos[1] - pos[0];
				else assert(false);
				assert(ReadSize <= sizeof(uc) + W * H * MCTS_Edge::Edge_Size);
			}
			else assert(false), file.close();
			//Initial from File
			return InitPosition(ID, Feature, rawPosition, file, ReadSize, alloc);
		}
		//create new Position
		else {
			auto p = alloc.front();
			p->p.PID = -1;
			p->p.edges.sz = 0; p->totVisit = 0; p->Virtual_File_Flag = false;
			return { p,0 };
		}
	}
	class Buffer {
		char*data;
		ui cnt, rcnt;
		fstream file;
		enum Mode :int;
		Mode mode;
		static const ui MaxSize = 1e5;
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
		void Init(const char*Path, Mode mode, int extraFlag = 0) {
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
		void seekp(ull pos, int way = ios::beg) {
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
	//edge,ID
	//set<pair<MCTS_Edge, int>*, edgecmp>Q;
	//void reWrite(const us&Feature) {
	//	if (NewEdgeBuff[Feature].empty())return;
	//	char Name[40], Name1[40];
	//	sprintf(Name, "%s%u", dir, Feature);sprintf(Name1, "%stmp", dir);
	//	string path = (string)Name + "_idx";
	//	Buffer file(Name, Buffer::Read), new_file(Name1, Buffer::Write, ios::trunc);
	//	Buffer new_file_idx(path.c_str(), Buffer::Write, ios::trunc);
	//	int cnt = IDCount[Feature] - NewPositionBuff[Feature].size();
	//	ui pos_idx = 0; auto it = NewEdgeBuff[Feature].begin();
	//	for (int i = 0; i < cnt; i++) {
	//		auto it = NewEdgeBuff[Feature].find(i);
	//		uc edge_Num; file.read(&edge_Num, sizeof(uc));
	//		uc num = edge_Num + (it != NewEdgeBuff[Feature].end() ? it->Y.size() : 0);
	//		new_file.write(&num, sizeof(uc));
	//		for (int j = 0; j < edge_Num; j++) {
	//			MCTS_Edge e; file.read(&e, Edge_Size);
	//			new_file.write(&e, Edge_Size);
	//		}
	//		if (it != NewEdgeBuff[Feature].end()) {
	//			for (auto&k : it->Y) {
	//				new_file.write(&k, Edge_Size);
	//			}
	//			NewEdgeBuff[Feature].erase(it);
	//		}			
	//		//write idx
	//		new_file_idx.write(&pos_idx, sizeof(ui));
	//		pos_idx += sizeof(uc) + num * Edge_Size;
	//	}
	//	assert(NewEdgeBuff[Feature].empty());
	//	file.close(); new_file.close();
	//	new_file_idx.close();
	//	remove(Name);rename(Name1, Name);
	//}
	/*map<int, int>Cache_map;
	void dfs(int idx,int l,int r){
		if (idx >= MaximumRawCache)return;
		if (l >= r)return;
		int M = l + r >> 1;
		Cache_map[M] = idx;
		dfs((idx << 1) + 1, l, M);
		dfs((idx << 1) + 2, M + 1, r);
	}*/
	//void WriteRawPosition(const us&Feature) {
	//	if (rawPosBuff[Feature].empty())return;
	//	char Name[40], Name1[40];
	//	sprintf(Name, "%s%u_raw", dir, Feature); sprintf(Name1, "%stmp1", dir);
	//	Buffer raw_file(Name, Buffer::Read), new_raw_file(Name1, Buffer::Write, ios::trunc);
	//	//linear combine rawPosition
	//	int cnt = IDCount[Feature] - rawPosBuff[Feature].size();
	//	ull raw[8]; int res = 1, _cnt = 0, ID;
	//	auto it = rawPosBuff[Feature].begin();
	//	Cache_map.clear();
	//	dfs(0, 0, IDCount[Feature]);
	//	for (int i = 0; i < IDCount[Feature]; i++) {
	//		pair<const ull*, int> p = { NULL,-1 };
	//		if (it != rawPosBuff[Feature].end())p = *it;
	//		if (_cnt < cnt) {
	//			if (res == 1) {
	//				raw_file.read(raw, rawSize);
	//				raw_file.read(&ID, sizeof(int));
	//			}
	//		}
	//		if (p.X&& _cnt < cnt)
	//			res = memcmp(p.X, raw, rawSize), assert(res != 0);
	//		else if (p.X)res = -1;
	//		else if (_cnt < cnt)res = 1;
	//		else assert(false);
	//		if (res > 0) {
	//			if (Cache_map.find(i) != Cache_map.end()) {
	//				auto&r = rawPosCache[Feature][Cache_map[i]];
	//				memcpy((void*)r.X, raw, rawSize);
	//				r.Y = ID;
	//			}
	//			new_raw_file.write(raw, rawSize);
	//			new_raw_file.write(&ID, sizeof(int));
	//			_cnt++;
	//		}
	//		else {
	//			new_raw_file.write(p.X, rawSize);
	//			new_raw_file.write(&p.Y, sizeof(int));
	//			if (Cache_map.find(i) != Cache_map.end()) {
	//				auto&r = rawPosCache[Feature][Cache_map[i]];
	//				memcpy((void*)r.X, p.X, rawSize);
	//				r.Y = p.Y;
	//			}
	//			delete[] it->X;
	//			it = rawPosBuff[Feature].erase(it);
	//		}
	//	}assert(it == rawPosBuff[Feature].end());
	//	rawPosBuff[Feature].clear();
	//	raw_file.close(); new_raw_file.close();
	//	remove(Name); rename(Name1, Name);
	//}
	Buffer _file[Maximum_Cache][2];
	void WriteNewPosition(const us&Feature, PList&NewPositionBuff) {
		if (NewPositionBuff.empty())return;
		char Name[100]; sprintf(Name, "%s%u", dir, Feature);
		string path = (string)Name + "_idx";
		Buffer&file = _file[Feature][0], &idx_file = _file[Feature][1];
		file.Init(Name, Buffer::Write, ios::app), idx_file.Init(path.c_str(), Buffer::Write, ios::app);
		file.seekp(0, ios::end);
		ull pos_idx = file.tellp();
		char buff[sizeof(uc) + W * H * MCTS_Edge::Edge_Size];
		for (const auto&it : NewPositionBuff) {
			auto&e = it->p.edges;
			uc num = e.sz, _num = 0;
			file.write(&num, sizeof(uc));
			assert(0 < num <= W * H);
			for (const auto&k : *e.edge) {
				if (k.Prior_Probability == -1)break;
				k.Write(buff);
				file.write(buff, MCTS_Edge::Edge_Size);
				_num++;
			}
			assert(num == _num);
			assert(it->p.PID != -1);
			//write idx
			idx_file.write(&pos_idx, sizeof(ull));
			pos_idx += sizeof(uc) + num * MCTS_Edge::Edge_Size;
		}
		file.close();
		idx_file.close();
	}
	void UpdatePosition(const us&Feature, PList&UpdateBuff) {
		if (UpdateBuff.empty())return;
		char Name[100]; sprintf(Name, "%s%u", dir, Feature);
		string path = (string)Name + "_idx";
		Buffer&file = _file[Feature][0];
		file.Init(Name, Buffer::Write, ios::in);
		ifstream idx_file(path.c_str(), ios::in | ios::binary);
		char buff[sizeof(uc) + W * H * MCTS_Edge::Edge_Size];
		for (const auto&it : UpdateBuff) {
			idx_file.seekg(it->p.PID * sizeof(ull));
			ull pos; idx_file.read((char*)&pos, sizeof(ull));
			file.seekp(pos);
			uc num = it->p.edges.sz, _num = 0;
			file.write(&num, sizeof(uc));
			assert(0 < num <= W * H);
			for (const auto&k : *it->p.edges.edge) {
				if (k.Prior_Probability == -1)break;
				k.Write(buff);
				file.write(buff, MCTS_Edge::Edge_Size);
				_num++;
			}
			assert(num == _num);
			assert(it->p.PID != -1);
		}
		file.close();
		idx_file.close();
	}

	list<PInfo*volatile> free_Node[Maximum_Cache];
	thread*File_thread[Maximum_Cache] = { NULL };
	PList _UpdateBuff[Maximum_Cache], _NewPositionBuff[Maximum_Cache];
	vector<PInfo*> Virtual_File_New[Maximum_Cache];
	//map<ui, PList::value_type>Virtual_File_UB[Maximum_Cache];
	//PList Virtual_File_New[Maximum_Cache];
	//int VF_StartID[Maximum_Cache];
	//int virtual_file_UB_Count[Maximum_Cache] = { 0 }, virtual_file_new_Count[Maximum_Cache] = { 0 };
	//PList::const_iterator _debug_ub_end[Maximum_Cache], _debug_new_end[Maximum_Cache];

	/*void File_Thread_End() {
		
	}*/
	void File_Thread_Start() {
		for (int i = 0; i < Maximum_Cache; i++) {
			File_thread[i] = new thread([this, i]() {
				UpdatePosition(i, _UpdateBuff[i]); //_UpdateBuff[i].clear();
				WriteNewPosition(i, _NewPositionBuff[i]); //_NewPositionBuff[i].clear();
			});
		}
		for (int i = 0; i < Maximum_Cache; i++) {
			for (auto&e : _UpdateBuff[i])
				for (auto&k : *e->p.edges.edge)assert(k.next_position_flag != 3 || k.next_position_ID != -1);
			for (auto&e : _NewPositionBuff[i])
				for (auto&k : *e->p.edges.edge)assert(k.next_position_flag != 3 || k.next_position_ID != -1);
		}
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
		//Virtual_File_New[i].remove_if([](auto const&e) {return !e->Virtual_File_Flag; });
		//free_Node[i].splice(free_Node[i].end(), Virtual_File_New[i]);
		/*_UpdateBuff[i].remove_if([](auto const&e) {return !(e.Y)->Virtual_File_Flag; });
		_NewPositionBuff[i].remove_if([](auto const&e) {return !(e.Y)->Virtual_File_Flag; });
		for (auto&e : _UpdateBuff[i]) assert(*e.X == e.Y&&e.Y->Virtual_File_Flag), free_Node[i].push_back(e.Y), *e.X = NULL;
		for (auto&e : _NewPositionBuff[i]) assert(*e.X == e.Y&&e.Y->Virtual_File_Flag), free_Node[i].push_back(e.Y), *e.X = NULL;
		_UpdateBuff[i].clear();
		_NewPositionBuff[i].clear();*/
		//for (auto&k : Virtual_File[i])free_Node[i].push_back(k.Y);
		//Virtual_File[i].clear();
		/*auto it = UpdateBuff[i].begin(); advance(it, virtual_file_UB_Count[i]);
		if (virtual_file_UB_Count[i] > 0)assert(prev(it) == _debug_ub_end[i]);
		for (auto _it = UpdateBuff[i].begin(); _it != it;) {
			if (!(*_it)->X.Y)_it = UpdateBuff[i].erase(_it);
			else PositionCache[i].erase((*_it)->X.X.PID), _it++;
		}
		free_Node.splice(free_Node.end(), UpdateBuff[i], UpdateBuff[i].begin(), it);

		it = NewPositionBuff[i].begin(); advance(it, virtual_file_new_Count[i]);
		if (virtual_file_new_Count[i] > 0)assert(prev(it) == _debug_new_end[i]);
		for (auto _it = NewPositionBuff[i].begin(); _it != it;) {
			if (!(*_it)->X.Y)_it = NewPositionBuff[i].erase(_it);
			else PositionCache[i].erase((*_it)->X.X.PID), _it++;
		}
		free_Node.splice(free_Node.end(), NewPositionBuff[i], NewPositionBuff[i].begin(), it);
		virtual_file_UB_Count[i] = 0, virtual_file_new_Count[i] = 0;*/
		//}
	//}
	}
	void RecycleBuffer(int i) {
		End_File_Write(i);
		//move Node
		//for (int i = 0; i < Maximum_Cache; i++) {
		auto it = UpdateBuff[i].begin();
		assert(_UpdateBuff[i].empty());
		advance(it, max((int)UpdateBuff[i].size() - BufferMaxSize*2, 0));// virtual_file_UB_Count[i]); //if (it != UpdateBuff[i].begin())_debug_ub_end[i] = prev(it);
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
	pair<PInfo*, int> getPosition(WR_Manager&File_Buffer, int&ID, const ull*rawPosition, const us&Feature, list<PInfo*volatile>&alloc, PInfo*volatile&Node) {
		if (ID != -1) {
			//find MostVisit
			for (auto&it : MostViste[Feature]) {
				if (it->p.PID == ID) {
					assert(it->Virtual_File_Flag == false);
					return { it,4 };
				}
			}
			//find RecentFreq
			map<ui, PList::iterator>::const_iterator it;
			if ((it = RF_map[Feature].find(ID)) != RF_map[Feature].end()) {
				assert((*it->Y)->Virtual_File_Flag == false);
				return { *it->Y,3 };
			}

			//find Update buffer and newPosition buffer
			if ((it = File_Buffer.PositionCache[Feature].find(ID)) != File_Buffer.PositionCache[Feature].end()) {
				//assert((*it->Y)->Virtual_File_Flag == false);
				//virtual file
				if ((*it->Y)->Virtual_File_Flag)
					return { *it->Y,5 };
				else return { *it->Y,2 };
			}
		}
		//Create or Initial from file
		return File_Buffer.ReadPosition(ID, rawPosition, Feature, alloc, Node);
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
void Insert_Edge(const us&Feature, ull*rawPosition, int step, EDGE*edge, const double&prior, int move) {
	edge->push(MCTS_Edge(prior, move, -1));
}

class MCTS_Policy {

	Agent**rollout_Agent = NULL;
	int rollout_Num = 0;
	//temporary stay in memory,interface with file
	WR_Manager*File_Buffer;
	PositionLookUp*LookUpTable;

	//resign
	double V_resign, Enable_resign;
	double Min_V_resign[2];
	static const int resign_MaxSize = 10000;
	static const int resign_MinSize = 5000;
	list<double>recent_resign;
	multiset<double>resign_Sort_Set;
public:
	MCTS_Policy(Agent**rollout,int rollout_Num,int Thr = 1, int Sim = 500, const char*Name = "0",const char*dir="MCTS_Policy\\") :rollout_Agent(rollout),Thr_Num(Thr),SimulationPerMCTS(Sim),rollout_Num(rollout_Num) {
		File_Buffer = new WR_Manager(dir, Name);
		LookUpTable = new PositionLookUp();
		for (int j = 0; j < Max_Thr; j++) {
			Ins_Position[j] = new int*[W];
			for (int i = 0; i < W; i++)
				Ins_Position[j][i] = new int[H] {0};
		}
		Eva_Loop_Start();

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
	~MCTS_Policy() {
		Eva_Loop_end();
		delete File_Buffer;
		delete LookUpTable;
		for (int j = 0; j < Max_Thr; j++) {
			pre_alloc[j].remove_if([](auto const&e) {delete e; return true; });
			for (int i = 0; i < W; i++)
				delete[] Ins_Position[j][i];
			delete[] Ins_Position[j];
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

	void Update_and_Write() {
		//recycle alloc
		int sum = 0;
		for (int i = 0; i < Thr_Num; i++) {
			for (auto&k : Path[i]) {
				if (k._new&&k.node != k._new)
					pre_alloc[i].push_back(k._new);
				sum++;
			}
		}
		for (int i = 0; i < Thr_Num; i++) {
			for (auto&k : Path[i]) {
				_Path[k.feature].emplace_back(k);
			}Path[i].clear();
		}
		for (int i = 0; i < Maximum_Cache; i++)sum -= _Path[i].size();
		assert(sum == 0);
		//File_Buffer->File_Thread_End();
		thread*io[Maximum_Cache];
		for (int i = 0; i < Maximum_Cache; i++) {
			io[i] = new thread([this, i]() {
				UpdatePath(i);
				//most visited Update move to MV
				LookUpTable->UB2MV(*File_Buffer, i);
				//async write to File
				File_Buffer->RecycleBuffer(i);
				});
		}
		for (int i = 0; i < Maximum_Cache; i++)
			io[i]->join(), delete io[i];
		File_Buffer->File_Thread_Start();
	}
	//write all data to file
	void Write2File() {
		Eva_Loop_end();
		resign_WR(true);
		//File_Buffer->File_Thread_End();
		for (int i = 0; i < Maximum_Cache; i++)
			File_Buffer->End_File_Write(i);
		//File_Buffer->BufferMaxSize = 0;
		for (int i = 0; i < Maximum_Cache; i++) {
			//move all to buffer
			File_Buffer->UpdateBuff[i].splice(File_Buffer->UpdateBuff[i].end(), LookUpTable->MostViste[i]);
			File_Buffer->UpdateBuff[i].splice(File_Buffer->UpdateBuff[i].end(), LookUpTable->RecentFrequent[i]);
			LookUpTable->RF_map[i].clear();

			File_Buffer->UpdatePosition(i, File_Buffer->UpdateBuff[i]);
			File_Buffer->WriteNewPosition(i, File_Buffer->NewPositionBuff[i]);
			for (auto&e : File_Buffer->UpdateBuff[i]) assert(!e->Virtual_File_Flag), File_Buffer->free_Node[i].push_back(e);
			for (auto&e : File_Buffer->NewPositionBuff[i]) assert(!e->Virtual_File_Flag), File_Buffer->free_Node[i].push_back(e);
			File_Buffer->UpdateBuff[i].clear();
			File_Buffer->NewPositionBuff[i].clear();
			//File_Buffer->free_Node[i].splice(File_Buffer->free_Node[i].end(), File_Buffer->UpdateBuff[i]);
			//File_Buffer->free_Node[i].splice(File_Buffer->free_Node[i].end(), File_Buffer->NewPositionBuff[i]);
			File_Buffer->PositionCache[i].clear();
		}
	}
	void Restart() {
		Eva_Loop_Start();
		//File_Buffer->BufferMaxSize = BufferSize;
	}
	void addStone(int**position, us&Feature, ull*rawPosition,const MCTS_Edge&best_act, int step) {
		position[best_act.getX()][best_act.getY()] = step % 2 + 1;
		Position::addStone(Feature, best_act.move, (Colour)(step % 2));
		Position::addStone(rawPosition, best_act.move, (Colour)(step % 2));
	}
	void addStone(int**position, us&Feature, ull*rawPosition, const int&move, int step) {
		position[move%W][move / W] = step % 2 + 1;
		Position::addStone(Feature, move, (Colour)(step % 2));
		Position::addStone(rawPosition, move, (Colour)(step % 2));
	}
	void removeStone(int**position, us&Feature, ull*rawPosition, MCTS_Edge&best_act, int step) {
		position[best_act.getX()][best_act.getY()] = 0;
		Position::addStone(Feature, best_act.move, (Colour)(step % 2));
		Position::removeStone(rawPosition, best_act.move);
	}
	void removeStone(int**position, us&Feature, ull*rawPosition, const int&move, int step) {
		position[move%W][move / W] = 0;
		Position::addStone(Feature, move, (Colour)(step % 2));
		Position::removeStone(rawPosition, move);
	}
	struct Evalution {
		int ID;
		PInfo*Node;
		int**position;
		int step;

		double OutPut[W*H];
		double Value;
		Evalution() { Node = NULL; }
		Evalution(int tid, PInfo*Node,int**position,int step):Node(Node) {
			ID = tid;
			this->position = position;
			this->step = step;
			fill(OutPut, OutPut + W * H, 0.0);
			Value = -2;
		}
	};

	static const int Max_Thr = 128;
	mt19937 rng[Max_Thr + 1];

	bool Response[Max_Thr];
	Evalution Eva_Stack[Max_Thr];
	condition_variable Response_CV[Max_Thr];

	volatile bool Loop;
	volatile int Batch_Size = 0;
	int rollout_Batch;

	list<Evalution*>agent_Q[Max_Thr];
	volatile int agent_Q_Ready[Max_Thr]{ 0 };
	condition_variable agent_CV[Max_Thr];
	void Agent_Eva(int id) {
		agent_Q_Ready[id] = -1;
		mutex m; unique_lock<std::mutex>cv_lock(m);
		while (true) {
			agent_Q_Ready[id] = -1;
			agent_CV[id].wait(cv_lock, [this, id]() {return agent_Q_Ready[id] != -1; });
			if (agent_Q_Ready[id] == -2)break;
			int rotation = agent_Q_Ready[id];
			assert(!agent_Q[id].empty());
			assert(0 <= rotation && rotation < 8);
			//Input
			auto it = agent_Q[id].begin();
			for (int i = 0; i < rollout_Batch; i++) {
				if (it != agent_Q[id].end()) {
					Eva_Response(0, i, rotation, (*it)->position, (*it)->step, (*it)->OutPut, &(*it)->Value, rollout_Agent[id]);
					it++;
				}
				//fullfill
				else Eva_Response(3, i, 0, NULL, 0, NULL, NULL, rollout_Agent[id]);
			}
			//run agent
			Eva_Response(1, 0, 0, NULL, 0, NULL, NULL, rollout_Agent[id]);
			//OupPut
			it = agent_Q[id].begin();
			for (int i = 0; i < rollout_Batch; i++) {
				Eva_Response(2, i, rotation, (*it)->position, (*it)->step, (*it)->OutPut, &(*it)->Value, rollout_Agent[id]);
				it++;
				if (it == agent_Q[id].end())break;
			}
			assert(it == agent_Q[id].end());
			agent_Q[id].clear();
		}
		//wake up flag
		agent_Q_Ready[id] = -3;
		//free cuda stream
		unBindStm();
	}

	bool find_same_Node(int Thr_Num, PInfo*_Node, int j) {
		for (int i = 0; i < Thr_Num; i++) {
			if (i != j && !Response[i]) {
				if (Eva_Stack[i].Value == -4 && Eva_Stack[i].Node == _Node)return true;
			}
		}return false;
	}
	void Evaluation_Loop(int Thr_Num) {
		thread*rollout_Loop[Max_Thr];
		fill(Response, Response + Max_Thr, true);
		int Thr_cnt = 0, yield_cnt = 0;
		list<Evalution*>Q; Q.clear();
		//list<Evalution*>wait_Q; wait_Q.clear();
		//map <PInfo*, double> raw_map; raw_map.clear();
		Loop = true;
		for (int i = 0; i < rollout_Num; i++)
			rollout_Loop[i] = new thread(&MCTS_Policy::Agent_Eva, this, i);
		int sum_Node = 0, sum_Batch = 0;
		while (Loop) {
			//assert(Thr_cnt == Q.size() + wait_Q.size());
			yield_cnt = 0;
			for (int i = 0; i < Thr_Num; i++) {
				if (!Response[i]) {
					if (Eva_Stack[i].Value == -2) {
						/*if (find_same_Node(Eva_Stack[i].Node) != raw_map.end()) {
							auto value = raw_map[Eva_Stack[i].Node];
							if (value != -2) {
								auto&it = Eva_Stack[i];
								it.Value = value; it.OutPut[0] = -1;
								Response[it.ID] = true;
								Response_CV[it.ID].notify_one();
							}
							else {
								wait_Q.push_back(&Eva_Stack[i]);
								Eva_Stack[i].Value = -4;
								Thr_cnt++;
							}
						}
						else {*/
						assert((Eva_Stack[i].Node)->p.PID == -1);
						//raw_map[Eva_Stack[i].Node] = -2;
						Q.push_back(&Eva_Stack[i]);
						Eva_Stack[i].Value = -4;
						Thr_cnt++;
						//}
					}
					//yield
					else if (Eva_Stack[i].Value == -3 && find_same_Node(Thr_Num, Eva_Stack[i].Node, i)) {
						yield_cnt++;
					}
				}
				//Agent evalution
				if (!Q.empty() && Thr_cnt + yield_cnt >= Batch_Size) {
					int rotation = -1;
					for (int j = 0; j < rollout_Num; j++) {
						//assign node
						if (agent_Q_Ready[j] == -1) {
							auto it = Q.begin();
							for (int k = 0; k < Batch_Size; k++) {
								if (it == Q.end())break;
								agent_Q[j].push_back(*it);
								it++; sum_Node++;
							}
							agent_Q_Ready[j] = rotation = rng[Thr_Num]() % 8;
							assert(0 <= rotation && rotation < 8);
							assert(agent_Q_Ready[j] != -1);
							//wake up rollout,have chance fail
							agent_CV[j].notify_one();

							Q.erase(Q.begin(), it);
							sum_Batch++;
							break;
						}
					}
				}
				//return back value
				if (!Response[i] && -2 < Eva_Stack[i].Value) {
					auto&it = Eva_Stack[i];
					assert(it.ID == i);
					assert(abs(it.Value) <= 1 + 1e-8);
					//raw_map[it.Node] = it.Value;
					Response[it.ID] = true;
					Response_CV[it.ID].notify_one();
					//check not evalution
					/*if (!wait_Q.empty())
						for (auto it = wait_Q.begin(); it != wait_Q.end();) {
							assert(raw_map.find((*it)->Node) != raw_map.end());
							double value = raw_map[(*it)->Node];
							if (value != -2) {
								(*it)->Value = value; (*it)->OutPut[0] = -1;
								Response[(*it)->ID] = true;
								Response_CV[(*it)->ID].notify_one();
								it = wait_Q.erase(it);
							}
							else it++;
						}*/
				}
				Thr_cnt = Q.size();// +wait_Q.size();
				//{
				//	//Input
				//	auto it = Q.begin();
				//	for (int i = 0; i < rollout_Batch; i++) {
				//		if (it != Q.end()) {
				//			Eva_Response(0, i, rotation, (*it)->position, (*it)->step, (*it)->OutPut, &(*it)->Value, rollout_Agent);
				//			it++;
				//		}
				//		//fill
				//		else Eva_Response(3, i, 0, NULL, 0, NULL, NULL, rollout_Agent);
				//	}
				//	//run agent
				//	Eva_Response(1, 0, 0, NULL, 0, NULL, NULL, rollout_Agent);
				//	//OupPut
				//	it = Q.begin();
				//	for (int i = 0; i < rollout_Batch; i++) {
				//		Eva_Response(2, i, rotation, (*it)->position, (*it)->step, (*it)->OutPut, &(*it)->Value, rollout_Agent);
				//		//(*it)->Value = 0;
				//		raw_map[(*it)->Node] = (*it)->Value;
				//		Response[(*it)->ID] = true;
				//		Response_CV[(*it)->ID].notify_one();
				//		it = Q.erase(it);
				//		if (it == Q.end())break;
				//	}
				//	//check not evalution
				//	if (!wait_Q.empty())
				//		for (auto it = wait_Q.begin(); it != wait_Q.end();) {
				//			assert(raw_map.find((*it)->Node) != raw_map.end());
				//			double value = raw_map[(*it)->Node];
				//			if (value != -2) {
				//				(*it)->Value = value; (*it)->OutPut[0] = -1;
				//				Response[(*it)->ID] = true;
				//				Response_CV[(*it)->ID].notify_one();
				//				it = wait_Q.erase(it);
				//			}
				//			else it++;
				//		}
				//	Thr_cnt = Q.size() + wait_Q.size();
				//}
			}
			//wake up rollout
			for (int j = 0; j < rollout_Num; j++)
				if (agent_Q_Ready[j] != -1) {
					agent_CV[j].notify_one();
				}
			if (Batch_Size == 0 && sum_Node != 0) {//&& !raw_map.empty()) {
				/*for (auto&k : raw_map)assert(abs(k.Y) <= 1 + 1e-8);
				raw_map.clear();*/
				sum_Batch = sum_Node = 0;
				mutex m; std::unique_lock<std::mutex> lock(m);
				Eva_CV.wait(lock);
			}
		}
		Batch_Size = -1;
		for (int i = 0; i < rollout_Num; i++) {
			agent_Q_Ready[i] = -2; 
			while (agent_Q_Ready[i] != -3)agent_CV[i].notify_one();
			rollout_Loop[i]->join(), delete rollout_Loop[i];
		}
	}
	struct Node_Info {
		PInfo*volatile&node, *volatile _new;
		int type;
		us feature;
		int&pid;
		Node_Info(PInfo*volatile&Node,int ty,us fea,int&PID,PInfo*volatile __new):node(Node),pid(PID) {
			type = ty;
			feature = fea;
			_new = __new;
		}
	};
	vector<Node_Info>Path[Max_Thr];
	list<PInfo*volatile> pre_alloc[Max_Thr];
	double MC_Tree_Search(PInfo* volatile&Node, int&PID, us Feature, ull*rawPosition,const int step, int**position,int tid) {
		int type = -1; PInfo*_new = NULL;
		//find Cache,Create or read Node
		if (!Node) {
			auto p = LookUpTable->getPosition(*File_Buffer, PID, rawPosition, Feature, pre_alloc[tid], Node);
			if (!Node || (p.Y == 1 && p.X == Node)) {
				type = p.Y, Node = p.X;
				if (type == 5)assert(Node->Virtual_File_Flag&&Node->p.PID == PID);
				if (type < 2) {
					_new = p.X; assert(_new == pre_alloc[tid].front());
					pre_alloc[tid].pop_front();
					//read Node from file,wake up yield
					if (type == 1)
						for (int i = 0; i < Thr_Num; i++)
							if (!Response[i] && Eva_Stack[i].Value == -3)
								Response_CV[i].notify_one();
				}
			}
		}
		//leaf Node,expanded by Agent
		if (type == 0) {
			Eva_Stack[tid] = Evalution(tid, Node, position, step);
			Response[tid] = false;
			//wait for evalution
			{
				mutex m; std::unique_lock<std::mutex> lock(m);
				Response_CV[tid].wait(lock, [this, tid]() { return Response[tid]; });
			}
			EDGE*edge = &Node->p.edges;
			Path[tid].emplace_back(Node_Info(Node, type, Feature, PID, _new));
			assert(_new != NULL);
			//node already evalution
			//if (Eva_Stack[tid].OutPut[0] == -1 || 
			assert(abs(Eva_Stack[tid].Value) <= 1 + 1e-8);
			if (Node != _new) {
				return -Eva_Stack[tid].Value;
			}
			int Ord[W*H], OrdCnt = 0;
			double*Action_OutPut = Eva_Stack[tid].OutPut, Position_Value = Eva_Stack[tid].Value;
			for (int i = 0; i < W*H; i++) {
				if (position[i%W][i / W]) { assert(Action_OutPut[i] == 0); continue; }
				Ord[OrdCnt++] = i;
			}
			assert(OrdCnt > 0);
			for (int k = 0; k < OrdCnt; k++) {
				int i = Ord[k];
				Insert_Edge(Feature, rawPosition, step, edge, Action_OutPut[i], i);
				assert(Action_OutPut[i] + 1e-8 >= 0);
				if (Action_OutPut[i] > 0)assert(position[i%W][i / W] == 0);
			}
			edge->push_end();
			assert(edge == &Node->p.edges);
			assert(edge->sz > 0 && edge->sz + step == W * H);
			//wake up yield
			for (int i = 0; i < Thr_Num; i++)
				if (!Response[i] && Eva_Stack[i].Value == -3)
					Response_CV[i].notify_one();
			return -Position_Value;
		}
		//traverse UCB edges
		else {
			//wait all edges insert
			while (!Node->p.edges.Ready()) {
				Eva_Stack[tid].Value = -3, Eva_Stack[tid].Node = Node;
				Response[tid] = false;
				mutex m; std::unique_lock<std::mutex> lock(m);
				Response_CV[tid].wait_for(lock, std::chrono::milliseconds(1), [Node]() { return Node->p.edges.Ready(); });
			}
			Response[tid] = true;

			EDGE* edge = &Node->p.edges;
			assert(edge->sz > 0 && edge->sz + step == W * H);
			MCTS_Edge&best_act = edge->top(Node->totVisit, rng[tid]);

			//add stone
			addStone(position, Feature, rawPosition, best_act, step);

			double result = 0;
			//Judge game Whether End
			if (best_act.next_position_flag <= 1)result = best_act.next_position_flag;
			//check result
			else if (best_act.next_position_flag == 2 && ((result = GameJudger(position, step + 1, best_act.getMov())), (best_act.next_position_flag = result == 0 ? 3 : 0), result)) {
				//draw
				if (result == 2) result = 0, best_act.next_position_flag = 0;
				else best_act.next_position_flag = 1;
			}
			//MCTS traverse subtree
			else if (best_act.next_position_flag == 3) {
				result = MC_Tree_Search(best_act.next_Node, best_act.next_position_ID, Feature, rawPosition, step + 1, position, tid);
				//best_act.next_Node->X.X.edges.check(best_act);
			}

			//remove stone
			removeStone(position, Feature, rawPosition, best_act, step);

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
	void UpdatePath(int tid) {
		//for (int i = 0; i < Thr_Num; i++) {
		for (auto&k : _Path[tid]) {
			assert(k.node != NULL);
			assert(!k.node->p.edges.empty());
			assert(k.feature == tid);
			//new Position
			assert(k.type == 5 || k.node->Virtual_File_Flag == false);
			if (k.type == 0) {
				if (k.node->p.PID == -1) {
					k.pid = k.node->p.PID = File_Buffer->newPosition(k.feature);
					//assert(File_Buffer->PositionCache[k.feature].find(k.node->X.X.PID) == File_Buffer->PositionCache[k.feature].end());
					File_Buffer->NewPositionBuff[k.feature].push_back(k.node);
					//File_Buffer->PositionCache[k.feature][k.node->X.X.PID] = prev(File_Buffer->NewPositionBuff[k.feature].end());
				}
			}
			//read from file/virtual file
			else if (k.type == 1 || k.type == 5) {
				assert(k.node->p.PID != -1);
				if (k.type == 5) {
					if (k.node->Virtual_File_Flag) {
						k.node->Virtual_File_Flag = false;
						//assert(File_Buffer->PositionCache[k.feature].find(k.node->p.PID) != File_Buffer->PositionCache[k.feature].end());
						File_Buffer->UpdateBuff[k.feature].push_back(k.node);
						File_Buffer->PositionCache[k.feature][k.node->p.PID] = prev(File_Buffer->UpdateBuff[k.feature].end());
					}
				}
				else if (File_Buffer->PositionCache[k.feature].find(k.node->p.PID) == File_Buffer->PositionCache[k.feature].end()) {
					File_Buffer->UpdateBuff[k.feature].push_back(k.node);
					File_Buffer->PositionCache[k.feature][k.node->p.PID] = prev(File_Buffer->UpdateBuff[k.feature].end());
					//if (k.type == 5)
						//assert(k.node->Virtual_File_Flag), k.node->Virtual_File_Flag = false;
						//File_Buffer->Virtual_File[k.feature].erase(k.node->X.X.PID);
				}
				//duplicate node
				else if (k.node == k._new&&*File_Buffer->PositionCache[k.feature][k.node->p.PID] != k.node) {
					File_Buffer->free_Node[k.feature].push_back(k.node);
				}
			}
			//hit PositionCache
			else if (k.type == 2) {
				//hit in UpdateCache,move to Recent Frequent
				//if (k.node->X.X.PID < File_Buffer->IDCount[k.feature] - File_Buffer->NewPositionBuff[k.feature].size()) {
				if (LookUpTable->RF_map[k.feature].find(k.node->p.PID) == LookUpTable->RF_map[k.feature].end()) {
					auto it = File_Buffer->PositionCache[k.feature][k.node->p.PID];
					assert((*it) == k.node);
					LookUpTable->RecentFrequent[k.feature].push_back(*it);
					LookUpTable->RF_map[k.feature][k.node->p.PID] = prev(LookUpTable->RecentFrequent[k.feature].end());
					File_Buffer->PositionCache[k.feature].erase(k.node->p.PID);
					File_Buffer->UpdateBuff[k.feature].erase(it);
				}
				else assert(k.node == (*LookUpTable->RF_map[k.feature][k.node->p.PID]));
				//}
				//newPosition virtual file,copy to Update
				/*else if (k.node->X.Y) {
					k.node->X.Y = false;
					File_Buffer->UpdateBuff[k.feature].push_back(k.node);
					File_Buffer->PositionCache[k.feature][k.node->X.X.PID] = prev(File_Buffer->UpdateBuff[k.feature].end());
				}*/
			}
			//find RecentFrequent
			else if (k.type == 3) {
				auto&it = LookUpTable->RF_map[k.feature][k.node->p.PID];
				assert(k.node == (*it));
				//move to list back
				if (it != prev(LookUpTable->RecentFrequent[k.feature].end())) {
					LookUpTable->RecentFrequent[k.feature].push_back(*it);
					LookUpTable->RecentFrequent[k.feature].erase(it);
					it = prev(LookUpTable->RecentFrequent[k.feature].end());
				}
			}//most visit
			else if (k.type == 4) {
				auto&ls = LookUpTable->MostViste[k.feature];
				if (ls.back() != k.node) {
					bool flag = false;
					for (auto it = ls.begin(); it != ls.end(); it++) {
						if (*it == k.node) {
							ls.push_back(*it); ls.erase(it);
							flag = true;
							break;
						}
					}assert(flag);
				}
			}
			else assert(false);
		}
		//}
		//reset and clear thread path
		for (auto&k : _Path[tid])
			//if (!(*k.node)->Virtual_File_Flag)
			k.node = NULL;
		_Path[tid].clear();
		//RF,MV to Update
		LookUpTable->Recycle(*File_Buffer, tid);
	}
	//add dirichlet noise to P(s,a)
	const double dir_alpha = 0.75;
	double dir_noise[W*H][W*H];
	void addDirNoise(PInfo*Node,double*dir_distribution) {
		for (int i = 0; i < W*H;i++) {
			auto&e = (*Node->p.edges.edge)[i];
			if (e.Prior_Probability == -1)break;
			e.Prior_Probability = dir_alpha * e.Prior_Probability + (1 - dir_alpha)*dir_distribution[i];
		}
	}
	void removeDirNoise(PInfo*Node, double*dir_distribution) {
		for (int i = 0; i < W*H; i++) {
			auto&e = (*Node->p.edges.edge)[i];
			if (e.Prior_Probability == -1)break;
			e.Prior_Probability = (e.Prior_Probability - (1 - dir_alpha)*dir_distribution[i]) / dir_alpha;
		}
	}
	void generate_noise(PInfo*NoiseNode, int step) {
		//~10 random moves over all average legal moves 
		const double times = 10.0 / (W*H / 4 * 3);
		std::gamma_distribution<double> dist(times, 1);
		double sum = 0;
		for (int i = 0; i < W*H; i++)sum += dir_noise[step][i] = dist(rng[0]), assert(dir_noise[step][i] > 0);
		for (int i = 0; i < W*H; i++)dir_noise[step][i] /= sum;
		addDirNoise(NoiseNode, dir_noise[step]);
	}

	void Sampling_Visit(PInfo*Node, MCTS_Edge*&best_act, int step, ull&tot, double*Max_Q = NULL, int Sampling_step = 0) {
		for (auto&e : *Node->p.edges.edge) {
			if (e.Prior_Probability == -1)break;
			if (e.next_position_flag == 2)continue;
			tot += e.Visit_Count, assert(e.Visit_Count >= 0);
			if (Max_Q)*Max_Q = max(*Max_Q, e.Q_Value);
		}assert(tot > 0);
		//sampling available moves
		if (step < Sampling_step) {
			ull chi = rng[0]() % tot, cnt = 0;
			for (auto&e : *Node->p.edges.edge) {
				if (e.Prior_Probability == -1) { assert(false); break; }
				if (e.next_position_flag == 2)continue;
				cnt += e.Visit_Count;
				if (cnt > chi) {
					best_act = &e;
					break;
				}
			}
		}
		//max count
		else {
			vector<MCTS_Edge*>Q; Q.clear();
			ull Visit_Count = 0;
			for (auto&e : *Node->p.edges.edge) {
				if (e.Prior_Probability == -1)break;
				if (e.next_position_flag == 2)continue;
				if (Visit_Count < e.Visit_Count)Visit_Count = e.Visit_Count, Q.clear(), Q.push_back(&e);
				else if (Visit_Count == e.Visit_Count)Q.push_back(&e);
			}
			int idx = rng[0]() % Q.size();
			best_act = Q[idx];
			assert(best_act->Visit_Count > 0);
		}
	}
	mutex lock;
	int Finish_Thr;
	void Thread_Start(int tid, PInfo*volatile&Node, int&PID, us Feature, ull(*rawPosition)[8], int step, int***position) {
		for (int i = 0; i < SimulationPerMCTS / Thr_Num; i++) {
			MC_Tree_Search(Node, PID, Feature, rawPosition[tid], step, position[tid], tid);
		}
		lock_guard<mutex>locker(lock);
		Finish_Thr++;
		Batch_Size = min(Thr_Num - Finish_Thr, Batch_Size);
	}
	thread*Thr[Max_Thr];
	//generate RL data
	void Generate_Move(PInfo*volatile&Node,int&PID, us Feature, ull (*rawPosition)[8], int step, int***position) {
		Batch_Size = rollout_Batch, Finish_Thr = 0;
		Eva_CV.notify_one();
		//X Simulation from root
		for (int t = 0; t < Thr_Num; t++)
			Thr[t] = new thread(&MCTS_Policy::Thread_Start, this, t, std::ref(Node), std::ref(PID), Feature, rawPosition, step, position);
		for (int i = 0; i < Thr_Num; i++) {
			Thr[i]->join();
			delete Thr[i];
		}
		PInfo*Root = Node, *NoiseNode = NULL;
		//Agent sampling move
		MCTS_Edge*best_act = NULL; ull tot = 0; double Max_Q = -1e9;
		Sampling_Visit(Root, best_act, step, tot, &Max_Q, 6);
		assert(Max_Q != -1e9);
		//add extra P(s,a) noise
		if (NoiseNode = best_act->next_Node)
			generate_noise(NoiseNode, step + 1);

		//enable resign,90% games 
		if (Enable_resign) {
			if (Max_Q < V_resign) {
				(GameResult = step % 2 ? 1 : -1), GameStep = step;
				return;
			}
		}
		else Min_V_resign[step % 2] = min(Min_V_resign[step % 2], Max_Q);

		//add Train Data
		trainData[step].move = best_act->move;
		memset(trainData[step].OutPut, 0, sizeof(trainData[step].OutPut));
		double sum = 0;
		for (auto&e : *Root->p.edges.edge) {
			if (e.Prior_Probability == -1)break;
			if (e.next_position_flag == 2)continue;
			sum += trainData[step].OutPut[e.move] = 1.0*e.Visit_Count / tot;
		}assert(abs(sum - 1) <= 1e-8);
		//add stone
		for (int i = 0; i < Thr_Num; i++) {
			us tmp = 0; addStone(position[i], i == 0 ? Feature : tmp, rawPosition[i], *best_act, step);
		}
		if (best_act->next_position_flag<=1)(GameResult = step % 2 ? -best_act->next_position_flag : best_act->next_position_flag), GameStep = step + 1;
		//MCTS traverse subtree
		else {
			assert(best_act->next_position_flag == 3);
			assert(best_act->next_Node != NULL || best_act->next_position_ID != -1);
			Generate_Move(best_act->next_Node, best_act->next_position_ID, Feature, rawPosition, step + 1, position);
		}
		//remove noise
		if (NoiseNode)
			removeDirNoise(NoiseNode, dir_noise[step + 1]);
	}
	thread*Eva_Loop = NULL;
	condition_variable Eva_CV;
	PInfo* getNode(int ID,us Feature) {
		if (!LookUpTable->MostViste[Feature].empty()) {
			for (auto&e : LookUpTable->MostViste[Feature]) {
				if (e->p.PID == ID) {
					return e;
				}
			}
		}
		if (LookUpTable->RF_map[Feature].find(ID) != LookUpTable->RF_map[Feature].end())
			return *LookUpTable->RF_map[Feature][ID];
		if (File_Buffer->PositionCache[Feature].find(ID) != File_Buffer->PositionCache[Feature].end())
			return *File_Buffer->PositionCache[Feature][ID];
		if (ID < File_Buffer->Virtual_File_New[Feature].size())
			return assert(File_Buffer->Virtual_File_New[Feature][ID]->p.PID == ID), File_Buffer->Virtual_File_New[Feature][ID];
		return NULL;
	}
	void MCTS_Start() {
		Init_State();
		//root noise
		PInfo*root = getNode(Ins_PID, Ins_Feature);
		assert(Step == 0 || root);
		if (root)
			assert(root->p.PID == Ins_PID), generate_noise(root, Step);
		Generate_Move(Node[0],Ins_PID, Ins_Feature, Ins_rawPos, Step, Ins_Position);
		//remove root noise
		if (root)
			removeDirNoise(root, dir_noise[Step]);
		Update_and_Write();
		//update resign value
		if (!Enable_resign) {
			int _sz = resign_Sort_Set.size();
			if (GameResult == 1)resign_Sort_Set.insert(Min_V_resign[0]), recent_resign.push_back(Min_V_resign[0]);
			else if (GameResult == -1)resign_Sort_Set.insert(Min_V_resign[1]), recent_resign.push_back(Min_V_resign[1]);
			if (_sz != resign_Sort_Set.size() && resign_Sort_Set.size() >= resign_MinSize) {
				//pop
				if (recent_resign.size() > resign_MaxSize) {
					assert(resign_Sort_Set.find(recent_resign.front()) != resign_Sort_Set.end());
					resign_Sort_Set.erase(resign_Sort_Set.find(recent_resign.front())), recent_resign.pop_front();
				}
				const double resign_rate = 0.03;//0.049;
				int sz = resign_Sort_Set.size()*resign_rate;
				for (auto&k : resign_Sort_Set)if (sz == 0) {
					V_resign = k; break;
				}
				else sz--;
			}
		}
	}
	void Eva_Loop_Start() {
		if (Eva_Loop == NULL)
			Eva_Loop = new thread(&MCTS_Policy::Evaluation_Loop, this, Thr_Num);
	}
	void Eva_Loop_end(){
		Loop = false; while (Batch_Size != -1)Eva_CV.notify_one();
		if (Eva_Loop)Eva_Loop->join(); delete Eva_Loop; Eva_Loop = NULL;
	}


	//action API
	int Ins_PID, Step;
	us Ins_Feature;
	ull Ins_rawPos[Max_Thr][8];
	int**Ins_Position[Max_Thr];
	PInfo*volatile Node[W*H];
	int Thr_Num;
	int SimulationPerMCTS;

	void addStone(const MCTS_Edge&best_act) {
		for (int i = 0; i < Thr_Num; i++) {
			us tmp = 0; addStone(Ins_Position[i], i == 0 ? Ins_Feature : tmp, Ins_rawPos[i], best_act, Step);
		}
		Step++;
		Ins_PID = best_act.next_position_ID;
		Node[Step] = best_act.next_Node;
	}
	/*void removeStone(const int&move) {
		Step--;
		for (int i = 0; i < Thr_Num; i++) {
			us tmp = 0; removeStone(Ins_Position[i], i == 0 ? Ins_Feature : tmp, Ins_rawPos[i], move, Step);
		}
		Ins_PID = -1;
		assert(Node[Step]);
	}*/
	void Init_State(int pre_alloc_factor = 150 * 2) {
		Ins_PID = File_Buffer->IDCount[0] == 0 ? -1 : 0;
		Step = 0; Ins_Feature = 0;
		memset((void*)(void**)Node, 0, sizeof(Node));
		for (int i = 1; i < Maximum_Cache; i++)
			File_Buffer->free_Node[0].splice(File_Buffer->free_Node[0].end(), File_Buffer->free_Node[i]);
		for (int j = 0; j < Thr_Num; j++) {
			Path[j].clear();
			int cnt = SimulationPerMCTS / Thr_Num * pre_alloc_factor - pre_alloc[j].size();
			auto it = File_Buffer->free_Node[0].begin(); advance(it, min(cnt, File_Buffer->free_Node[0].size()));
			pre_alloc[j].splice(pre_alloc[j].end(), File_Buffer->free_Node[0], File_Buffer->free_Node[0].begin(), it);
			cnt = SimulationPerMCTS / Thr_Num * pre_alloc_factor - pre_alloc[j].size();
			for (int k = 0; k < cnt; k++) {
				auto p = new PInfo();
				p->p = -1;
				pre_alloc[j].emplace_back(p);
			}
			memset(Ins_rawPos[j], 0, sizeof(Ins_rawPos[j]));
			for (int i = 0; i < W; i++)
				memset(Ins_Position[j][i], 0, H * sizeof(int));
		}
		
		rollout_Batch = rollout_Agent[0]->Net_Param["Batch"];
		Min_V_resign[0] = Min_V_resign[1] = 1e9;
		Enable_resign = recent_resign.size() < resign_MinSize ? false : ((rng[0]() % 10) < 7);
	}
	void Select_Move(int&SelectMove, int&result) {
		Batch_Size = rollout_Batch, Finish_Thr = 0;
		Eva_CV.notify_one();
		for (int t = 0; t < Thr_Num; t++)
			Thr[t] = new thread(&MCTS_Policy::Thread_Start, this, t, std::ref(Node[Step]), std::ref(Ins_PID), Ins_Feature, Ins_rawPos, Step, Ins_Position);
		for (int i = 0; i < Thr_Num; i++) {
			Thr[i]->join();delete Thr[i];
		}
		PInfo*Root = Node[Step];

		//select Maximum visit count
		MCTS_Edge*best_act = NULL; ull tot = 0;
		Sampling_Visit(Root, best_act, Step, tot, NULL, 0);

		addStone(*best_act);
		
		SelectMove = best_act->move;
		if (best_act->next_position_flag <= 1)result = best_act->next_position_flag;
		else assert(best_act->next_position_flag == 3), assert(Node[Step] != NULL || Ins_PID != -1);
	}
	void Opponent_Move(const int&opponent_move) {
		//assert(Node == NULL);
		if (Step > 0 || File_Buffer->IDCount[0] > 0)assert(Node[Step] != NULL || Ins_PID != -1);
		Batch_Size = 1;
		Eva_CV.notify_one();
		if (Node[Step] == NULL)
			MC_Tree_Search(Node[Step], Ins_PID, Ins_Feature, Ins_rawPos[0], Step, Ins_Position[0], 0);
		//oppponent move
		MCTS_Edge*best_act = NULL;
		for (auto&k : *Node[Step]->p.edges.edge) {
			if (k.Prior_Probability == -1)break;
			if (k.move == opponent_move) {
				best_act = &k;
				break;
			}
		}
		addStone(*best_act);
		if (best_act->next_position_flag == 3)assert(Node[Step] != NULL || Ins_PID != -1);// , assert(best_act->Visit_Count > 0);
		//expand new Node
		else {
			assert(best_act->next_position_flag == 2 && best_act->next_position_ID == -1 && best_act->next_Node == NULL);
			MC_Tree_Search(best_act->next_Node, best_act->next_position_ID, Ins_Feature, Ins_rawPos[0], Step, Ins_Position[0], 0);
			best_act->next_position_flag = 3;
			assert(best_act->next_Node);
			Node[Step] = best_act->next_Node;
		}
	}
	void TakeBack(int _step, pi*moves) {
		Step = _step; Ins_Feature = 0; Ins_PID = -1;
		for (int j = 0; j < Thr_Num; j++) {
			for (int i = 0; i < W; i++)
				memset(Ins_Position[j][i], 0, H * sizeof(int));
		}
		for (int j = 0; j < _step; j++) {
			for (int i = 0; i < Thr_Num; i++) {
				us tmp = 0; addStone(Ins_Position[i], i == 0 ? Ins_Feature : tmp, Ins_rawPos[i], moves[j].Y*W + moves[j].X, j);
			}
		}
		DEBUG(Node[Step] != NULL);
		//removeStone(move);
	}
};


MCTS_Policy*MCTS_Agent = NULL;
void MCTS_Agent_Init(Agent**agents, int agent_Num, int Thr_Num, int Sims, const char*dir_path, const char*name) {
	pipeOut("DEBUG MCTS Clear\n");
	std::experimental::filesystem::remove_all(((string)dir_path + "MCTS\\" + name + "\\").c_str());
	pipeOut("DEBUG MCTS Init\n");
	Init_MCTS(dir_path);
	if (!MCTS_Agent)
		MCTS_Agent = new MCTS_Policy(agents, agent_Num, Thr_Num, Sims, name, ((string)dir_path + "MCTS\\").c_str());
}
pi move_track[W*H];
pi MCTS_Agent_Response(int**board, pi*moves, int moves_cnt) {
	pipeOut("DEBUG response\n");
	//detect takeback
	for (int i = 0; i < MCTS_Agent->Step; i++) {
		//change moves
		if (i >= moves_cnt || moves[i].X != move_track[i].X || moves[i].Y != move_track[i].Y) {
			pipeOut("DEBUG take back\n");
			MCTS_Agent->TakeBack(i, moves);
			break;
		}
	}
	DEBUG(moves_cnt >= MCTS_Agent->Step);
	while (moves_cnt != MCTS_Agent->Step) {
		pi&mov = moves[MCTS_Agent->Step];
		move_track[MCTS_Agent->Step] = mov;
		MCTS_Agent->Opponent_Move(mov.Y*W + mov.X);
	}

	int move = -1, result = 2;
	MCTS_Agent->Select_Move(move, result);
	move_track[MCTS_Agent->Step - 1] = { move%W,move / W };
	return { move%W,move / W };
}
void MCTS_Agent_New_Game() {
	pipeOut("DEBUG MCTS New Game\n");
	//wait for modification
	MCTS_Agent->Update_and_Write();
	MCTS_Agent->Init_State(100);
	pipeOut("DEBUG MCTS End Write\n");
}
void MCTS_Agent_End() {
	pipeOut("DEBUG MCTS End\n");
	MCTS_Agent->Update_and_Write();
	//MCTS_Agent->Write2File();
	MCTS_Agent->remove_dir();
	delete MCTS_Agent;
}

int tot_data_set_Count = 0;
void MCTS_Param(HyperParamSearcher&param) {
	param["paramNum"] = tot_data_set_Count;
	param["trainNum"] = param["paramNum"] * 1.0;
	param["ExtraTrainFactorPerEpoch"] = 1;
	param["ExtraTestFactorPerEpoch"] = 1;
	param["Stochastic"] = 1;
	param["EpochUpdate"] = 0;
	param["Max_Epoch"] = 1;
}
const int Console_delay = 100;
char OutPut[H + 1][W + 3];
void DisplayBoard(int(*Scr)[H]) {
	system("cls");
	::memset(::OutPut[0], 0, sizeof(::OutPut[0]));
	for (int i = -1; i < H; i++)
		sprintf(::OutPut[i + 1], "%02d", i + 1);
	for (int i = 0; i < W; i++)
		::OutPut[0][i + 2] = (char)('a' + i);
	for (int i = 0; i < H; i++)
		for (int j = 0; j < W; j++) {
			char c = ' ';
			switch (Scr[j][i]) {
				//Black
			case 1:c = '1';
				break;
				//White
			case 2:c = '2';
				break;
			default:
				break;
			}
			::OutPut[i + 1][j + 2] = c;
		}
	for (int i = 0; i < H + 1; i++)
		std::printf("%s\n", ::OutPut[i]);
	//Sleep(Console_delay);
}
int Turn(int(*Screen)[H],int move,int result,int step) {
	Screen[move%W][move / W] = step + 1;
	DisplayBoard(Screen);
	if (result != 2) {
		//win
		if (result == 1)return step % 2 ? -1 : 1;
		//draw
		else return 0;
	}
	return 2;
}
double MCTS_Match(MCTS_Policy*P1, MCTS_Policy*P2) {
	int Screen[W][H] = { 0 }, ret;
	int move = -1, result = 2;

	P1->Init_State();
	P2->Init_State();
	while (true) {
		P1->Select_Move(move, result);
		if (ret = Turn(Screen, move, result, 0), ret != 2)break;

		P2->Opponent_Move(move);
		P2->Select_Move(move, result);
		if (ret = Turn(Screen, move, result, 1), ret != 2)break;

		P1->Opponent_Move(move);
	}
	P1->Update_and_Write();
	P2->Update_and_Write();
	return ret;
}
mutex ds_lock, rollout_lock;
const int Max_Data_Size = 8 * 8 * 15 * 15 * 80 * 2;
bool Write_New_Data = false;
void MCTS_generate_data(MCTS_Policy*ag, DataSet&ds, condition_variable&train_agent,const char*data_Path,int GamesPerAgent) {
	ll tot_step = 0, game_count = 0;
	clock_t start_time = clock();
	while (true) {
		{
			lock_guard<mutex>rollout_locker(rollout_lock);
			//execute MCTS,generate train data
			ag->MCTS_Start();
		}
		{
			lock_guard<mutex>ds_locker(ds_lock);
			//train RL agent
			for (int i = 0; i < GameStep; i++)
				ds.trainSet_Add_data(Gomoku(ds.trainSet_gameCount(), i, trainData[i].OutPut, W*H));
			//add into trainSet
			for (int j = 0; j < GameStep; j++) {
				Gomoku::SetMove(ds.trainSet_Param(ds.trainSet_gameCount()), { trainData[j].move%W,trainData[j].move / W }, W*H);
			}
			//Reward,Win:1 Loss:-1 Draw:0
			ds.trainSet_Add_Reward(GameResult);

			tot_step += GameStep, game_count++;
			printf("step:%d result:%d average_step:%.02lf tot_step:%d Count:%d time:%d min(s)\n", GameStep, GameResult, 1.0*tot_step / game_count, tot_step, ds.trainSet_gameCount(), (clock() - start_time) / CLOCKS_PER_SEC / 60);
			//Batch
			if (ds.trainSet_gameCount() % GamesPerAgent == 0 || (_kbhit() && _getch() == 'l')) {
				ag->Write2File();
				//Save data
				ds.trainSet_Save_Load(true, -1, data_Path);

				//::ShowWindow(::GetConsoleWindow(), SW_HIDE);
				break;
			}
		}
		/*if (ds.trainSet_gameCount() % 64 == 1) {
			train_agent.notify_one();
			this_thread::sleep_for(std::chrono::milliseconds(4000));
			lock_guard<mutex>ds_locker(ds_lock);
			if (Write_New_Data)
				ds.trainSet_Save_Load(false, Max_Data_Size, "trainSet_data_1"), Write_New_Data = false;
		}*/
	}
}
void Sampling_DataSet(DataSet&train_ds,int data_Count,int recent_Max_data,int dataSet_Max_ID) {
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
			char Path[100]; sprintf(Path, "trainSet_data_%d", dataSet_Max_ID);
			ds.trainSet_Save_Load(false, Max_Data_Size, Path);
			Start += ds.dataCount;
			dataSet_Max_ID--;
		}
		int game_id = ds.trainSet_Param(Start).In(0)[0] / 1000;
		auto&data = ds.trainSet_Param(Start);
		train_ds.trainSet_Add_data(Gomoku(train_ds.gameCount, (int)data.In(0)[0] % 1000, &data.In(0)[1], W*H));
		train_ds.trainSet_Param(train_ds.gameCount).Out(0) = ds.trainSet_Param(game_id).Out(0);
		train_ds.trainSet_Add_Reward(ds.Reward[game_id]);
	}
	tot_data_set_Count = train_ds.trainSet_dataCount();
	//train_ds.trainSet_Save_Load(true, -1, "trainSet_data_sampling");
}
double TestScore(double Cor, vector<double>&part_Loss, double TestNum) {
	return Cor / 0.5*0.001 + 0.25 - (part_Loss[0] * 2.0 / TestNum / 4.0);
}
void RL_SelfPlay_with_MCTS(Judger GameJudger, Agent*agent, int rollout_Num, Agent**rollout_agent, Agent**rollout_agent0, Agent**rollout_agent1, AgentResponse ResponseFunc, DataSet*player_ds, int agent_id,int Maximum_DataSet_Number) {

	Init_MCTS("", ResponseFunc, GameJudger);

	//auto Player0 = new MCTS_Policy(rollout_agent, rollout_Num, 25, 500, "1");
	//auto Player1 = new MCTS_Policy(rollout_agent1, rollout_Num, 25, 500, "2");
	//int p0 = 0, p1 = 0;
	//for (int t = 0; t < 100; t++) {
	//	double res = MCTS_Match(Player0, Player1);
	//	if ((res == 1 && t % 2 == 0) || (res == -1 && t % 2 == 1))printf("P0 win\n"), p0++;
	//	else if (res != 0)printf("P1 win\n"), p1++;
	//	else printf("Draw\n");
	//	printf("margin of %.02lf%%\n", 100.0*(p1 - p0) / p1);

	//	//swap hand
	//	swap(Player0, Player1);
	//}
	////delete dir
	//Player0->remove_dir();
	//Player1->remove_dir();
	//delete Player0;
	//delete Player1;
	//getchar();
	//return;
	//Player data test
	//DataSet _player_ds;
	//_player_ds.trainSet_Save_Load(false, 2673103, "Player_DataSet");
	////test player data
	//if (&_player_ds)
	//	cout << _player_ds.Test(agent, _player_ds.trainSet_dataCount()/5, Gomoku::MCTS_GomokuSimulation, TestScore) << endl;

	int GamesPerAgent = 5000, Agents_Count = agent_id, GamesPerAgent_Factor = 8, extra_games_count = GamesPerAgent * GamesPerAgent_Factor;
	//while (true) 
	{
		//DataSet ds;
		//mutex m; std::unique_lock<std::mutex> lock(m); condition_variable train;
		//char path[100]; sprintf(path, "trainSet_data_%d", Maximum_DataSet_Number);
		//ds.trainSet_Save_Load(false, Max_Data_Size, path);
		////thread*MCTS = new thread(&MCTS_generate_data, Main_ag, ref(ds), ref(train));
		//do {
		//	MCTS_Policy* Main_ag = new MCTS_Policy(rollout_agent, rollout_Num, 25, 500, "0", "D:\\MCTS_Policy\\");
		//	MCTS_generate_data(Main_ag, ds, train, path, GamesPerAgent);
		//	//delete cache
		//	if (ds.trainSet_gameCount() % 40000 == 0)
		//		Main_ag->remove_dir();
		//	delete Main_ag;
		//	extra_games_count -= GamesPerAgent;
		//	if (_kbhit() && _getch() == 'l')
		//		break;
		//} while (extra_games_count > 0);
		//ds.Disponse();
		//Agents_Count++;
		
		int train_cnt = 0; double Test_Table[1000]{ 0 };
		fill(Test_Table, Test_Table + 1000, -1e9);
		double best_score = 0; int best_idx = -1;
		double ori_Speed = agent->Net_Param["Speed"];
		while (true) {
			//MCTS_generate_data(Main_ag, ds, train);
			//train.wait(lock);
			{
				//while (Write_New_Data)this_thread::sleep_for(std::chrono::milliseconds(100));
				//int epochs = 1, win = 0;
				//while (epochs--) {
				//rollout_agent1[0]->Data_Assignment(agent);
				DataSet train_ds;
				//agent->Net_Param["Speed"] = max(ori_Speed * pow(0.1, floor(train_cnt / agent->Net_Param["Speed_decay_time"])), 2e-5);
				//train_ds.trainSet_Save_Load(false, Max_Data_Size, "trainSet_data_sampling");
				int cnt = 8;
				while (cnt-- > 0) {
					Sampling_DataSet(train_ds, 400000, 36 * 5000 * 21, Maximum_DataSet_Number);
					train_ds.miniTrain_Start(agent, NULL, MCTS_Param, Gomoku::MCTS_GomokuSimulation);
					agent->Write_to_File("SL_Net");
					//::ShowWindow(::GetConsoleWindow(), SW_HIDE);
				}
			}

			//1.train data loss,test data loss
			//2.player data test
			//3.match
			//Compare new Agent
			//int win = 0, sum_p0 = 0, sum_p1 = 0, matches_count = 4;
			//for (int g = 0; g < matches_count; g++) {
			//	auto Player0 = new MCTS_Policy((g % 2 == 0) ? rollout_agent0 : rollout_agent1, rollout_Num, 25, 500, "1", "D:\\MCTS_Policy\\");
			//	auto Player1 = new MCTS_Policy((g % 2 == 0) ? rollout_agent1 : rollout_agent0, rollout_Num, 25, 500, "2", "D:\\MCTS_Policy\\");
			//	int p0 = 0, p1 = 0, matches = 100;
			//	for (int t = 0; t < matches; t++) {
			//		double res = MCTS_Match(Player0, Player1);
			//		if ((res == 1 && t % 2 == 0) || (res == -1 && t % 2 == 1))printf("P0 win\n"), p0++;
			//		else if (res != 0)printf("P1 win\n"), p1++;
			//		else printf("Draw\n");
			//		printf("margin of %.02lf%%\n", 100.0*(p1 - p0) / p1);
			//		//early stop
			//		if (g % 2 == 0 && t > 20 && 100.0*(p1 - p0) / p1 < -100)break;
			//		if (g % 2 == 1 && t > 20 && 100.0*(p0 - p1) / p0 < -100)break;
			//		if (g % 2 == 0 && p0 >= matches / (1 + 1 / (1 - 0.55)))break;
			//		if (g % 2 == 1 && p1 >= matches / (1 + 1 / (1 - 0.55)))break;
			//		//swap hand
			//		swap(Player0, Player1);
			//	}
			//	//delete dir
			//	Player0->remove_dir();
			//	Player1->remove_dir();
			//	delete Player0;
			//	delete Player1;
			//	
			//	if (g % 2 == 1)swap(p1, p0);
			//	sum_p0 += p0, sum_p1 += p1;
			//	if (p1 > 1 / (1 - 0.55)*p0)win++;
			//	else break;
			//}
			//a margin of >55%
			//Update MCTS rollout agent

			char path[50];sprintf(path, "SL_Net_%d", train_cnt);
			agent->Write_to_File(path);

			if (train_cnt == 10) {
				//lock_guard<mutex>rollout_locker(rollout_lock);
				//printf("Update Agent\n");

				//for (int i = 0; i < rollout_Num; i++) {
				//	agent->Data_Assignment(rollout_agent[i]);
				//	//agent->Data_Assignment(rollout_agent0[i]);
				//}
				//char path[50]; sprintf(path, "Agent_#%d", Agents_Count);
				//agent->Write_to_File(path);
				//fstream file; file.open("lastest_Agent_id", ios::out);
				//if (file.is_open()) {
				//	char buf[50]; int sz = sprintf(buf, "%d %d", Agents_Count, Maximum_DataSet_Number);
				//	file.write(buf, sz);
				//}file.close();
				//extra_games_count = GamesPerAgent * GamesPerAgent_Factor;

				break;
			}
			

			//control train time
			train_cnt++;
			//double res = 1.0*(sum_p1 - sum_p0) / sum_p1;
			//Test_Table[train_cnt] = max(Test_Table[train_cnt - 1], res);
			////early Stop
			////fail to train better agent
			////try generate more data
			//if (train_cnt > 8&&res <= Test_Table[train_cnt / 3] && true) {
			//	extra_games_count = GamesPerAgent;
			//	rollout_agent[0]->Data_Assignment(agent);
			//	break;
			//}
		}
	}
}


