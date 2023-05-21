#pragma once
#ifdef MAHJONG_EXPORTS
#define __RiiChi__ __declspec(dllexport)
#else
#define __RiiChi__ __declspec(dllimport)

#endif

#define _CRT_SECURE_NO_WARNINGS

#include<iostream>
#include<assert.h>
#include<algorithm>
#include<vector>
#include<map>
#include<fstream>

#include<Game.h>
#include"RewardPredictor.h"
#include"AI_Agent.h"

#include<random>


#include <fcntl.h>
#include <io.h>
#include <strsafe.h>

#include<atlconv.h>

using namespace std;
#define ui unsigned int
#define X first
#define Y second
using std::vector; 
using std::map;

//class RiiChi :public Environment {
//
//	int step;
//public:
//	static const int MaximumTilesNum = (3 * 9 + 7) * 4;
//	static const int TileMaximumID = 3 * 9 + 7;
//	static const int InitialTilesCount = 13;
//	static const int ActionSpace = 2 * TileMaximumID + 5 + 3 + 1;
//
//	static const int PlayerNum = 4;
//	int InsPlayerID;
//
//	vector<int> DesktopTilesDisplay[PlayerNum];
//	//set of three or four tile
//	map<int, pair<int, int>>tileSet[PlayerNum];
//	//int SelectNotChoice[PlayerNum];
//	//vector<int>bannedTiles[PlayerNum];
//	//have win but continue game
//	int GameWinds;
//	bool riichi[PlayerNum];
//	bool closedhand[PlayerNum];
//	map<int,bool>riichi_after_drop[PlayerNum];
//	int listenTile, lastGangTile;
//	//tile,score
//	//map<int, int>WinningTile[PlayerNum];
//
//	int Score[PlayerNum];
//	//vector<pair<int, int>>GangScore[PlayerNum];
//	//tiles pass flag
//	//vector<int>TurnBannedPengTile[PlayerNum];
//	//vector<int>TurnBannedHuTile[PlayerNum];
//	// 
//	//Agent paramters
//	static const int W = TileMaximumID;
//	static const int H = 1;
//	static const int filters = 32;
//	static const int H_filters = 32;
//	int block_Num = 8;
//	static const int Action_Depth = 7;
//
//	static const int MaximumGameLength = 256;
//	static const int History_Maximum_Num = 8;
//	static const int Screen_Size = PlayerNum * (27 + 8) + 27 * 4 + 6 + 4;
//	static const int Image_Depth = 4 * 8 + 4 * 4 + 2 + 4 + 4 + 3 * 4 + 4;
//
//	struct History {
//		struct Player {
//			int handtileCount;
//			map<int, pair<int, int>>tileSet;
//			vector<int>droptile;
//			//vector<int>listeningTile;
//			int score;
//			bool listened;
//		};
//		Player player[PlayerNum];
//	}history[MaximumGameLength];
//
//
//	//Tiles library
//	vector<int>TilesLib;
//	vector<int>DeadLib;
//	ui last_move;
//	int InsCheckPlayerID;
//	int lastPlayerID;
//
//	bool HasDrawed;
//	int GameOver;
//
//	vector<int>HandTiles[PlayerNum];
//public:
//	RiiChi() :Environment(10, getHiddenStateSize()) {}
//
//	void Reset(mt19937* rng);
//	int getType(const int tile) const;
//	int getNumber(const int tile) const;
//	int GetInsActionMask(bool* ActIsValid, int CheckPlayerID, bool AddHistory);
//	int TilesCount[3][9];
//	bool checkThree();
//	bool checkYaku(int PlayerID);
//	bool WinCheck(const vector<int>& hand,int PlayerID);
//	bool checkYaoJiu();
//	bool yaojiuCheck(const int* tiles);
//	void DropLastTile(int ID);
//	bool deleteTile(int ID, int* three_tile);
//	void insertTile(int ID, vector<int>& three_tile);
//
//	void GetExtraPreActionMask(bool* ActIsValid, int CheckPlayerID, int Type) {}
//	void PreAct(ui Action, int Type, int shift) {}
//	int getInsCheckPlayerID();
//
//	void calScore(int score, int targetID = -1);
//	int calTilePatternScore();
//	int DrawTile();
//	void DropTile(int TileID);
//	void Chi(int TileID, int Type);
//	void ThreeSet(int TileID);
//	void FourSet(int TileID, int targetCount);
//	void Act(ui Action, double* Reward);
//	int OpponentsAct(int* Others_Action);
//	inline int GetNextPlayerID(int action, int InsPlayerID);
//	bool GetGameState(int* result_scores);
//	static const int ScrH = 13 + 6 + 2;
//	static const int ScrW = 30;
//	static const int blank_size = 4;
//	int Scr[ScrH][ScrW];
//	void DrawScreen();
//	char* getPieceAtPosition(int y, int x);
//	void PrintScr();
//
//
//
//	void GetGameScreen(double* Scr);
//	const double getRewardDiscount() { return 1.0; }
//	const ui getMaxUnrolledStep() { return 6; }
//	const ui getRealGameActionSpace() { return ActionSpace; }
//	const ui getSimplifyActionSpace() { return ActionSpace; }
//	const ui getScreen_Size() { return Screen_Size; }
//	const ui getHiddenStateSize() { return W * H * H_filters; }
//	const ui getAct(ui move) {
//		return move;
//	}
//	const ui getSimAct(ui move) {
//		return move;
//	}
//
//	//double Data_Scr[Screen_Size]{ 0 };
//	void DirectRepresent(double* Scr) {
//		//memcpy(Data_Scr, Scr, sizeof(double)*Screen_Size);
//	}
//	void DynamicEncode(const list<Net*>& InPut_Nets, Param* Data, int idx, Mat* Hidden_State, const ui Action, int fillZero);
//	void RepresentEncode(Param* Data, int fillZero);
//
//	void RepresentDecode(Agent* agent);
//	void DynamicDecode(Agent* agent);
//
//	void Get_NextState_Reward_And_Policy_Value(const list<Net*>& OutPut_Nets, Param* Data, int idx, Mat* Next_State, double* Reward, double* Policy, double* Value);
//	void Get_Initial_State_And_Policy_Value(const list<Net*>& OutPut_Nets, Param* Data, int idx, Mat* Next_State, double* Policy, double* Value);
//
//	Net* RepresentationNet(HyperParamSearcher& param);
//	Net* DynamicsNet(HyperParamSearcher& param);
//
//	Net* JointNet(HyperParamSearcher& param);
//
//	bool Train_In_Out_Process(Agent* agent, Mat* RandomGenerator, int _step, int test_start_col/*, Agent*Server*/);
//};

//tenhou account
//jackzz7 ID0BBE74EB - dnDCD8ZY

struct Riichi_Python:public Environment {
	static const int PlayerNum = 4;
	static const int TileMaximumID = 3 * 9 + 7;

	//discard tile==>[0,33]
	//closed kan==>34
	//shouminkan kan==>35
	//chi(left,middle,right)==>36,37,38
	//pon==>39
	//open kan==>40
	//ron/chankan/tsumo==>41
	//draw==>42
	//pass==>43
	//riichi==>44
	
	//(34discard+1pass)+35*check+35*4*nextplayer
	static const int W = TileMaximumID;
	static const int H = 1;
	static const int ActionSpace = W * (3 * 4 * 4 + 9);
	//static const int _ActionSpace = W * H * (3 * 4 * 4 + 9);
	//Agent paramters
	static const int filters = 32;
	static const int H_filters = 32;
	int block_Num = 8;
	static const int Action_Depth = 21;

	static const int MaximumGameRounds = 64 + 1;
	//static const int History_Maximum_Num = 8;
	static const int Screen_Size = PlayerNum * (25 + 4 * 4 + 2 + 14) + 6 + 9 + 2 + 10 + 10 + 4 + 4 + 1;
	static const int Image_Depth = 4 * (4 + 4 + 4 + 12) + 25 + 10 + 6 + 22;

	Round_Info round_info[MaximumGameRounds];
	int total_round_number;

	//host
	int InsPlayerID;
	//other seat
	int InsCheckPlayerID;
	//check tile
	int CheckTile;
	//chan kan
	bool Chan_kan;

	vector<int>HandTiles[PlayerNum];
	vector<int> DesktopTilesDisplay[PlayerNum];
	//set of three or four tile
	//(tiles,call tile)
	vector<pair<vector<int>, int>>tileSet[PlayerNum];
	int Score[PlayerNum];
	bool IsRiichi[PlayerNum];
	vector<int>dora_indicators;
	vector<int>wall_tiles;
	//10 tiles
	vector<int>dead_wall_dora;
	//4 tiles
	vector<int>rinshan_tiles;
	//rinshan pop tiles
	vector<int>rinshan_pop_tiles;
	//previous player meld
	//vector<int>previous_player_meld_action;


	vector<int>Action_Mask;
	int Wind[PlayerNum];
	int dealer;
	int remain_tiles_count;
	int round_wind_number;
	int riichi_sticks, honba_sticks;
	bool riichi_furiten, temporary_furiten;
	int results[PlayerNum];
	bool Is_Draw[MaximumGameRounds];



	State_Func StateFunc;
	State_Func MoveStateFunc;
	State_Func SetStateMoveFunc;
	State_Func InitRoundFunc;
	State_Func TerminalStateFunc;
	GameResult_Func SetGameResultFunc;

	DataSet* Reward_ds;
	RewardAgent* rewardAgent = NULL;
	vector<pair<vector<double>,bool>> getRoundsReward() {
		/*vector<vector<double>> res;
		for (int i = 0; i < total_round_number - 1; i++) {
			vector<double>reward;
			for (int j = 0; j < 4; j++)
				reward.push_back(round_info[i + 1].Scores[j]);
			res.push_back(reward);
		}return res;*/
		if (!rewardAgent)rewardAgent = new RewardAgent(new SimpleNet("Reward_Net", "reward_best_param"));
		assert(total_round_number - 1 <= rewardAgent->Batch);
		for (int i = 0; i < rewardAgent->Batch; i++) {
			rewardAgent->DataEncode(i, round_info[i], round_info[i + 1], i >= total_round_number - 1);
		}
		auto rewards = rewardAgent->getReward();
		vector<pair<vector<double>, bool>>res; int _i = 0;
		for (auto& k : rewards) {
			res.push_back({ k,Is_Draw[_i++] });
		}
		return res;
	}
	Riichi_Agent* policyAgent[4] = { 0 };
	vector<vector<double>> getAgentPolicy(int ID) {
		if (!policyAgent[ID]) {
			char path[100]; sprintf(path, "riichi_%d", ID);
			policyAgent[ID] = new Riichi_Agent(new SimpleNet(path, "policy_best_param"));
		}
		double In[Screen_Size]{ 0 };
		GetGameScreen(In);
		MaskScreen(In);
		policyAgent[ID]->DataEncode(0, Screen_Size, In, false);
		return policyAgent[ID]->getAction();
	}

	//int Environment_ID;
	Riichi_Python() :Environment(10, getHiddenStateSize()) {
		rewardAgent = NULL;
	}

	const int seat_Info_Count = 5;
	const int prefix = 16;
	const int LoopCount = prefix + 4 * seat_Info_Count;
	int Loop = 0;
	//for repeat policy without new input info
	bool Repeat_Policy;// , Updata_Screen;
	void ParseList(const string& In, vector<int>& tiles, bool format_to_34_tiles = true);
	void ParseMeld(const string& In, vector<pair<vector<int>, int>>& melds);
	int get_stdout(const string& In, bool Match = false);
	void init_round() {
		step = 0;
		InsPlayerID = InsCheckPlayerID = -1;
		Repeat_Policy = true;
		//Updata_Screen = false;
		for (int i = 0; i < PlayerNum; i++)
			IsRiichi[i] = false;
		total_round_number++;
	}
	inline bool cmp(const string& In, const char* cmd);

	//typedef ui(*Policy_Search_Func)(int step);
	void Environment_Loop(vector<Policy_Func>PolicyFunc, GameResult_Func GameResult_Func, vector<State_Func> StateFuncs, int* match_result, DataSet* reward_ds);
	//void Environment_Match_Loop(vector<Policy_Func> PolicyFunc,int*result);
	void Reset(mt19937* rng = NULL) {
		Loop = 0;
		memset(results, 0, sizeof(results));
		total_round_number = 0;
		for (int i = 0; i < PlayerNum; i++)
			Score[i] = 25000;
	}
	int getInsCheckPlayerID() { return InsCheckPlayerID; }
	int GetNextPlayerID(int action, int InsPlayerID);


	//Console Test Draw Param
	static const int ScrH = 13 + 6 + 2;
	static const int ScrW = 30;
	static const int blank_size = 4;
	int Scr[ScrH][ScrW];
	void DrawScreen();
	char* getPieceAtPosition(int y, int x);
	void PrintScr();


	void GetGameScreen(double* Scr);
	const double getRewardDiscount() { return 1.0; }
	const ui getMaxUnrolledStep() { return 6; }
	const ui getRealGameActionSpace() { return ActionSpace; }
	const ui getSimplifyActionSpace() { return ActionSpace; }
	const ui getScreen_Size() { return Screen_Size; }
	const ui getHiddenStateSize() { return W * H * H_filters; }
	//format action
	const ui getAct(ui move) {
		//return move;
		int type = move / 34;
		int act = move % 34;
		assert(0 <= type && type < 3 * 4 * 4 + 9);
		return act * (3 * 4 * 4 + 9) + type;
	}
	const ui getSimAct(ui move) {
		//return move;
		int type = move % (3 * 4 * 4 + 9), tile = move / (3 * 4 * 4 + 9);
		if (type < 3 * 4 * 4) {
			return type % 3 * 34 + tile + 34 * 3 * (type % (3 * 4) / 3) + 34 * 3 * 4 * (type / (3 * 4));
		}
		else return 34 * 3 * 4 * 4 + (type - 3 * 4 * 4) * 34 + tile;
	}
	bool getSpecialActionMask(ui move, ui action_idx, double Prior) {
		if (move == -1)return true;
		assert(0 <= move && move < ActionSpace);
		int type = move % (3 * 4 * 4 + 9), tile = move / (3 * 4 * 4 + 9);
		int _tile = action_idx / (3 * 4 * 4 + 9);
		int _type = action_idx % (3 * 4 * 4 + 9);
		//chi
		if (type == 49) {
			if (_tile == tile)
				return false;
			if (_tile == tile + 3 && tile % 9 < _tile % 9)
				return false;
		}
		else if (type == 51) {
			if (_tile == tile)
				return false;
			if (_tile == tile - 3 && tile % 9 > _tile % 9)
				return false;
		}
		//pon/shouminkan/open kan
		else if (type == 52 || type == 50 || type == 48 || type == 53) {
			if (_tile == tile)
				return false;
		}
		//only discard after meld
		if (49 <= type && type <= 52) {
			if (_type >= 48 || _type % 3 != 0)
				return false;
		}
		//disable all check ron
		if (_type < 48 && _type % 12 / 3 == 3)
			return false;
		//remove less possible happened actions
		if (Prior < 5e-4)
			return false;


		//only enable some action after check
		if (type < 48) {
			//after no check
			//only discard in hidden search 
			//disable all other actions
			if (type % 12 / 3 == 0) {
				if (_type < 48 && _type % 3 == 0)return true;
				else return false;
			}
			//after check chi
			if (type % 12 / 3 == 1) {
				if (((49 <= _type && _type <= 51) || (_type < 48 && _type % 3 == 1)) && _tile == tile)return true;
				else return false;
			}
			//after check pon/kan
			if (type % 12 / 3 == 2) {
				if (((52 <= _type && _type <= 53) || (_type < 48 && _type % 3 == 1)) && _tile == tile)return true;
				else return false;
			}
			//after check ron
			if (type % 12 / 3 == 3) {
				if (((_type == 54) || (_type < 48 && _type % 3 == 1)) && _tile == tile)return true;
				else return false;
			}
		}
		//no check after kan/riichi
		else if (type == 48 || type == 53 || type == 56) {
			if (_type < 48 && _type % 3 == 0)return true;
			else return false;
		}

		//remove all riichi/ron/check ron action in hidden Search
		//the reason is that ensure value is absolute about state not (win) action(Q value influent)
		//not include terminal actions(easy to fuse value)
		if (_type == 54 || _type == 55 || _type == 56)
			return false;


		return true;
	}
	void StochasticTile(int* Tiles);
	void StochasticEnv();
	void ModifyStochasticActionMask(double* Action_Prior, bool* ActionMask);
	int last_action = -1;
	void MaskScreen(double* Scr);
	void GetInsActionMask(bool* ActIsValid);
	void Act(ui Action, double* Reward);
	bool draw_tile();
	bool GetGameState(int* result);

	//void DirectRepresent(double* Scr);
	void DynamicEncode(const list<Net*>& InPut_Nets, Param* Data, int idx, Mat* Hidden_State, const ui Action, int fillZero);
	void RepresentEncode(Param* Data, int fillZero);

	void RepresentDecode(Agent* agent);
	void DynamicDecode(Agent* agent, int step);

	void Get_NextState_Reward_And_Policy_Value(const list<Net*>& OutPut_Nets, Param* Data, int idx, Mat* Next_State, double* Reward, double* Policy, double* Value);
	void Get_Initial_State_And_Policy_Value(const list<Net*>& OutPut_Nets, Param* Data, int idx, Mat* Next_State, double* Policy, double* Value);

	Net* RepresentationNet(HyperParamSearcher& param);
	Net* DynamicsNet(HyperParamSearcher& param);

	Net* JointNet(HyperParamSearcher& param);

	bool Train_In_Out_Process(Agent* agent, Mat* RandomGenerator, int _step, int test_start_col/*, Agent*Server*/);


	//C++ Process API
	HANDLE g_hChildStd_IN_Rd = NULL;
	HANDLE g_hChildStd_IN_Wr = NULL;
	HANDLE g_hChildStd_OUT_Rd = NULL;
	HANDLE g_hChildStd_OUT_Wr = NULL;
	PROCESS_INFORMATION piProcInfo;
	void Env_CreatePipe() {
		SECURITY_ATTRIBUTES saAttr;

		// Set the bInheritHandle flag so pipe handles are inherited. 

		saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
		saAttr.bInheritHandle = TRUE;
		saAttr.lpSecurityDescriptor = NULL;

		// Create a pipe for the child process's STDOUT. 

		if (!CreatePipe(&g_hChildStd_OUT_Rd, &g_hChildStd_OUT_Wr, &saAttr, 0))
			ErrorExit(TEXT("StdoutRd CreatePipe"));

		// Ensure the read handle to the pipe for STDOUT is not inherited.

		if (!SetHandleInformation(g_hChildStd_OUT_Rd, HANDLE_FLAG_INHERIT, 0))
			ErrorExit(TEXT("Stdout SetHandleInformation"));

		// Create a pipe for the child process's STDIN. 

		if (!CreatePipe(&g_hChildStd_IN_Rd, &g_hChildStd_IN_Wr, &saAttr, 0))
			ErrorExit(TEXT("Stdin CreatePipe"));

		// Ensure the write handle to the pipe for STDIN is not inherited. 

		if (!SetHandleInformation(g_hChildStd_IN_Wr, HANDLE_FLAG_INHERIT, 0))
			ErrorExit(TEXT("Stdin SetHandleInformation"));
	}
	void Env_Close() {
		if (!CloseHandle(g_hChildStd_IN_Wr))
			ErrorExit(TEXT("StdInWr CloseHandle"));

		if (!CloseHandle(g_hChildStd_OUT_Rd))
			ErrorExit(TEXT("StdOutRd CloseHandle"));

		//if (!TerminateProcess(piProcInfo.hProcess, 0))
			//ErrorExit(TEXT("TerminateProcess"));
	}
	void ErrorExit(const wchar_t* lpszFunction)

		// Format a readable error message, display a message box, 
		// and exit from the application.
	{
		LPVOID lpMsgBuf;
		LPVOID lpDisplayBuf;
		DWORD dw = GetLastError();

		FormatMessage(
			FORMAT_MESSAGE_ALLOCATE_BUFFER |
			FORMAT_MESSAGE_FROM_SYSTEM |
			FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL,
			dw,
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
			(LPTSTR)&lpMsgBuf,
			0, NULL);

		lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
			(lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)lpszFunction) + 40) * sizeof(TCHAR));
		StringCchPrintf((LPTSTR)lpDisplayBuf,
			LocalSize(lpDisplayBuf) / sizeof(TCHAR),
			TEXT("%s failed with error %d: %s"),
			lpszFunction, dw, lpMsgBuf);
		MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);

		LocalFree(lpMsgBuf);
		LocalFree(lpDisplayBuf);
		ExitProcess(1);
	}
	void CreateChildProcess(const char*path_to_exe)
		// Create a child process that uses the previously created pipes for STDIN and STDOUT.
	{
		USES_CONVERSION;
		TCHAR*szCmdline = A2W(path_to_exe);
		STARTUPINFO siStartInfo;
		BOOL bSuccess = FALSE;

		// Set up members of the PROCESS_INFORMATION structure. 

		ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));

		// Set up members of the STARTUPINFO structure. 
		// This structure specifies the STDIN and STDOUT handles for redirection.

		ZeroMemory(&siStartInfo, sizeof(STARTUPINFO));
		siStartInfo.cb = sizeof(STARTUPINFO);
		siStartInfo.hStdError = g_hChildStd_OUT_Wr;
		siStartInfo.hStdOutput = g_hChildStd_OUT_Wr;
		siStartInfo.hStdInput = g_hChildStd_IN_Rd;
		siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

		// Create the child process. 

		bSuccess = CreateProcess(NULL,
			szCmdline,     // command line 
			NULL,          // process security attributes 
			NULL,          // primary thread security attributes 
			TRUE,          // handles are inherited 
			0,             // creation flags 
			NULL,          // use parent's environment 
			NULL,          // use parent's current directory 
			&siStartInfo,  // STARTUPINFO pointer 
			&piProcInfo);  // receives PROCESS_INFORMATION 

		 // If an error occurs, exit the application. 
		if (!bSuccess)
			ErrorExit(TEXT("CreateProcess"));
		else
		{
			// Close handles to the child process and its primary thread.
			// Some applications might keep these handles to monitor the status
			// of the child process, for example. 

			//CloseHandle(piProcInfo.hProcess);
			//CloseHandle(piProcInfo.hThread);

			// Close handles to the stdin and stdout pipes no longer needed by the child process.
			// If they are not explicitly closed, there is no way to recognize that the child process has ended.

			CloseHandle(g_hChildStd_OUT_Wr);
			CloseHandle(g_hChildStd_IN_Rd);
		}
	}
	void WriteToPipe(const char*chBuf, DWORD Buf_Size)

		// Read from a file and write its contents to the pipe for the child's STDIN.
		// Stop when there is no more data. 
	{
		DWORD dwRead = Buf_Size, dwWritten;
		//CHAR chBuf[BUFSIZE];
		BOOL bSuccess = FALSE;

		//for (;;)
		{
			//bSuccess = ReadFile(g_hInputFile, chBuf, BUFSIZE, &dwRead, NULL);
			//if (!bSuccess || dwRead == 0) break;

			bSuccess = WriteFile(g_hChildStd_IN_Wr, chBuf, dwRead, &dwWritten, NULL);
			assert(dwRead == dwWritten);
			if (!bSuccess) 
				printf("write to pip error\n"), assert(false);
		}

		// Close the pipe handle so the child process stops reading. 

		//if (!CloseHandle(g_hChildStd_IN_Wr))
			//ErrorExit(TEXT("StdInWr CloseHandle"));
	}
	static const int Max_Buff_Size = 1e5;
	char buff[Max_Buff_Size];
	string _str;
	DWORD ReadFromPipe(char*dst, DWORD BUFSIZE)

		// Read output from the child process's pipe for STDOUT
		// and write to the parent process's pipe for STDOUT. 
		// Stop when there is no more data. 
	{
		DWORD dwRead, dwWritten;
		//CHAR chBuf[BUFSIZE];
		BOOL bSuccess = FALSE;
		//HANDLE hParentStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
		//for (;;)
		{
			bSuccess = ReadFile(g_hChildStd_OUT_Rd, dst, BUFSIZE, &dwRead, NULL);
			if (!bSuccess)
				ErrorExit(TEXT("Read from pip error"));

			//bSuccess = WriteFile(hParentStdOut, chBuf,
				//dwRead, &dwWritten, NULL);
			//if (!bSuccess) break;
		}
		return dwRead;
	}
};