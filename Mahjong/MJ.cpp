#define MAHJONG_EXPORTS
#define _CRT_SECURE_NO_WARNINGS

#include"python_API.h"

#include<Windows.h>

#include"riichi.h"

#include"tenhou.h"

#include"AI_Agent.h"


//#define DEBUG(expr,str)if(expr)printf(str),assert(false);
#define DEBUG(expression,str)(void)(                                                       \
            (!!(expression)) ||                                                              \
            (printf("ERROR expr:%s info:%s file:%s line:%u\n",#expression,str, __FILE__, (unsigned)(__LINE__)),assert(false), 0)) \

char Tiles_Pic[50][10] = { "一萬", "二萬","三萬","四萬","五萬","六萬","七萬","八萬","九萬",
	"一筒", "二筒","三筒","四筒","五筒","六筒","七筒","八筒","九筒",
	"一条", "二条","三条","四条","五条","六条","七条","八条","九条",
	"東風","南風","西風","北風","    ","發财","红中","empty"
};


//PyThreadState**pyThreadState;
bool Enable_Console_Log = true;
void Riichi_Python::Environment_Loop(vector<Policy_Func>PolicyFunc, GameResult_Func GameResult_Func, vector<State_Func> StateFuncs, int* match_result, DataSet* reward_ds) {
	Reward_ds = reward_ds;
	bool Match = match_result ? true : false;
	if (Match)
		Reset();
	Env_CreatePipe();

	CreateChildProcess("Python_Env.exe");

	
	string str = "";
	if (!Match) {
		StateFunc = StateFuncs[0];
		MoveStateFunc = StateFuncs[1];
		SetStateMoveFunc = StateFuncs[2];
		InitRoundFunc = StateFuncs[3];
		TerminalStateFunc = StateFuncs[4];
		SetGameResultFunc = GameResult_Func;
	}
	//send main script path
	string script_path = "C:\\Users\\zz7\\source\\repos\\riichi_python\\Project_extend_\\battle.py";
	//string script_path = "C:\\Users\\ZERGZ\\source\\repos\\Riichi_API\\Project_extend_\\battle.py";
	WriteToPipe(script_path.c_str(), script_path.length());
	int read_size = ReadFromPipe(buff, Max_Buff_Size);

	//if enable bot AI
	int bot_idx = 0;
	for (int i = 0; i < PolicyFunc.size(); i++)if (!PolicyFunc[i])bot_idx = i;
	WriteToPipe(string(1, ('0' + bot_idx)).c_str(), 1);


	std::function<int(const std::string&)>Proc_Output = [&](const std::string& s) {
		int ret = get_stdout(s, Match);
		//response to env
		if (ret == 1) {
			if (!Repeat_Policy) {
				//parse step
				//int shift = InsCheckPlayerID == InsPlayerID ? 0 : ((InsCheckPlayerID - InsPlayerID + PlayerNum) % PlayerNum - 1);
				auto moves = PolicyFunc[Match ? InsCheckPlayerID : 0](step, Match ? 0 : 15, Action_Mask);
				assert(!moves.empty());
				step++;
				//extra_step = 0;
				//to list string
				str = "";
				for (auto& k : moves) {
					str += to_string(k);
					str += " ";
				}
				Repeat_Policy = true;
				//Updata_Screen = false;
				str += '\n';
			}
			WriteToPipe(str.c_str(), str.length());
		}
		return ret;
	};
	while(true) {
		int read_size = ReadFromPipe(buff, Max_Buff_Size);
		buff[read_size] = '\0';
		if (read_size) {
			size_t pos = 0, idx = -1, ret = 0;
			std::string token, delimiter = "@"; _str = buff;
			_str += "@";
			while ((pos = _str.find(delimiter)) != std::string::npos) {
				token = _str.substr(0, pos);
				//std::cout << token << std::endl;
				if (strcmp(token.c_str(), "close pipe") == 0) {
					ret = 1; break;
				}
				if (idx != -1) {
					Proc_Output(token);
				}
				_str.erase(0, pos + delimiter.length());
				idx++;
			}
			//End
			if (ret == 1)
				break;
		}
	}
	

	int sum = 0;
	for (int i = 0; i < 4; i++)sum += results[i];
	if (!match_result&&sum != 25000 * 4) {
		printf("sum scores error\n%s\n", buff), assert(false);
	}

	string msg = "End Process";
	WriteToPipe(msg.c_str(), msg.length());

	if (match_result)
		memcpy(match_result, results, sizeof(results));
	//End Process
	Env_Close();

}
void Riichi_Python::ParseList(const string& In, vector<int>& tiles, bool format_to_34_tiles) {
	tiles.clear();
	const char* buf = In.c_str(); char tmp[100];
	int pos = 0, _pos = 0, tile = -1;
	DEBUG((sscanf(buf, "[%n", &pos)!= -1), "'except ['");
	while (sscanf(&buf[pos], "%d%n", &tile, &_pos) == 1) {
		if (format_to_34_tiles)
			tiles.push_back(tile / 4);
		else tiles.push_back(tile);
		pos += _pos;
		if (sscanf(&buf[pos], "%[^]^0-9^-]%n", tmp, &_pos) == 1)
			pos += _pos;
	}
	assert(buf[pos - 1] == ']' || buf[pos] == ']');
}
void Riichi_Python::ParseMeld(const string& In, vector<pair<vector<int>, int>>& melds) {
	melds.clear();
	const char* buf = In.c_str(); char tmp[100];
	assert(buf[0] == '[');
	int pos = 1, _pos = 0, tile = -1;
	while (sscanf(&buf[pos], "%*[^[]%n", &_pos) != -1) {
		pos += _pos;
		if (buf[pos] != '[')break;
		vector<int>tiles;
		ParseList(buf + pos, tiles);
		//last tile is checktile
		int check = tiles.back();
		if (check >= 34)tiles.pop_back(), check -= 34;
		else check = -1;
		melds.push_back({ tiles,check });
		pos++;
	}assert(buf[pos - 1] == ']');
}
bool Riichi_Python::cmp(const string& In,const char*cmd) {
	return strncmp(In.c_str(), cmd, strlen(cmd)) == 0;
}
int Riichi_Python::get_stdout(const string& In, bool Match) {
	if (In[0] == '\n' || In[0] == '\0')return 0;
	//input message
	if (In[0] == ':') {
		ParseList(In.c_str() + 1, Action_Mask, false);
		if (Enable_Console_Log)
			cout << In.c_str() << endl;
		return 1;
	}
	//for repeat policy without new input info
	Repeat_Policy = false;
	
	//environment real action
	if (In[0] == '#') {
		//ron and open kan
		if ((cmp(In, "#win") || cmp(In, "#openkan") || cmp(In, "#meld") || cmp(In, "#pass"))) {
			vector<int>win;
			int idx = In.find('[');
			assert(idx != -1);
			ParseList(In.c_str() + idx, win);
			int action = -1;
			const int MOVE_WIN = 1;
			if (cmp(In, "#win"))action = MOVE_WIN;
			else if (cmp(In, "#openkan"))action = -1;//MOVE_OPEN_KAN;
			else if (cmp(In, "#pass")) {
				assert(win.size() == 1);
				/*int id = (win[0] - InsPlayerID + PlayerNum) % PlayerNum;
				if (id == 1)action = MOVE_ALL_PASS_4;
				else if (id == 2)action = MOVE_ALL_PASS_3;
				else if (id == 3)action = MOVE_ALL_PASS_2;
				else assert(false);*/
			}
			else action = -1;
			assert(action == MOVE_WIN || win.size() == 1);
			//tsumo
			if (action == MOVE_WIN && win[0] == InsPlayerID) {
				//assert(InsCheckPlayerID == InsPlayerID);
				/*SetStateMoveFunc(step + 1, MOVE_WIN);
				step++;*/
			}
			else {
				//int move[4] = { 0 }; for (auto& k : win)move[(k - InsPlayerID + PlayerNum - 1) % PlayerNum] = 1, assert(k != InsPlayerID);
				//int cnt = 0; for (int i = 0; i < 3; i++)if (move[i] == 1)cnt = i + 1;
				////chankan additional meld pass
				//if (Chan_kan) {
				//	assert(action == MOVE_WIN);
				//	for (int i = cnt - 1; i > -1; i--)
				//		if (move[i] == 1)
				//			MoveStateFunc(step + i + 1, step + i);
				//	StateFunc(step, MOVE_MELD_PASS);
				//	step++;
				//}
				////force passed
				//int two_pass = 0;
				//for (int i = 0; i < cnt; i++)
				//	if (move[i] == 0) {
				//		if (i + 1 < cnt && move[i + 1] == 0) {
				//			StateFunc(step + i, MOVE_MELD_PASS_2);
				//			i++; two_pass = 1;
				//			MoveStateFunc(step + 1, step + 2);
				//		}
				//		else StateFunc(step + i, MOVE_MELD_PASS);
				//	}
				//	else if (action != -1)SetStateMoveFunc(step + i + 1 - two_pass, action);
				//step += cnt;
				////extra discard after meld
				////chi or pon
				//if (cmp(In, "#meld")) {
				//	assert(win.size() == 1);
				//	MoveStateFunc(step - two_pass, step - 1 + 3);
				//	//step++;
				//}
				//step -= two_pass;
				//assert(InsCheckPlayerID != InsPlayerID);
			}
			//Terminal State
			if (cmp(In, "#win") && !Match) {
				TerminalStateFunc(step, -1);
				step++;
			}
		}
		//discard after pon,do some changes
		else if (cmp(In, "#pon")) {
			/*if (!Match) {
				int shift = (InsCheckPlayerID - InsPlayerID + PlayerNum) % PlayerNum - 1;
				SetStateMoveFunc(step + shift + 1, MOVE_PON);
				extra_step = 3;
			}*/
			//do some changes
			int cnt = 2; auto& hand = HandTiles[InsCheckPlayerID];
			for (auto it = hand.begin(); it != hand.end();) {
				if (cnt > 0 && *it == CheckTile)it = hand.erase(it), cnt--;
				else it++;
			}
			assert(cnt == 0);
			vector<int>tmp; tmp.assign(3, CheckTile);
			tileSet[InsCheckPlayerID].push_back({ tmp,CheckTile });
		}
		else if (cmp(In, "#chi")) {
			//int shift = (InsCheckPlayerID - InsPlayerID + PlayerNum) % PlayerNum - 1;
			vector<int>tiles;
			ParseList(In.c_str() + 4, tiles);
			assert(0 <= CheckTile < 27);
			//int move = CheckTile == tiles[0] ? MOVE_CHI_L : (CheckTile == tiles[2] ? MOVE_CHI_R : MOVE_CHI_M);
			//if (!Match)SetStateMoveFunc(step + shift + 1, move);
			//extra_step = 3; 
			auto& hand = HandTiles[InsCheckPlayerID];
			for (auto& tile : tiles) {
				if (tile == CheckTile)continue;
				bool pass = false; for (auto it = hand.begin(); it != hand.end();it++)if (*it == tile) { hand.erase(it); pass = true; break; }
				assert(pass);
			}
			tileSet[InsCheckPlayerID].push_back({ tiles,CheckTile });
		}
		else if (cmp(In, "#discard")) {
			int tile = atoi(In.c_str() + 8) / 4;
			//SetStateMoveFunc(step + 1, tile);
			assert(0 <= tile && tile < 34);
			//auto discard from riichi
			/*if (Updata_Screen) {
				assert(InsCheckPlayerID == InsPlayerID);
				assert(IsRiichi[InsPlayerID]);
			}*/
			//step++;
		}
		else if ((cmp(In, "#draw") || cmp(In, "#shouminkan") || cmp(In, "#closedkan"))) {
			//int move = cmp(In, "#draw") ? MOVE_DRAW : (cmp(In, "#shouminkan") ? MOVE_SHOUMIN_KAN : MOVE_CLOSED_KAN);
			//SetStateMoveFunc(step + 1, move);
			//step++;
		}
		else if (cmp(In, "#riichi")) {
			/*if (!Match) {
				SetStateMoveFunc(step + 1, MOVE_RIICHI);
				step++;
			}*/
			IsRiichi[InsPlayerID] = true;
		}
		else if (cmp(In, "#init")) {
			Is_Draw[total_round_number] = false;
			init_round();
			if (!Match) {
				InitRoundFunc(-1, -1);
			}
		}
		//End Game
		else if (cmp(In, "#result")) {
			vector<int>result;
			ParseList(In.c_str() + 7, result, false);
			if (Is_Draw[total_round_number - 1])
				result.assign(4, 25000);
			int i = 0; for (auto& score : result)results[i++] = score;
			if (!Match) {
				SetGameResultFunc(results);
				//add reward predictor train data
				round_info[total_round_number++] = Round_Info(-1, results, 0, 0);
				for (int i = 0; i < total_round_number - 1; i++) {
					Reward_ds->trainSet_Add_data(RewardData(round_info[i], round_info[i + 1], result));
				}
			}
			//check riichi flag
			else {
				for (int i = 0; i < 4; i++)if (IsRiichi[i])results[i] |= 1;
			}
			return 2;
		}
		else {
			string draw = In;
			Is_Draw[total_round_number - 1] = true;
			//Terminal State
			if (!Match)
				TerminalStateFunc(step, -1);
			step++;
		}

		return 0;
	}




	//get full Game Info
	//seat No
	if (Loop % LoopCount < prefix) {
		//seat
		if (Loop == 0)
			InsPlayerID = atoi(In.c_str());
		//check seat
		else if (Loop == 1)InsCheckPlayerID = atoi(In.c_str());
		//remain tiles
		else if (Loop == 2)remain_tiles_count = atoi(In.c_str());
		//check tile
		else if (Loop == 3)CheckTile = atoi(In.c_str()) / 4;
		//chan kan
		else if (Loop == 4)Chan_kan = strcmp(In.c_str(), "False") != 0;
		//dora
		else if (Loop == 5) {
			ParseList(In, dora_indicators);
		}
		//dealer
		else if (Loop == 6)dealer = atoi(In.c_str());
		//round wind
		else if (Loop == 7)round_wind_number = atoi(In.c_str());
		else if (Loop == 8)riichi_sticks = atoi(In.c_str());
		else if (Loop == 9)honba_sticks = atoi(In.c_str());
		else if (Loop == 10)riichi_furiten = strcmp(In.c_str(), "False") != 0;
		else if (Loop == 11)temporary_furiten = strcmp(In.c_str(), "False") != 0;
		else if (Loop == 12)ParseList(In, wall_tiles);
		else if (Loop == 13)ParseList(In, dead_wall_dora);
		else if (Loop == 14)ParseList(In, rinshan_tiles);
		else if (Loop == 15)ParseList(In, rinshan_pop_tiles);
		//else if (Loop == 16)ParseList(In, previous_player_meld_action, false);
		else assert(false);
		Loop++;
		return 0;
	}
	int No = (Loop - prefix) % seat_Info_Count, id = (Loop - prefix) / seat_Info_Count % PlayerNum;
	if (No == 0)
		ParseList(In, HandTiles[id]);
	else if (No == 1)
		ParseList(In, DesktopTilesDisplay[id]);
	else if (No == 2)
		Score[id] = atoi(In.c_str());
	else if (No == 3)
		ParseMeld(In, tileSet[id]);
	else if (No == 4)
		Wind[id] = atoi(In.c_str());

	if (++Loop % LoopCount == 0) {
		if (Enable_Console_Log)
			PrintScr();
		Loop = 0; //Updata_Screen = true;

		if (step == 0) {
			round_info[total_round_number-1] = Round_Info(round_wind_number, Score, riichi_sticks, honba_sticks);
		}
	}
		
	return 0;
}


int Riichi_Python::GetNextPlayerID(int action, int InsPlayerID) {
	//discard:34action+35*check+35*4*nextplayer
	//pass:1pass+35*check+35*4*nextplayer
	int type = action % (3 * 4 * 4 + 9);
	if (type < 3 * 4 * 4) {
		return (InsPlayerID + type / (3 * 4)) % PlayerNum;
	}
	//other action return self
	else return InsPlayerID;
	//if (action < 36 * 4 * 4) {
	//	return (InsPlayerID + action / (36 * 4)) % PlayerNum;
	//}
	////other action return self
	//else return InsPlayerID;



	//if (action < TileMaximumID)return (++InsPlayerID) %= PlayerNum;
	//assert(0 <= action && action < ActionSpace);
	//if ((MOVE_CLOSED_KAN <= action && action <= MOVE_OPEN_KAN) || action == MOVE_RIICHI||action==MOVE_DRAW)return InsPlayerID;
	//else {
	//	//assert(action == MOVE_PASS);
	//	return (++InsPlayerID) %= PlayerNum;
	//}
}

void Riichi_Python::DrawScreen() {
	memset(Scr, -1, ScrH * ScrW * sizeof(int));
	//front player
	int ID = (InsPlayerID + 2) % PlayerNum; int idx = 0;
	for (auto& tile : tileSet[ID]) {
		for (int i = 0; i < tile.X.size(); i++)
			Scr[0][2 + idx] = tile.X[i], idx++;
		//idx++;
	}
	idx = 0; vector<int>tiles = HandTiles[ID];
	sort(tiles.begin(), tiles.end());
	for (auto& tile : tiles) {
		Scr[1][2 + 13 - idx] = tile;
		idx++;
	}
	idx = 0; auto* droptiles = &DesktopTilesDisplay[ID];
	for (auto& tile : *droptiles) {
		Scr[5 - idx / 6][6 + 5 - idx % 6] = tile;
		idx++;
	}
	//left player
	ID = (InsPlayerID + 3) % PlayerNum; idx = 0;
	for (auto& tile : tileSet[ID]) {
		for (int i = 0; i < tile.X.size(); i++)
			Scr[15 - idx][0] = tile.X[i], idx++;
		//idx++;
	}
	idx = 0; tiles = HandTiles[ID];
	sort(tiles.begin(), tiles.end());
	for (auto& tile : tiles) {
		Scr[idx + 2][1] = tile;
		idx++;
	}
	idx = 0; droptiles = &DesktopTilesDisplay[ID];
	for (auto& tile : *droptiles) {
		Scr[6 + idx % 6][5 - idx / 6] = tile;
		idx++;
	}
	//self
	ID = InsPlayerID; idx = 0;
	for (auto& tile : tileSet[ID]) {
		for (int i = 0; i < tile.X.size(); i++)
			Scr[17][15 - idx] = tile.X[i], idx++;
		//idx++;
	}
	idx = 0; tiles = HandTiles[ID];
	sort(tiles.begin(), tiles.end());
	for (auto& tile : tiles) {
		Scr[16][2 + idx] = tile;
		idx++;
	}
	idx = 0; droptiles = &DesktopTilesDisplay[ID];
	for (auto& tile : *droptiles) {
		Scr[12 + idx / 6][6 + idx % 6] = tile;
		idx++;
	}
	//right Player
	ID = (InsPlayerID + 1) % PlayerNum;; idx = 0;
	for (auto& tile : tileSet[ID]) {
		for (int i = 0; i < tile.X.size(); i++)
			Scr[2 + idx][17] = tile.X[i], idx++;
		//idx++;
	}
	idx = 0; tiles = HandTiles[ID];
	sort(tiles.begin(), tiles.end());
	for (auto& tile : tiles) {
		Scr[15 - idx][16] = tile;
		idx++;
	}
	idx = 0; droptiles = &DesktopTilesDisplay[ID];
	for (auto& tile : *droptiles) {
		Scr[6 + 5 - idx % 6][12 + idx / 6] = tile;
		idx++;
	}
}
char* Riichi_Python::getPieceAtPosition(int y, int x) {
	if (Scr[y][x] == -1)return Tiles_Pic[TileMaximumID];
	else return Tiles_Pic[Scr[y][x]];
}

#define WHITE_SQUARE 0
//BACKGROUND_BLUE | BACKGROUND_GREEN | BACKGROUND_RED
#define BLACK_SQUARE BACKGROUND_BLUE | BACKGROUND_GREEN | BACKGROUND_RED
HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
//template<class T>
void printLine(int iLine, int iColor1, int iColor2, Riichi_Python& chess);
int CELL = 2;
const int char_count = 3;
// x is the column, y is the row. The origin (0,0) is top-left.
void setCursorPosition(int x, int y)
{
	static const HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	std::cout.flush();
	COORD coord = { (SHORT)x, (SHORT)y };
	SetConsoleCursorPosition(hOut, coord);
}
void clearline() {
	CONSOLE_SCREEN_BUFFER_INFO cbsi;
	if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &cbsi));
	COORD coord = cbsi.dwCursorPosition;
#undef X
#undef Y
	setCursorPosition(0, coord.Y);
	for (int i = 0; i < cbsi.dwSize.X; i++)cout << " ";
	cout << '\n';
}
//template<class T>
void printBoard(Riichi_Python& mj)
{
	mj.DrawScreen();
	setCursorPosition(0, 0);
	//system("cls");
	for (int iLine = 0; iLine <= 17; iLine++)
	{
		printLine(iLine, BLACK_SQUARE, WHITE_SQUARE, mj);
	}
	SetConsoleTextAttribute(hConsole, FOREGROUND_BLUE | FOREGROUND_RED | FOREGROUND_GREEN);
	for (int i = 0; i < char_count + 1; i++)cout << " ";
	cout << endl;
	for (int i = 0; i < 4; i++) {
		int ID = (mj.InsPlayerID + i) % mj.PlayerNum;
		printf("%d(%s):%d ", ID, Tiles_Pic[(ID - mj.dealer + mj.PlayerNum) % mj.PlayerNum + 27], mj.Score[ID]);
	}cout << endl;
	//clear dynamic text line
	for (int i = 0; i < 10; i++)
		clearline();
	CONSOLE_SCREEN_BUFFER_INFO cbsi;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &cbsi);
	setCursorPosition(0, cbsi.dwCursorPosition.Y - 10);
}
//template<class T>
void printLine(int iLine, int iColor1, int iColor2, Riichi_Python& mj)
{
	for (int subLine = 0; subLine < CELL / 2; subLine++)
	{
		for (int iPair = 0; iPair < 18; iPair++)
		{
			for (int subColumn = 0; subColumn < 1; subColumn++) {
				SetConsoleTextAttribute(hConsole, iColor1);
				const char* str = mj.getPieceAtPosition(iLine, iPair);
				if (str == (string)("empty"))
					SetConsoleTextAttribute(hConsole, iColor2), str = "    ";
				else {
					int TextColor = 0;
					if (mj.Scr[iLine][iPair] < 9 || mj.Scr[iLine][iPair] == 32)TextColor |= FOREGROUND_RED;
					else if (mj.Scr[iLine][iPair] < 18 || mj.Scr[iLine][iPair] == 31)TextColor |= FOREGROUND_BLUE;
					else if (mj.Scr[iLine][iPair] < 27)TextColor |= FOREGROUND_GREEN;
					else TextColor |= WHITE_SQUARE;
					SetConsoleTextAttribute(hConsole, iColor1 | TextColor);
				}
				printf("%s", str);
				SetConsoleTextAttribute(hConsole, iColor2);
				printf(" ");
			}
		}
		//blank
		//SetConsoleTextAttribute(hConsole, FOREGROUND_BLUE | FOREGROUND_RED | FOREGROUND_GREEN);
		cout << '\n';
	}
}
void Riichi_Python::PrintScr() {
	//_setmode(_fileno(stdout), _O_U16TEXT);
	printBoard(*this);
	//_setmode(_fileno(stdout), _O_TEXT);
}
void setFontSize(int size) {
	HANDLE Hout = ::GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_FONT_INFOEX Font = { sizeof(Font) };
	::GetCurrentConsoleFontEx(Hout, FALSE, &Font);
	COORD fsize = { size,size };
	Font.dwFontSize = fsize;
	SetCurrentConsoleFontEx(Hout, 0, &Font);
}
bool SetWindowSize(size_t width, size_t height, size_t buff_width, size_t buff_height)
{
	HANDLE output_handle = ::GetStdHandle(STD_OUTPUT_HANDLE);
	if (output_handle == INVALID_HANDLE_VALUE)
		return false;

	RECT r; GetWindowRect(GetConsoleWindow(), &r);
	MoveWindow(GetConsoleWindow(), r.left, r.top, r.left + width, r.top + height, TRUE);

	return (::SetConsoleScreenBufferSize(output_handle, { (SHORT)buff_width ,(SHORT)buff_height }) != FALSE);
}

#include <locale>
#include <string>
#include <codecvt>

Riichi_Python test;

//void Python_Environment_Loop() {
//	PyImport_AppendInittab("emb", emb::PyInit_emb);
//	_putenv_s("PYTHONPATH", "C:\\Users\\ZERGZ\\source\\repos\\Riichi_API\\Project\\");
//	//_putenv_s("PYTHONPATH", "C:\\Users\\zz7\\source\\repos\\riichi_python\\Project\\");
//	Py_Initialize();
//	PyImport_ImportModule("emb");
//	//PyObject* sys = PyImport_ImportModule("sys");
//	//char filename[] = "C:\\Users\\zz7\\source\\repos\\riichi_python\\Project\\battle.py";
//	char filename[] = "C:\\Users\\ZERGZ\\source\\repos\\Riichi_API\\Project\\battle.py";
//	FILE* fp;
//	
//	// here comes the ***magic***
//	std::string buffer;
//	{
//		// switch sys.stdout to custom handler
//		emb::stdout_write_type write = [&buffer](const std::string&s) { buffer += s; test.get_stdout(s); };
//		emb::stdin_read_type read = [&buffer](std::string& s) { cin >> s; };
//		emb::set_stdout_and_stdin(write, read);
//		fp = _Py_fopen(filename, "r");
//		PyRun_SimpleFile(fp, filename);
//		emb::reset_stdout_and_stdin();
//	}
//	Py_Finalize();
//}
TenHou tenhou;
int main() {
	setFontSize(20);
	SetWindowSize(1200, 600, 0, 9999);
	SetConsoleOutputCP(65001);

	//auto* thr = new std::thread(&Python_Environment_Loop);

	srand(time(0));
	//freopen("test", "r", stdin);

	//while (true) {
	//	this_thread::sleep_for(std::chrono::milliseconds(1000));
	//	//pipeOut("adf\n");
	//	//char ch = getchar();
	//	//cout << ch;
	//	//string str = "1\n";
	//	//std::stringstream ss;
	//	//ss << "Number of people is ";
	//	//
	//	////_write(STDIN_FILENO, str.c_str(), str.size());
	//	////solve(std::cin, "bla");
	//	//const std::string input("String with 3.14159 * 2");
	//	//FILE* old_stdin = stdin;

	//	//FILE* strm = fmemopen((void*)input.c_str(), input.size(), "r");
	//	//stdin = strm;
	//}
	
	//test.ParseList("[3, 9, 7]");
	

	// output what was written to buffer object
	//std::clog << buffer << std::endl;

	//PyImport_AppendInittab("emb", emb::PyInit_emb);
	//_putenv_s("PYTHONPATH", "C:\\Users\\zz7\\source\\repos\\riichi_python\\Project\\");
	//////Init Python thread support
	//Py_Initialize();
	//PyEval_InitThreads();
	//PyThreadState* _main = PyThreadState_Get();
	
	/*tenhou.run();
	return 0;*/
	
	//RewardAgent reward(new SimpleNet("Reward_Net", "reward_best_param", true));
	//RewardAgent reward("reward_best_param");
	////reward.combine_ds(5, 5);
	//reward.train("extend_trainSet_data_reward_5", 50);
	//return 0;

	/*Riichi_Agent agent("agent_best_param");
	agent.train("", 20, 60);*/

	//add extra meld pass Q ?? compare

	//add wall tiles to perfect MCTS Policy Search with RNN Agent
	//generate better (State,Policy) data (test)
	//remove part State to train a new State,Action Agent (more block,filters)


	//chi=>chi_disacrd=>pon/pass
	//specify meld type and chi discard tile
	//1.modify env,change action select priority(ron>pon/kan>chi)
	//2.use absolute playerID to full train ID softmax
	//reanalyse

    //as an extend:
	//train a function:f(S,a)=>A(target=S'- S using real simple State data)as a State Updata Func for State transition f(S,A)=>S'
	//for the purpose:easy to know State,but hard to know or difficult to encode abstract action
	
	//enchance Policy using prior Policy sample??
	//check whether enchance when using imperfect info(using more simulations)
	


	/*1.add 场风，客风，lastest discard tiles feature
		2.remove check ron / ron
		3.remove shouminkan ron transfer, discard ron
		3.modify dynamic action, binary plane*/

	//test AI bug:win but result not change
	//fstream file; file.open("loss_data.txt", ios::out);
	//char tmp[100];
	//sprintf(tmp, "%.02f", 1.22);
	//string str = tmp;
	//file.write(str.c_str(), str.length());
	//file.close();

	//一般模型有序表示打出牌

	int Run_Type = 2;

	Enable_Console_Log = Run_Type;

	int Generator_Count = Run_Type == 1 ? 4 : 10;
	Riichi_Python** riichi = new Riichi_Python * [Generator_Count];
	//pyThreadState = new PyThreadState * [Generator_Count];
	for (int i = 0; i < Generator_Count; i++)riichi[i] = new Riichi_Python();// , pyThreadState[i] = Py_NewInterpreter();

	//PyEval_SaveThread();
	//PyThreadState_Swap(NULL);
	//PyEval_SaveThread();
	AgentGenerator agents("riichi_param_Rec", "riichi_best_param");
	agents.MCTS_run_extend(Run_Type, (Environment**)riichi, Generator_Count, true);


	//for (int i = 0; i < Generator_Count; i++) {
	//	delete riichi[i];
	//	PyThreadState_Swap(pyThreadState[i]);
	//	Py_EndInterpreter(pyThreadState[i]);
	//}
	//delete[] riichi;
	//delete[] pyThreadState;
	////End Python thread state
	////PyThreadState_Swap(_main);
	//Py_Finalize();
	return 0;

	//mt19937 rng(time(0));
	//RiiChi Game; bool actMask[RiiChi::ActionSpace];
	//Game.Reset(&rng);
	//int result[4];
	////host draw one tile
	//Game.PrintScr();
	//while (true) {
	//	if (Game.GetGameState(result))break;
	//	//drop tile or FourSet or win
	//	Game.GetInsActionMask(actMask, -1, true);
	//	//select action
	//	int cnt = 0, action;
	//	for (int i = 0; i < Game.ActionSpace; i++)
	//		if (actMask[i])action = i, cnt++;
	//	if (cnt > 1) {
	//		do {
	//			printf("Input: ");
	//			while (scanf("%d", &action) != 1);
	//		} while (action >= Game.ActionSpace || !actMask[action]);
	//	}
	//	else printf("Input: %d\n", action), assert(cnt == 1);
	//	printf("player:%d Select: %d\n", Game.InsPlayerID, action);

	//	Game.Act(action, 0);
	//	Game.PrintScr();

	//	//check other players' priority actions
	//	int Others_Action[RiiChi::PlayerNum] = { 0 };
	//	for (int i = 0; i < RiiChi::PlayerNum - 1; i++) {
	//		Game.GetInsActionMask(actMask, i, i == 0);

	//		int cnt = 0;
	//		for (int i = 0; i < Game.ActionSpace; i++)
	//			if (actMask[i])action = i, cnt++;
	//		if (cnt > 1) {
	//			do {
	//				printf("Input: ");
	//				while (scanf("%d", &action) != 1);
	//			} while (action >= Game.ActionSpace || !actMask[action]);
	//		}
	//		else printf("Input: %d\n", action), assert(cnt == 1);
	//		printf("player:%d Select: %d\n", (Game.InsPlayerID + i + 1) % RiiChi::PlayerNum, action);
	//		Others_Action[i] = action;
	//	}

	//	Game.OpponentsAct(Others_Action);

	//	Game.PrintScr();
	//}
	//
	//printf("final result score: ");
	//for (int i = 0; i < Game.PlayerNum; i++)printf("%d ", Game.Score[i]);
	//printf("\n");
}