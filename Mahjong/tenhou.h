#pragma once
#include"riichi.h"

#include<string>

struct TenHou :Riichi_Python {
	TenHou() {

	}

	bool Enable_Console_Log = true;
	void Environment_Loop(vector<Policy_Func>PolicyFunc, GameResult_Func GameResult_Func, vector<State_Func> StateFuncs, int* match_result, DataSet* reward_ds) {
		Reset();
		Env_CreatePipe();
		CreateChildProcess("Python_Env.exe");

		string str = "";
		//send main script path
		//string script_path = "C:\\Users\\zz7\\source\\repos\\riichi_python\\Project_extend_mod\\tenhou_battle.py";
		string script_path = "C:\\Users\\ZERGZ\\source\\repos\\Riichi_API\\Project_extend_mod\\tenhou_battle.py";
		WriteToPipe(script_path.c_str(), script_path.length());
		std::function<int(const std::string&)>Proc_Output = [&](const std::string& s) {
			int ret = get_tenhou_response(s);
			//response to env
			if (ret == 1) {
				if (!Repeat_Policy) {
					//parse step
					//int shift = InsCheckPlayerID == InsPlayerID ? 0 : ((InsCheckPlayerID - InsPlayerID + PlayerNum) % PlayerNum - 1);
					auto moves = PolicyFunc[0](step, 0, Action_Mask);
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
		while (true) {
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
		string msg = "End Process";
		WriteToPipe(msg.c_str(), msg.length());

		int sum = 0;
		for (int i = 0; i < 4; i++)sum += results[i];
		if (sum != 25000 * 4)printf("sum scores error\n"), assert(false);

		if (match_result)
			memcpy(match_result, results, sizeof(results));
		//End Process
		Env_Close();
	}
	int get_tenhou_response(const string& In) {
		if (In[0] == '\n' || In[0] == '\0')return 0;
		//input message
		if (In[0] == ':') {
			ParseList(In.c_str() + 1, Action_Mask, false);
			if (Enable_Console_Log)
				cout << In.c_str() << endl;
			return 1;
		}
		//for repeat policy without new input info
		//save policy search cal
		Repeat_Policy = false;

		//environment real action
		if (In[0] == '#') {
			//discard after pon,do some changes
			if (cmp(In, "#pon")) {
				//do some changes after meld
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
				vector<int>tiles;
				ParseList(In.c_str() + 4, tiles);
				assert(0 <= CheckTile < 27);
				//int move = CheckTile == tiles[0] ? MOVE_CHI_L : (CheckTile == tiles[2] ? MOVE_CHI_R : MOVE_CHI_M);
				auto& hand = HandTiles[InsCheckPlayerID];
				for (auto& tile : tiles) {
					if (tile == CheckTile)continue;
					bool pass = false; for (auto it = hand.begin(); it != hand.end(); it++)if (*it == tile) { hand.erase(it); pass = true; break; }
					assert(pass);
				}
				tileSet[InsCheckPlayerID].push_back({ tiles,CheckTile });
			}
			else if (cmp(In, "#riichi")) {
				//set player riichi
				if (In.length() == 7)
					IsRiichi[InsPlayerID] = true;
				else {
					int id = atoi(In.c_str() + 7);
					IsRiichi[id] = true;
				}
			}
			else if (cmp(In, "#init")) {
				init_round();
			}
			//End Game
			else if (cmp(In, "#result")) {
				vector<int>result;
				ParseList(In.c_str() + 7, result);
				int res[PlayerNum] = { 0 }, i = 0;; for (auto& score : result)res[i++] = score * 4;
				memcpy(results, res, sizeof(res));
				return 2;
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
		}

		return 0;
	}
	void run() {
		Agent_API* API = Get_multiMCTS_Extend_API();
		API->MCTS_Agent_Init("ExAgent_#6", "riichi_best_param", "", *this);
		API->MCTS_Agent_Run();
	}
};