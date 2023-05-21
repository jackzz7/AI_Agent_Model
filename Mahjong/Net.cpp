#include"riichi.h"

//8*4*(1+4*3+5+1+1)+4*36+14+5
//8*4*(1+4*5+3)+4+3+9
void Riichi_Python::GetGameScreen(double* Scr) {
	fill(Scr, Scr + Screen_Size, -1);
	int _i = 0;
	for (int j = 0; j < PlayerNum; j++) {
		int ID = (InsCheckPlayerID + j) % PlayerNum;
		//players full drop tiles
		for (auto& tile : DesktopTilesDisplay[ID])Scr[_i++] = tile;
		assert(DesktopTilesDisplay[ID].size() <= 25);
		_i += 25 - DesktopTilesDisplay[ID].size();
		//players tiles meld
		auto it = tileSet[ID].begin();
		for (int k = 0; k < 4; k++)
			if (k < tileSet[ID].size()) {
				for (auto& tile : it->X)Scr[_i++] = tile;
				if (it->X.size() == 3)_i++;
				it++;
				//Scr[_i++] = it.X, Scr[_i++] = it->Y.X, /*Scr[_i++] = (it->Y.Y - InsCheckPlayerID + PlayerNum) % PlayerNum,*/ it++;
			}
			else _i += 4;
		//player with riichi 
		Scr[_i++] = IsRiichi[ID];
		//score
		Scr[_i++] = Score[ID] / 100;
	}
	//first using full hand tiles
	//InsPlayer hand tiles
	for (int i = 0; i < PlayerNum; i++) {
		int ID = (InsCheckPlayerID + i) % PlayerNum;
		auto& tiles = HandTiles[ID];
		for (auto& tile : tiles)Scr[_i++] = tile;
		_i += 14 - tiles.size();
	}
	//remain library tiles count
	Scr[_i++] = remain_tiles_count;
	//check tile
	assert(0 <= CheckTile < TileMaximumID);
	Scr[_i++] = CheckTile;
	//chan kan
	Scr[_i++] = Chan_kan;
	//lastest PlayerID
	Scr[_i++] = (InsPlayerID - InsCheckPlayerID + PlayerNum) % PlayerNum;
	//InsPlayer seat
	Scr[_i++] = InsCheckPlayerID;
	//dealer seat
	Scr[_i++] = (dealer - InsCheckPlayerID + PlayerNum) % PlayerNum;
	//dora indicators
	assert(dora_indicators.size() <= 5);
	for (auto& dora : dora_indicators)Scr[_i++] = dora;
	_i += 5 - dora_indicators.size();
	//round wind number
	assert(0 <= round_wind_number && round_wind_number < 12);
	//round wind
	Scr[_i++] = round_wind_number / 4;
	//remain least round number
	Scr[_i++] = round_wind_number;// max(0, 4 - round_wind_number);
	//riichi sticks
	Scr[_i++] = riichi_sticks;
	//honba sticks
	Scr[_i++] = honba_sticks;
	//furiten
	Scr[_i++] = riichi_furiten;
	Scr[_i++] = temporary_furiten;
	//10+10+4+4+8
	//recent n=10 wall tiles
	const int recent_tiles = 10;
	for (int j = 0; j < recent_tiles; j++) {
		if (j >= wall_tiles.size())continue;
		Scr[_i++] = wall_tiles[j];
	}
	_i += recent_tiles - min((int)wall_tiles.size(), recent_tiles);
	//dead wall dora tiles
	assert(dead_wall_dora.size() == 14);
	for (int j = 0; j < 14; j++) {
		if (j == 0 || j == 1 || j == 7 || j == 8)continue;
		Scr[_i++] = dead_wall_dora[j];
	}
	//rinshan tiles
	assert(rinshan_tiles.size() <= 4);
	for (auto& tile : rinshan_tiles)Scr[_i++] = tile;
	_i += 4 - rinshan_tiles.size();
	//rinshan pop wall tiles
	assert(rinshan_pop_tiles.size() <= 4);
	for (auto& tile : rinshan_pop_tiles)Scr[_i++] = tile;
	_i += 4 - rinshan_pop_tiles.size();
	//special actions hint
	//check ron/kan/pon/chi/none
	int action_type = -1;
	if (InsPlayerID != InsCheckPlayerID) {
		for (auto& act : Action_Mask)if (act / 34 == 54)assert(action_type == -1), action_type = 3;
		else if (act / 34 == 53 || act / 34 == 52)assert(action_type == -1 || action_type == 2), action_type = 2;
		else if (act / 34 == 49 || act / 34 == 50 || act / 34 == 51)assert(action_type == -1 || action_type == 1), action_type = 1;
	}
	if (action_type == -1)action_type = 0;
	Scr[_i++] = action_type;



	assert(_i == Screen_Size);
}
void Riichi_Python::StochasticTile(int*Tiles) {
	mt19937 rng(time(0));
	vector<int>tilelib;
	for (int i = 0; i < 34; i++)
		for (int j = 0; j < Tiles[i]; j++)tilelib.push_back(i * 4 + j);
	int ord[136]; for (int i = 0; i < 136; i++)ord[i] = rng();
	sort(tilelib.begin(), tilelib.end(), [&](const int& a, const int& b) {return ord[a] < ord[b]; });
	for (auto& tile : tilelib)tile /= 4;
	
	//fill three opponent tiles
	for (int i = 1; i < 4; i++) {
		int ID = (InsCheckPlayerID + i) % PlayerNum;
		auto& tiles = HandTiles[ID];
		for (auto& tile : tiles)
			tile = tilelib.back(), tilelib.pop_back();
	}
	//wall tiles
	for (auto& tile : wall_tiles)tile = tilelib.back(), tilelib.pop_back();
	//dead wall tiles
	for (auto& tile : dead_wall_dora)tile = tilelib.back(), tilelib.pop_back();
	assert(tilelib.empty());

	//rinshan tiles
	int rinshan_idx[4] = { 0, 1, 7, 8 }, _i = 0;
	//rinshan pop tiles
	for (auto& tile : rinshan_pop_tiles)tile = dead_wall_dora[rinshan_idx[_i++]];
	for (auto& tile : rinshan_tiles)tile = dead_wall_dora[rinshan_idx[_i++]];
	assert(_i == 4);
}
void Riichi_Python::StochasticEnv() {
	int Tiles[34];
	fill(Tiles, Tiles + 34, 4);
	
	for (int i = 0; i < 4; i++) {
		//remove desktop tiles
		for (auto& tile : DesktopTilesDisplay[i])Tiles[tile]--;
		//remove meld tiles
		for (auto& it : tileSet[i]) {
			for (auto& tile : it.X)Tiles[tile]--;
			if (it.Y != -1)Tiles[it.Y]++;
		}
	}
	//remove InsPlayer tiles
	auto& tiles = HandTiles[InsCheckPlayerID];
	for (auto& tile : tiles)Tiles[tile]--;


	//randomize tiles
	//random hand tile,wall tiles,etc
	StochasticTile(Tiles);


	//check tiles
	for (int i = 1; i < 4; i++) {
		int ID = (InsCheckPlayerID + i) % PlayerNum;
		auto& tiles = HandTiles[ID];
		for (auto& tile : tiles)Tiles[tile]--;
	}
	for (auto& tile : wall_tiles)Tiles[tile]--;
	for (auto& tile : dead_wall_dora)Tiles[tile]--;
	for (int i = 0; i < 34; i++)assert(Tiles[i] == 0);
}
//transform root action mask
void Riichi_Python::ModifyStochasticActionMask(double* Action_Prior,bool*ActionMask) {
	map<int, bool>actions;
	for (int i = 0; i < ActionSpace; i++) {
		if (i % 57 < 48 && ActionMask[i]) {
			//assert(i % 57 < 3);
			ActionMask[i] = false;
			//check all 16 possible transfer
			double mx = -1, idx = -1;
			for (int j = 0; j < 16; j++) {
				int _i = i / 57 * 57 + i % 57 % 3 + j * 3;
				if (Action_Prior[_i] > mx) {
					mx = Action_Prior[_i];
					idx = _i;
				}
			}
			//int type = i % 57 % 3, idx = -1;
			////discard
			//if (type == 0)
			//	idx = i / 57 * 57 + i % 57 % 3 + 12;
			////pass
			//else if (type == 1)
			//	idx = i / 57 * 57 + i % 57 % 3 + 12 * ((InsPlayerID + 1 - InsCheckPlayerID + 4) % 4);
			////shouminkan
			//else idx = i / 57 * 57 + i % 57 % 3;
			assert(idx != -1);
			assert(actions.find(idx) == actions.end());
			actions[idx] = true;
		}
	}
	for (auto k : actions)ActionMask[k.first] = true;
}
//provide env
void Riichi_Python::MaskScreen(double* Scr) {
	int _i = 0;
	_i += (25 + 16 + 2) * 4;
	_i += 14;
	/*for (int i = 0; i < PlayerNum; i++) {
		int ID = (InsCheckPlayerID + i) % PlayerNum;
		auto& tiles = HandTiles[ID];
		for (auto& tile : tiles)Scr[_i++] = tile;
		_i += 14 - tiles.size();
	}*/
}
//only get simple discard action
//inner node
void Riichi_Python::GetInsActionMask(bool* ActionMask) {
	fill(ActionMask, ActionMask + ActionSpace, false);
	assert(InsCheckPlayerID == InsPlayerID);
	//discard after riichi
	if (last_action % 57 == 56) {
		int idx = last_action / 57 * 57 + 12;
		ActionMask[idx] = true;
	}
	//normal discard
	else {
		auto& tiles = HandTiles[InsPlayerID];
		for (auto& tile : tiles) {
			int idx = tile * 57 + 12;
			ActionMask[idx] = true;
		}
	}
	Action_Mask.clear();
	for (int i = 0; i < ActionSpace; i++) {
		if (ActionMask[i])Action_Mask.push_back(getSimAct(i));
	}
}
bool Riichi_Python::draw_tile() {
	//kan,draw rinshan tile
	if (last_action % 57 == 48 || last_action % 57 == 53 || (last_action < 48 && last_action % 57 == 2)) {
		assert(!wall_tiles.empty());
		int tile = wall_tiles.back(); wall_tiles.pop_back();
		rinshan_pop_tiles.push_back(tile);
		int drawn_tile = rinshan_tiles.front(); rinshan_tiles.erase(rinshan_tiles.begin());
		HandTiles[InsPlayerID].push_back(drawn_tile);
		CheckTile = drawn_tile;
	}
	//draw wall tiles
	else {
		if (wall_tiles.empty())return false;
		int tile = wall_tiles.front(); wall_tiles.erase(wall_tiles.begin());
		HandTiles[InsPlayerID].push_back(tile);
		CheckTile = tile;
	}
	return true;
}
void erase_handtile(vector<int>&HandTile,const vector<int>&erase_tiles) {
	for (auto& it : erase_tiles) {
		bool pass = false;
		for (auto tile = HandTile.begin(); tile != HandTile.end();) {
			if (*tile == it) {
				tile = HandTile.erase(tile); pass = true;
				break;
			}
			else tile++;
		}assert(pass);
	}
}
void Riichi_Python::Act(ui Action, double* Reward) {
	int type = Action % (3 * 4 * 4 + 9), tile = Action / (3 * 4 * 4 + 9);
	if (type < 48) {
		//discard
		if (type % 3 == 0) {
			assert(type == 12 && InsCheckPlayerID == InsPlayerID);
			(++InsCheckPlayerID) %= 4;
			(++InsPlayerID) %= 4;
			DesktopTilesDisplay[InsPlayerID].push_back(tile);
			erase_handtile(HandTiles[InsPlayerID], { tile });
		}
		//pass
		else if (type % 3 == 1) {
			assert(type % 12 < 3 && InsCheckPlayerID != InsPlayerID);
			InsCheckPlayerID = (InsPlayerID + type / 12) % 4;
			InsPlayerID = (InsPlayerID + type / 12) % 4;
		}
		//shouminkan
		else {
			assert(type == 2 && InsCheckPlayerID == InsPlayerID);
			assert(CheckTile == tile);
			erase_handtile(HandTiles[InsPlayerID], { CheckTile });
			auto& melds = tileSet[InsPlayerID];
			int cnt = 0;
			for (auto& meld : melds) {
				if (meld.Y == CheckTile) {
					assert(meld.X[0] == CheckTile && meld.X[0] == meld.X[2]);
					meld.X.push_back(CheckTile); cnt++;
				}
			}assert(cnt == 1);
		}
	}
	//closed kan
	else if (type == 48) {
		assert(InsCheckPlayerID == InsPlayerID);
		erase_handtile(HandTiles[InsPlayerID], { tile,tile,tile,tile });
		tileSet[InsPlayerID].push_back({ {tile,tile,tile,tile},-1 });
	}
	//chi
	else if (type < 52) {
		assert(tile == CheckTile && InsCheckPlayerID != InsPlayerID);
		vector<int>meld;
		if (type == 49)
			meld = { tile,tile + 1,tile + 2 };
		else if (type == 50)
			meld = { tile - 1,tile,tile + 1 };
		else meld = { tile - 2,tile - 1,tile };
		tileSet[InsPlayerID].push_back({ meld,CheckTile });
	}
	//pon
	else if (type == 52) {
		assert(tile == CheckTile && InsCheckPlayerID != InsPlayerID);
		erase_handtile(HandTiles[InsPlayerID], { tile,tile });
		tileSet[InsPlayerID].push_back({ {tile,tile,tile},CheckTile });
	}
	//open kan
	else if (type == 53) {
		assert(tile == CheckTile && InsCheckPlayerID != InsPlayerID);
		erase_handtile(HandTiles[InsPlayerID], { tile,tile,tile });
		tileSet[InsPlayerID].push_back({ {tile,tile,tile,tile},CheckTile });
	}
	//win
	else if (type == 54) {

	}
	//draw
	else if (type == 55) {

	}
	//riichi
	else if (type == 56) {
		assert(!IsRiichi[InsPlayerID]);
		IsRiichi[InsPlayerID] = true;
	}



	last_action = Action;
}
bool Riichi_Python::GetGameState(int* result) {
	//win/draw
	if (last_action % 57 == 54 || last_action % 57 == 55)
		return true;
	int type = last_action % 57;
	if (!((49 <= type && type <= 52) || type == 56))
		//exhaustive
		if (!draw_tile())
			return true;


	return false;
}





void Riichi_Python::DynamicEncode(const list<Net*>& InPut_Nets, Param* Data, int idx, Mat* Hidden_State, const ui Action, int fillZero) {
	Data->_DataIn->Reset(1);
	Data->_DataIn->Count = Data->_DataIn->MaxCount;
	Data->_DataOut->Reset(0);
	if (fillZero)return;
	//hidden state
	for (auto& k : InPut_Nets) {
		if (k->Net_Node_Num == getHiddenStateSize()) {
			//rotate
			if (k->GetOutPut().GetRow() == getHiddenStateSize())
				k->GetOutPut().Reset(k->GetOutPut().GetCol(), getHiddenStateSize());
			k->GetOutPut().Append(*Hidden_State, idx);
			break;
		}
	}
	//action
	(*Data->_DataIn)[0] = Action;//SimActS2ActS[Action];
}
void Riichi_Python::RepresentEncode(Param* Data, int fillZero) {
	Data->_DataIn->Reset(Screen_Size);
	Data->_DataIn->Count = Data->_DataIn->MaxCount;
	Data->_DataOut->Reset(0);
	//in order to pass debug
	//fill(Data->_DataIn->data, Data->_DataIn->data + Screen_Size, -1);
	//if (fillZero)return;
	double In[Screen_Size]{ 0 };
	GetGameScreen(In);
	for (int i = 0; i < Screen_Size; i++)
		(*Data->_DataIn)[i] = In[i];
}

void Riichi_Python::RepresentDecode(Agent* agent) {
	Mat& BoardInPut = agent->InPut_Net.front()->GetOutPut();
	assert(agent->InPut_Net.size() == 1);
	assert(agent->All_In[0].GetCol() == agent->Net_Param["Batch"]);
	BoardInPut._ZeroMemory();
	BoardInPut.MahjongRepresentDecode(agent->All_In[0], Image_Depth);
}
void Riichi_Python::DynamicDecode(Agent* agent, int step) {
	Mat& BoardInPut = agent->InPut_Net.front()->Net_Node_Num == getHiddenStateSize() ? agent->InPut_Net.back()->GetOutPut() : agent->InPut_Net.front()->GetOutPut();
	assert(agent->InPut_Net.size() == 2);
	assert(agent->All_In[0].GetCol() == agent->Net_Param["Batch"]);
	for (auto& k : agent->InPut_Net) {
		if (k->Net_Node_Num == getHiddenStateSize()) {
			assert(k->GetOutPut().GetRow() == agent->Net_Param["Batch"]);
			k->GetOutPut() = !k->GetOutPut();
			break;
		}
	}
	BoardInPut._ZeroMemory();
	BoardInPut.Mahjong_Action_Encode(agent->All_In[0], NULL, 0, ActionSpace, Action_Depth);
}

void Riichi_Python::Get_NextState_Reward_And_Policy_Value(const list<Net*>& OutPut_Nets, Param* Data, int idx, Mat* Next_State, double* Reward, double* Policy, double* Value) {
	ui offset = 0; double val[4] = { 0 };
	for (auto& k : OutPut_Nets) {
		//Hidden head
		if (k->CostFunction == MeanSquaredError && k->Net_Node_Num == getHiddenStateSize()) {
			//rotate
			Mat& State = k->Pre_Net_Head.front()->GetOutPut();
			if (State.GetRow() == getHiddenStateSize())State = !State;
			Next_State->Append_(State, idx);
			continue;
		}
		else if (k->CostFunction == CrossEntropy) {
			if (k->Net_Node_Num == ActionSpace)
				for (int i = 0; i < ActionSpace; i++) {
					Policy[i] = (*Data->_DataOut)[i + offset];
				}
			else assert(false);
		}
		else if (k->Net_Flag & Net_Flag_Reward_Net)
			*Reward = (*Data->_DataOut)[offset];
		else {
			for (int i = 0; i < 4; i++)
				val[i] = (*Data->_DataOut)[i + offset];
		}
		offset += k->Net_Node_Num;
	}
	for (int i = 0; i < 4; i++)
		Value[i] = val[i];
}
void Riichi_Python::Get_Initial_State_And_Policy_Value(const list<Net*>& OutPut_Nets, Param* Data, int idx, Mat* Next_State, double* Policy, double* Value) {
	ui offset = 0; double val[4] = { 0 };
	for (auto& k : OutPut_Nets) {
		//Hidden head
		if (k->CostFunction == MeanSquaredError && k->Net_Node_Num == getHiddenStateSize()) {
			//rotate
			Mat& State = k->Pre_Net_Head.front()->GetOutPut();
			if (State.GetRow() == getHiddenStateSize())State = !State;
			Next_State->Append_(State, idx);
			continue;
		}
		else if (k->CostFunction == CrossEntropy) {
			for (int i = 0; i < ActionSpace; i++) {
				Policy[i] = (*Data->_DataOut)[i + offset];
			}
		}
		else {
			for (int i = 0; i < 4; i++)
				val[i] = (*Data->_DataOut)[i + offset];
		}
		offset += k->Net_Node_Num;
	}
	//assign at last
	for (int i = 0; i < 4; i++)
		Value[i] = val[i];
}

Net* Riichi_Python::RepresentationNet(HyperParamSearcher& param) {
	Net* net = (new Net(W * H * Image_Depth, InPut));
	net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, Image_Depth, W);
	net = (new Net(relu))->Add_Forward(BNTransform, net);
	//residual block
	for (int i = 0; i < block_Num; i++) {
		Net* ori_Image = net;
		net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
		net = (new Net(null_Func))->Add_Forward(BNTransform, net);
		Net* Cell = (new Net(net->Net_Node_Num))->Add_Pair_Forward(DotPlus, net, ori_Image);
		net = (new Net(net->Net_Node_Num, relu))->Add_Forward(OpsType::function, Cell);
	}
	net = (new ConvNet(H_filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
	net = (new Net(relu))->Add_Forward(BNTransform, net);
	//Hidden State Head
	(new Net(W * H * H_filters, MeanSquaredError))->Add_Forward(OpsType::OutPut, net);

	Net* _net = net;
	//Policy head
	const int last_filters = 32;
	net = (new ConvNet(last_filters, 1, 1, 0, param, null_Func))->Add_Forward(_net, H_filters, W);
	net = (new Net(relu))->Add_Forward(BNTransform, net);
	net = (new ConvNet(3 * 4 * 4 + 9, 1, 1, 0, param, null_Func, true))->Add_Forward(net, last_filters, W);
	net = (new Net(ActionSpace, softmax))->Add_Forward(OpsType::function, net);
	net = (new Net(ActionSpace, CrossEntropy))->Add_Forward(OpsType::OutPut, net);
	/*net = (new ConvNet(2, 1, 1, 0, param, null_Func))->Add_Forward(_net, H_filters, W);
	net = (new Net(relu))->Add_Forward(BNTransform, net);
	net = (new Net(ActionSpace, softmax))->Add_Forward(Transform, net);
	net = (new Net(ActionSpace, CrossEntropy))->Add_Forward(OpsType::OutPut, net);*/
	//Value head
	net = (new ConvNet(1, 1, 1, 0, param, null_Func))->Add_Forward(_net, H_filters, W);
	net = (new Net(relu))->Add_Forward(BNTransform, net);
	net = (new Net(128, relu))->Add_Forward(Transform, net);
	net = (new Net(4, Base_Net::tanh))->Add_Forward(Transform, net);
	net = (new Net(4, Base_Net::CostFunc::MeanSquaredError))->Add_Forward(OpsType::OutPut, net);

	return net;
}
Net* Riichi_Python::DynamicsNet(HyperParamSearcher& param) {
	Net* actionInPut = (new Net(W * H * Action_Depth, InPut));
	Net* StateInPut = (new Net(W * H * H_filters, InPut));
	StateInPut = (new Net())->Add_Forward(Scale, StateInPut);
	Net* net = (new Net(W * H * (H_filters + Action_Depth), W * H))->Add_Pair_Forward(SpatialConcatenate, StateInPut, actionInPut);

	net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, H_filters + Action_Depth, W);
	net = (new Net(relu))->Add_Forward(BNTransform, net);
	//residual block
	for (int i = 0; i < block_Num; i++) {
		Net* ori_Image = net;
		net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
		net = (new Net(relu))->Add_Forward(BNTransform, net);
		net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(net, filters, W);
		net = (new Net(null_Func))->Add_Forward(BNTransform, net);
		Net* Cell = (new Net(net->Net_Node_Num))->Add_Pair_Forward(DotPlus, net, ori_Image);
		net = (new Net(net->Net_Node_Num, relu))->Add_Forward(OpsType::function, Cell);
	}
	Net* _net = net;
	net = (new ConvNet(H_filters, 3, 1, (3 - 1) / 2, param, null_Func))->Add_Forward(_net, filters, W);
	Net* next_State = net = (new Net(relu))->Add_Forward(BNTransform, net);
	//Hidden State head
	(new Net(W * H * H_filters, MeanSquaredError))->Add_Forward(OpsType::OutPut, net);
	//Reward head
	/*net = (new ConvNet(2, 1, 1, 0, param, null_Func))->Add_Forward(_net, filters, W);
	net = (new Net(relu))->Add_Forward(BNTransform, net);
	net = (new Net(128, relu))->Add_Forward(Transform, net);
	net = (new Net(1, Base_Net::tanh))->Add_Forward(Transform, net);
	net = (new Net(1, Base_Net::CostFunc::MeanSquaredError, Net_Flag_Reward_Net))->Add_Forward(OpsType::OutPut, net);*/

	_net = next_State;
	//Policy head
	const int last_filters = 32;
	net = (new ConvNet(last_filters, 1, 1, 0, param, null_Func))->Add_Forward(_net, H_filters, W);
	net = (new Net(relu))->Add_Forward(BNTransform, net);
	net = (new ConvNet(3 * 4 * 4 + 9, 1, 1, 0, param, null_Func, true))->Add_Forward(net, last_filters, W);
	net = (new Net(ActionSpace, softmax))->Add_Forward(OpsType::function, net);
	net = (new Net(ActionSpace, CrossEntropy))->Add_Forward(OpsType::OutPut, net);
	/*net = (new ConvNet(2, 1, 1, 0, param, null_Func))->Add_Forward(_net, H_filters, W);
	net = (new Net(relu))->Add_Forward(BNTransform, net);
	net = (new Net(ActionSpace, softmax))->Add_Forward(Transform, net);
	net = (new Net(ActionSpace, CrossEntropy))->Add_Forward(OpsType::OutPut, net);*/
	
	//Value head
	net = (new ConvNet(1, 1, 1, 0, param, null_Func))->Add_Forward(_net, H_filters, W);
	net = (new Net(relu))->Add_Forward(BNTransform, net);
	net = (new Net(128, relu))->Add_Forward(Transform, net);
	net = (new Net(4, Base_Net::tanh))->Add_Forward(Transform, net);
	net = (new Net(4, Base_Net::CostFunc::MeanSquaredError))->Add_Forward(OpsType::OutPut, net);
	return net;
}

Net* Riichi_Python::JointNet(HyperParamSearcher& param) {
	//RepresentationNet
	Net* net = (new Net(W * H * Image_Depth, InPut, Net_Flag_RNN_Initial_Step));
	net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_Initial_Step))->Add_Forward(net, Image_Depth, W);
	net = (new Net(relu, Net_Flag_RNN_Initial_Step))->Add_Forward(BNTransform, net);
	//residual block
	for (int i = 0; i < block_Num; i++) {
		Net* ori_Image = net;
		net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_Initial_Step))->Add_Forward(net, filters, W);
		net = (new Net(relu, Net_Flag_RNN_Initial_Step))->Add_Forward(BNTransform, net);
		net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_Initial_Step))->Add_Forward(net, filters, W);
		net = (new Net(null_Func, Net_Flag_RNN_Initial_Step))->Add_Forward(BNTransform, net);
		Net* Cell = (new Net(net->Net_Node_Num, null_Ops, Net_Flag_RNN_Initial_Step))->Add_Pair_Forward(DotPlus, net, ori_Image);
		net = (new Net(net->Net_Node_Num, relu, Net_Flag_RNN_Initial_Step))->Add_Forward(OpsType::function, Cell);
	}
	net = (new ConvNet(H_filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_Initial_Step))->Add_Forward(net, filters, W);
	Net* RepresentState = net = (new Net(relu, Net_Flag_RNN_Initial_Step))->Add_Forward(BNTransform, net);


	//DynamicsNet
	Net* actionInPut = (new Net(W * H * Action_Depth, InPut, Net_Flag_RNN_non_Initial_Step));
	Net* ScaleState = (new Net(0, Scale, Net_Flag_RNN_non_Initial_Step));
	Net* StateAction = net = (new Net(W * H * (H_filters + Action_Depth), W * H, Net_Flag_RNN_non_Initial_Step))->Add_Pair_Forward(SpatialConcatenate, ScaleState, actionInPut);
	net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_non_Initial_Step))->Add_Forward(net, H_filters + Action_Depth, W);
	net = (new Net(relu, Net_Flag_RNN_non_Initial_Step))->Add_Forward(BNTransform, net);
	//residual block
	for (int i = 0; i < block_Num; i++) {
		Net* ori_Image = net;
		net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_non_Initial_Step))->Add_Forward(net, filters, W);
		net = (new Net(relu, Net_Flag_RNN_non_Initial_Step))->Add_Forward(BNTransform, net);
		net = (new ConvNet(filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_non_Initial_Step))->Add_Forward(net, filters, W);
		net = (new Net(null_Func, Net_Flag_RNN_non_Initial_Step))->Add_Forward(BNTransform, net);
		Net* Cell = (new Net(net->Net_Node_Num, null_Ops, Net_Flag_RNN_non_Initial_Step))->Add_Pair_Forward(DotPlus, net, ori_Image);
		net = (new Net(net->Net_Node_Num, relu, Net_Flag_RNN_non_Initial_Step))->Add_Forward(OpsType::function, Cell);
	}
	Net* __net = net;
	//Hidden_State
	net = (new ConvNet(H_filters, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_non_Initial_Step))->Add_Forward(net, filters, W);
	Net* DynamicState = net = (new Net(relu, Net_Flag_RNN_non_Initial_Step))->Add_Forward(BNTransform, net);
	//Reward
	/*net = (new ConvNet(2, 1, 1, 0, param, null_Func, false, Net_Flag_RNN_non_Initial_Step))->Add_Forward(__net, filters, W);
	net = (new Net(relu, Net_Flag_RNN_non_Initial_Step))->Add_Forward(BNTransform, net);
	net = (new Net(128, relu, Net_Flag_RNN_non_Initial_Step))->Add_Forward(Transform, net);
	net = (new Net(1, Base_Net::tanh, Net_Flag_RNN_non_Initial_Step))->Add_Forward(Transform, net);
	net = (new Net(1, Base_Net::CostFunc::MeanSquaredError, Net_Flag_RNN_non_Initial_Step, getMaxUnrolledStep()))->Add_Forward(OpsType::OutPut, net);*/
	//player id head(Start from 0 with first player)
	/*net = (new ConvNet(filters / 2, 3, 1, (3 - 1) / 2, param, null_Func, false, Net_Flag_RNN_non_Initial_Step))->Add_Forward(__net, filters, W);
	net = (new Net(relu, Net_Flag_RNN_non_Initial_Step))->Add_Forward(BNTransform, net);
	net = (new Net(4, softmax, Net_Flag_RNN_non_Initial_Step))->Add_Forward(Transform, net);
	net = (new Net(4, CrossEntropy, Net_Flag_RNN_non_Initial_Step, getMaxUnrolledStep()))->Add_Forward(OpsType::OutPut, net);*/


	Net* StateSwitch = (new Net(W * H * H_filters))->Add_Pair_Forward(RNNInPutSwitch, DynamicState, RepresentState);
	ScaleState->Add_Forward(Scale, StateSwitch, Net_Flag_Reconnect);
	//StateAction->Add_Pair_Forward(SpatialConcatenate, StateSwitch, actionInPut, Net_Flag_Reconnect);


	//prediction
	//large ActionSpace using flat distribution need more train time(trian slower)
	//last filters not fixed
	const int last_filters = 32;
	net = (new ConvNet(last_filters, 1, 1, 0, param, null_Func))->Add_Forward(StateSwitch, H_filters, W);
	net = (new Net(relu))->Add_Forward(BNTransform, net);
	net = (new ConvNet(3 * 4 * 4 + 9, 1, 1, 0, param, null_Func, true))->Add_Forward(net, last_filters, W);
	net = (new Net(ActionSpace, softmax))->Add_Forward(OpsType::function, net);
	net = (new Net(ActionSpace, CrossEntropy, 0, getMaxUnrolledStep()))->Add_Forward(OpsType::OutPut, net);
	//flat distribution(train slower)
	/*net = (new ConvNet(2, 1, 1, 0, param, null_Func))->Add_Forward(StateSwitch, H_filters, W);
	net = (new Net(relu))->Add_Forward(BNTransform, net);
	net = (new Net(ActionSpace, softmax))->Add_Forward(Transform, net);
	net = (new Net(ActionSpace, CrossEntropy, 0, getMaxUnrolledStep()))->Add_Forward(OpsType::OutPut, net);*/
	//Value
	net = (new ConvNet(1, 1, 1, 0, param, null_Func))->Add_Forward(StateSwitch, H_filters, W);
	net = (new Net(relu))->Add_Forward(BNTransform, net);
	net = (new Net(128, relu))->Add_Forward(Transform, net);
	net = (new Net(4, Base_Net::tanh))->Add_Forward(Transform, net);
	net = (new Net(4, Base_Net::CostFunc::MeanSquaredError, 0, getMaxUnrolledStep()))->Add_Forward(OpsType::OutPut, net);
	return net;
}

bool Riichi_Python::Train_In_Out_Process(Agent* agent, Mat* RandomGenerator, int _step, int test_start_col/*, Agent*Server*/) {
	Mat& BoardInPut = (agent->InPut_Net.front()->Net_Flag & Net_Flag_RNN_Initial_Step) ? agent->InPut_Net.front()->GetOutPut() : agent->InPut_Net.back()->GetOutPut();
	Mat& ActionInPut = (agent->InPut_Net.front()->Net_Flag & Net_Flag_RNN_Initial_Step) ? agent->InPut_Net.back()->GetOutPut() : agent->InPut_Net.front()->GetOutPut();

	Mat* BoardOutPut = NULL, * ValueOutPut = NULL, * RewardOutPut = NULL, * PlayerIDOutPut = NULL;
	for (auto& k : agent->OutPut_Net) {
		if (k->Net_Flag & Net_Flag_RNN_non_Initial_Step) {
			//RewardOutPut = &k->GetOutPut();
			//PlayerIDOutPut = &k->GetOutPut();
		}
		else if (k->CostFunction == CostFunc::MeanSquaredError)
			ValueOutPut = &k->GetOutPut();
		else BoardOutPut = &k->GetOutPut();
	}
	Mat In_Sample(agent->All_In[0].GetRow(), agent->Net_Param["Batch"]);
	Mat Out_Sample(agent->All_Out[0].GetRow(), agent->Net_Param["Batch"]);

	Train_Test(In_Sample, agent->All_In[_step], RandomGenerator, test_start_col);
	Train_Test(Out_Sample, agent->All_Out[0], &In_Sample, -1);
	//InPut
	if (_step == 0) {
		BoardInPut._ZeroMemory();
		BoardOutPut->_ZeroMemory();
		ValueOutPut->_ZeroMemory();
		//PlayerIDOutPut->_ZeroMemory();

		BoardInPut.MahjongRepresentDecode(Out_Sample, Image_Depth);
		BoardOutPut->Mahjong_Policy_Encode(Out_Sample, Screen_Size + 2);

		/*auto*M = Out_Sample.ReadFromDevice();
		for (int i = 0; i < 50; i++)printf("%lf ", M[(i+ Screen_Size + 2) * 128 + 0]);

		M = BoardOutPut->ReadFromDevice();
		for (int i = 0; i < 50; i++)printf("%lf ", M[i * 128 + 0]);*/

		ValueOutPut->Append_(In_Sample, 1);
		//this maybe better
		//*ValueOutPut = *ValueOutPut*1.5;
		
		//ValueOutPut->f(Nor, 1.0);
		/*auto*M = ValueOutPut->ReadFromDevice();
		for (int i = 0; i < 4; i++)printf("%lf ", M[i * 128 + 0]);*/

		//ValueOutPut->Mahjong_Values_Encode();
		//ValueOutPut->Mahjong_Reward_Sample();


		/*M = ValueOutPut->ReadFromDevice();
		for (int i = 0; i < 4; i++)printf("%lf ", M[i * 128 + 0]);*/

		//Mat value(ValueOutPut->GetRow(), ValueOutPut->GetCol()), tmp(1, ValueOutPut->GetCol());
		//value.Append_(In_Sample, 1);
		////auto* M = value.ReadFromDevice();
		////for (int i = 0; i < 4; i++)printf("%lf ", M[i * 128 + 0]);
		//value.MinMax_Normalization(*ValueOutPut, tmp);
		//*ValueOutPut = (*ValueOutPut - 0.5) * 2;
		//M = ValueOutPut->ReadFromDevice();
		//for (int i = 0; i < 4; i++)printf("%lf ", M[i * 128 + 0]);
	}
	else {
		ActionInPut._ZeroMemory();
		BoardOutPut->_ZeroMemory();
		//ValueOutPut->_ZeroMemory();
		//PlayerIDOutPut->_ZeroMemory();

		ActionInPut.Mahjong_Action_Encode(Out_Sample, *ValueOutPut, Screen_Size, ActionSpace, Action_Depth);
		BoardOutPut->Mahjong_Policy_Encode(Out_Sample, Screen_Size + 2);

		/*printf("\n");
		auto*M = Out_Sample.ReadFromDevice();
		for (int i = 0; i < 1; i++)printf("%lf ", M[(i + Screen_Size) * 128 + 0]);
		printf("\n");
		M = ValueOutPut->ReadFromDevice();
		for (int i = 0; i < 4; i++)printf("%lf ", M[i * 128 + 0]);*/

		//ValueOutPut->Append_(In_Sample, 1);
		//*ValueOutPut = *ValueOutPut * 1.5;
		//*ValueOutPut = *ValueOutPut * 2.0;
		//ValueOutPut->f(Thresholding, 1.0);
		//ValueOutPut->Mahjong_Values_Encode();

		//PlayerIDOutPut->Mahjong_PlayerID_Decode(In_Sample);
		/*auto* M = In_Sample.ReadFromDevice();
		for (int i = 0; i < 6; i++)printf("%lf ", M[i * 128 + 0]);
		M = PlayerIDOutPut->ReadFromDevice();
		for (int i = 0; i < 4; i++)printf("%lf ", M[i * 128 + 0]);*/
		//ValueOutPut->Mahjong_Values_Encode(Out_Sample, Screen_Size + 1, In_Sample);
		//Mat value(ValueOutPut->GetRow(), ValueOutPut->GetCol()), tmp(1, ValueOutPut->GetCol());
		//value.Append_(In_Sample, 1);
		////auto* M = value.ReadFromDevice();
		////for (int i = 0; i < 4; i++)printf("%lf ", M[i * 128 + 0]);
		//value.MinMax_Normalization(*ValueOutPut, tmp);
		//*ValueOutPut = (*ValueOutPut - 0.5) * 2;
		/*auto* M = ValueOutPut->ReadFromDevice();
		for (int i = 0; i < 4; i++)printf("%lf ", M[i * 128 + 0]);*/
		//RewardOutPut->Append_(Out_Sample, Screen_Size + 2);
	}
	return true;
}