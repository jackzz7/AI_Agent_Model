#pragma once
#ifndef CUDA_ACC_H
#define CUDA_ACC_H

#ifdef CUDA_ACC_EXPORTS
#define CUDA_ACC __declspec(dllexport)
#else
#define CUDA_ACC __declspec(dllimport)

#endif


#ifdef _WIN64
#define __GPU__
#endif

#include<iostream>
#include<assert.h>
#include<thread>
#include<list>
#include<map>

#define DEBUG(exp,str)if(exp)printf(str),assert(false)
//CUDA version difference ignore
//using cmd
//git config --global filter.gitignore.clean "sed '/<!--#gitignoreBEGIN-->/,/<!--#gitignoreEND-->$/d'"
//git config --global filter.gitignore.smudge cat
#define floatType float
const int float_Max_Int = 1 << 24;

namespace _CUDA_ {

	using namespace std;

	typedef unsigned int ui;
	typedef floatType(*_fun)(const floatType&);
	typedef floatType(*_fun2)(const floatType&,const floatType&);

	enum Device_Func {
		Null_Func, ReLU, D_ReLU, Ln, Pow2, Sigmoid, D_Sigmoid, Bool, Tanh, D_Tanh, Sqrt, HardTanh, D_HardTanh, Leaky_HardTanh, D_Leaky_HardTanh, Pow3,Nor,
		DropOut_Bernoulli, Assignment, Compare, Thresholding, Uniform, Division, number_div
	};


	const int Cuda_Max_Stream = 30;
#define get_stm_id stmID.find(this_thread::get_id())!=stmID.end()?stmID[this_thread::get_id()]:BindStm()
#define get_stm stms[get_stm_id]
	CUDA_ACC extern map<thread::id, int>stmID;
	CUDA_ACC int BindStm();
	CUDA_ACC void unBindStm();

	template<class T>  class Matrix
	{
	public:
		const static size_t sz = sizeof(T);

		//rows,cols
		ui row, col;
		//pointer to data
		T*M;
		//copy of M,used for Cuda
		//pointer to data
		T*Cuda_M;
		ui WidthStride;
		//Cuda ACC
		size_t pitch;
		
		Matrix() {
			Clear();
		}
		Matrix(ui rows, ui cols, bool CPU_Memory = false) :row(rows), col(cols) {
			assert(row > 0 && col > 0);
			if (CPU_Memory)M = new T[row*col];
			else M = NULL;
			Cuda_M = NULL;
			WidthStride = sz*col;
			pitch = 0;
		}
		Matrix(ui rows, ui cols,const T&value):Matrix(rows,cols,true) {
			std::fill(M, M + row * col, value);
		}
		Matrix(ui rows, ui cols, T*Data, bool CPU_Memory = true) : Matrix(rows, cols, CPU_Memory) {
			if (CPU_Memory) {
				size_t _sz = row * col * sz;
				memcpy_s(M, _sz, Data, _sz);
			}
			else M = Data;
		}
	    Matrix(ui rows, ui cols, T**Data) : Matrix(rows, cols,true) {
			//size_t sz = col * sizeof(T);
			for (int i = 0; i < row; i++) {
				memcpy_s(&M[i*col], WidthStride, Data[i], WidthStride);
			}
		}
		Matrix(ui rows, ui cols, ui Start_col, T**Data) : Matrix(rows, cols,true) {
			for (int i = 0; i < row; i++) {
				memcpy_s(&M[i*col], WidthStride, &Data[i][Start_col], WidthStride);
			}
		}
		//move constructor
		//inline Matrix(const Matrix<T>&&R) {
		//	(*this) = R;
		//};
		/*inline Matrix(const Matrix<T>&R) {
			memcpy(this, &R, sizeof(Matrix<T>));
		};*/


		~Matrix();
		inline void Disponse();
		inline void FreeCuda_M();
		inline void Clear() {
			//fixed,not safe
			memset(this, 0, sizeof(Matrix<T>));
		}

		inline T* alloc_Cuda_M();


		T* WriteToCuda(int write_row = -1);
		T* ReadFromCuda(bool pinned = false);

		typedef void(*Kernel)(int Out_pitch, int A_pitch, int B_pitch, floatType*A, floatType*B, floatType*Out, ui Out_col, ui A_row, ui A_col, ui B_row, ui B_col);
		inline Matrix<T>* Matrix_row_col_expand(Kernel MatrixKernel, Matrix<T>*R, bool TemporaryVariable);

		inline Matrix<T>* Cuda_ACC_Basic_Cal(int Cal_Ops,Matrix<T>*R, bool TemporaryVariable = true);
		inline Matrix<T>* Cuda_ACC_mul(const Matrix<T>*R);
		//inline Matrix<T>* Cuda_ACC_plus_(Matrix<T>*R, const int& sign = 1);
		//inline Matrix<T>* Cuda_ACC_plus_minus(Matrix<T>*R, const int& sign, bool TemporaryVariable = true);
		inline Matrix<T>* Cuda_ACC_minus(Matrix<T>*R);
		inline const Matrix<T>* Cuda_ACC_minus_(Matrix<T>*R);
		inline Matrix<T>* Cuda_ACC_number(int Ops, const T&number, bool lnumber = false);
		inline Matrix<T>* Cuda_ACC_assign(const Matrix<T>*R, int Copy_Row_Num, int dst_Start_row = 0, int src_Start_col = 0, int src_Start_row = 0);
		inline Matrix<T>* Cuda_ACC_assign_(const Matrix<T>*R, int Copy_Row_Num, int dst_Start_col);
		inline Matrix<T>* Cuda_ACC_function(int fun_id, bool TemporaryVariable = false,const T& param = -1,int Start_Row=0);
		inline Matrix<T>* Cuda_ACC_ScaleOneCol(bool OneRow = false);
		inline Matrix<T>* Cuda_ACC_T();
		inline Matrix<T>* Cuda_ACC_Sum();
		inline const Matrix<T>* Cuda_ACC_ZeroMemory();
		inline const Matrix<T>* Cuda_ACC_SoftMaxFunction(bool Max = false, Matrix<T>*One_Col = NULL);
		inline const Matrix<T>* Cuda_ACC_ResetRow(int new_row){
			row = new_row;
			return this;
		}
		inline const Matrix<T>* Cuda_ACC_ResetCol(int new_col) {
			col = new_col;
			WidthStride = sz*col;
			return this;
		}
		inline const Matrix<T>* Order_Assign_Val(Matrix<T>*Order,T val);
		inline const Matrix<T>* Cuda_ACC_RandMatrixs(Matrix<T>*R,Matrix<T>*Order,int Start_row, bool Right_Mat_random_Order,ui Max_Order);
		inline const Matrix<T>* RandomPositiveSample(Matrix<T>*Word_Context_Possibility, Matrix<T>*Word_Context_Context_Idx, Matrix<T>*Word_Context_Range, Matrix<T>*Negative_Sampling_Generator);
		inline const Matrix<T>* GenerateNegaOrder(Matrix<T>*Context,Matrix<T>*Context_Range, Matrix<T>*Order,Matrix<T>*Unigram_Table);
		inline const Matrix<T>* GenerateOutPutMask(Matrix<T>*Negative_Sampling_Order, Matrix<T>*Order, Matrix<T>*Context_Idx, Matrix<T>*Context_Range, Matrix<T>*OutPut_Mask, Matrix<T>*OutPut_Mask1);
		inline const Matrix<T>* Cuda_ACC_generate_random_number(ui Mod_Value);
		inline const Matrix<T>* Normal_Sampling();
		inline Matrix<T>* ScaleImage(int Image_W, int Image_Depth, Matrix<T>*Location, int Scale_Image_WH, int Scale_Num);
		inline const Matrix<T>* OneHot_Sampling();

		inline const Matrix<T>* GomokuSimulation(Matrix<T>*Moves, Matrix<T>*BoradID, bool InPut, Matrix<T>*Reward,const Matrix<T>*Random,Matrix<T>*Value);
		inline const Matrix<T>* GomokuSimulation_Extend(Matrix<T>*Board, int Planes);
		inline const Matrix<T>* Go_Action_Encode(Matrix<T>*Out_Sample, int Start_Row, ui ActionSpace);
		inline const Matrix<T>* ChessRepresentDecode(Matrix<T>*Board, int Planes);
		inline const Matrix<T>* Chess_Action_Encode(Matrix<T>* Out_Sample, int Start_Row, ui ActionSpace, Matrix<T>* ActionMap, int W, int Plane, bool rotateAction);
		inline const Matrix<T>* Chess_Policy_Encode(Matrix<T>*Out_Sample, int Start_Row, Matrix<T>*ActionMap, bool rotateAction);

		//Convolutional Net Ops
		inline Matrix<T>* Conv_im2col(int Image_Depth, int Image_W, int W, int H, int Receptive,int Padding,int Stride, Matrix<T>*bp);
		inline const Matrix<T>* Conv_im2col_Restore(int Image_W, Matrix<T>*Restore_Image,bool bp);
		inline Matrix<T>* Image_Pooling(int Image_Depth, int Image_W, int W, int H, int Receptive, int Padding, int Stride,Matrix<T>*Pool_idx, Matrix<T>*bp);
		
		inline Matrix<T>* Image_SpatialConcatenate(Matrix<T>*first, Matrix<T>*second, int size, bool bp = false, bool first_Order = true);

		//-0.1,0.1
		void RandData(floatType factor = 1.0f) {
			if (M == NULL)M = new T[row * col];
			for (int i = 0; i < row; i++)
				for (int j = 0; j < col; j++)
					M[i * col + j] = (0.2f * rand() / RAND_MAX - 0.1f) * factor;
			WriteToCuda();
		}
		bool operator==(T*M) {
			for (int i = 0; i < row; i++)
				for (int j = 0; j < col; j++) {
					if (abs(this->M[i*col + j] - M[i*col + j]) > 1e-8)return false;
				}
			return true;
		}
		void ConsolePrint() {
			ReadFromCuda();
			for (int i = 0; i < row; i++) {
				for (int j = 0; j < col; j++)
					cout << M[i*col + j] << ' ';
				cout << endl;
			}cout << endl;
		}
	};
	template<class T> ostream& operator<<(ostream&os, Matrix<T>*a) {
		a->ConsolePrint();
		return os;
	}
	template CUDA_ACC std::ostream& operator<<(std::ostream&os, Matrix<floatType>*a);


	//Inititalize func variables
	CUDA_ACC void Matrix_Set_Function(void**d_f, void**host_f,int MaxRandomStates_rows);
	CUDA_ACC void StreamSynchronize();

	template<class T>
	struct Matrix_Heap {
		using _pi = std::pair<int, int>;
		typedef map<_pi, list<Matrix<T>*>> HEAP;
		HEAP Heap;
		Matrix_Heap() {
			clear();
		}
		~Matrix_Heap() {
			clear();
		}
		Matrix<T>*malloc(const int&row,const int&col) {
			if (Heap.find({ row,col }) != Heap.end()) {
				auto&ls = Heap[{ row, col }];
				auto _new = ls.front(); ls.pop_front();
				if (ls.size() == 0)Heap.erase({ row,col });
				return _new;
			}
			else return new Matrix<T>(row, col);
		}
		void free(Matrix<T>*&del) {
			if (del&&del->col > 0 && del->row > 0) {
				Heap[{del->row, del->col}].push_back(del);
			}
			else delete del;
			del = NULL;
		}
		void clear() {
			for (auto&m : Heap) {
				for (auto&e : m.second)delete e;
			}
			Heap.clear();
		}
	};
	extern Matrix_Heap<floatType> Heap[];

	template<class T> class MatrixPtr {
	private:
		Matrix<T>*ptr;
	public:
		inline MatrixPtr() {
			ptr = NULL;
		}
		inline MatrixPtr(Matrix<T>*Matrix) {
			ptr = Matrix;
		}
		inline MatrixPtr(ui row, ui col, bool WriteToDevice = true) :MatrixPtr() {
			Reset(row, col, WriteToDevice);
		}
		inline MatrixPtr(ui row, ui col, T*Data, bool WriteToDevice = true) : MatrixPtr() {
			Reset(row, col, Data, WriteToDevice);
		}
		inline MatrixPtr(ui row, ui col, T**Data, bool WriteToDevice = true) : MatrixPtr() {
			Reset(row, col, Data, WriteToDevice);
		}
		inline MatrixPtr(ui row, ui col, ui Start_col, T**Data, bool WriteToDevice = true) : MatrixPtr() {
			Reset(row, col, Start_col, Data, WriteToDevice);
		}
		inline MatrixPtr(ui row, ui col, const T&value, bool WriteToDevice = true) : MatrixPtr() {
			Reset(row, col, value, WriteToDevice);
		}
		void Reset(ui row, ui col, bool WriteToDevice = true, bool pinned = false, bool auto_alloc_free = true);
		void Reset(ui row, ui col, const T&value, bool WriteToDevice = true, bool CPU_Memory = true) {
			if (ptr == NULL || ptr->row != row || ptr->col != col) {
				Heap[get_stm_id].free(ptr); ptr = Heap[get_stm_id].malloc(row, col);
				//if (ptr)delete ptr;ptr = new Matrix<T>(row, col, value);
			}
			if (!ptr->M)ptr->M = new T[row*col];
			std::fill(ptr->M, ptr->M + row * col, value);
			if (WriteToDevice)
				this->WriteToDevice();
			if (!CPU_Memory)delete[]ptr->M, ptr->M = NULL;
		}
		void Reset(ui row, ui col, T*Data, bool WriteToDevice = true, bool CPU_Memory = true, bool auto_alloc_free = true) {
			if (ptr == NULL || ptr->row != row || ptr->col != col) {
				if (auto_alloc_free)
					Heap[get_stm_id].free(ptr), ptr = Heap[get_stm_id].malloc(row, col);
				else {
					delete ptr; ptr = new Matrix<T>(row, col);
				}
			}
			if (CPU_Memory) {
				size_t _sz = row * col * Matrix<T>::sz;
				if (!ptr->M)ptr->M = new T[row*col];
				memcpy_s(ptr->M, _sz, Data, _sz);
			}
			else ptr->M = Data;
			if (WriteToDevice)
				this->WriteToDevice();
			if (!CPU_Memory)ptr->M = NULL;
		}
		void Reset(ui row, ui col, T**Data, bool WriteToDevice = true) {
			if (ptr == NULL || ptr->row != row || ptr->col != col) {
				Heap[get_stm_id].free(ptr); ptr = Heap[get_stm_id].malloc(row, col);
			}
			if (!ptr->M)ptr->M = new T[row*col];
			for (int i = 0; i < row; i++) {
				memcpy_s(&ptr->M[i*col], ptr->WidthStride, Data[i], ptr->WidthStride);
			}
			if (WriteToDevice)
				this->WriteToDevice();
		}
		void Reset(ui row, ui col, ui Start_col, T**Data, bool WriteToDevice = true) {
			if (ptr == NULL || ptr->row != row || ptr->col != col) {
				Heap[get_stm_id].free(ptr); ptr = Heap[get_stm_id].malloc(row, col);
			}
			if (!ptr->M)ptr->M = new T[row*col];
			for (int i = 0; i < row; i++) {
				memcpy_s(&ptr->M[i*col], ptr->WidthStride, &Data[i][Start_col], ptr->WidthStride);
			}
			if (WriteToDevice)
				this->WriteToDevice();
		}
		inline ~MatrixPtr() {
			int stm_id = stmID.find(this_thread::get_id()) != stmID.end() ? stmID[this_thread::get_id()] : -1;
			if (stm_id != -1)Heap[stm_id].free(ptr);
			else {
				Disponse();
			}
		}
		inline void Disponse() {
			delete ptr; ptr = NULL;
		}
		inline void Clear() {
			if (ptr) {
				ptr->Disponse();
				ptr->WidthStride = Matrix<T>::sz * ptr->col;
			}
		}
		inline bool IsValid() const {
			return ptr != NULL;
		}
		inline Matrix<T>*& getMatrix(){
			return ptr;
		}
		inline Matrix<T>* getMatrix()const {
			return ptr;
		}
		inline MatrixPtr<T>& _ZeroMemory(){
			ptr->Cuda_ACC_ZeroMemory();
			return *this;
		}
		//check Mat Valid
		inline MatrixPtr<T>& _ZeroMemory_Valid(ui row, ui col) {
			Reset(row, col, false);
			ptr->Cuda_ACC_ZeroMemory();
			return *this;
		}
		//Matrix multiplication
		inline const MatrixPtr<T> operator%(const MatrixPtr<T>&Right)const {
			return ptr->Cuda_ACC_mul(Right.getMatrix());
		}
		inline const MatrixPtr<T> operator*(const MatrixPtr<T>&Right)const {
			return ptr->Cuda_ACC_Basic_Cal(2, Right.getMatrix());
		}
		inline const MatrixPtr<T> operator+(const MatrixPtr<T>&Right)const {
			return ptr->Cuda_ACC_Basic_Cal(0, Right.getMatrix());
		}
		inline const MatrixPtr<T> operator-(const MatrixPtr<T>&Right)const {
			return ptr->Cuda_ACC_Basic_Cal(1, Right.getMatrix());
		}
		inline const MatrixPtr<T> operator/(const MatrixPtr<T>&Right)const {
			return ptr->Cuda_ACC_Basic_Cal(3, Right.getMatrix());
		}
		inline const MatrixPtr<T>& operator/=(const MatrixPtr<T>&Right)const {
			ptr->Cuda_ACC_Basic_Cal(3, Right.getMatrix(), false);
			return *this;
		}
		inline const MatrixPtr<T>& operator*=(const MatrixPtr<T>&Right)const {
			ptr->Cuda_ACC_Basic_Cal(2, Right.getMatrix(), false);
			return *this;
		}
		inline const MatrixPtr<T>& operator-=(const MatrixPtr<T>&Right)const {
			ptr->Cuda_ACC_Basic_Cal(1, Right.getMatrix(), false);
			return *this;
		}
		inline const MatrixPtr<T>& operator+=(const MatrixPtr<T>&Right)const {
			ptr->Cuda_ACC_Basic_Cal(0, Right.getMatrix(), false);
			return *this;
		}

		//number OP Mat
		inline const MatrixPtr<T> operator*(const T&number)const {
			return ptr->Cuda_ACC_number(2, number);
		}
		inline const MatrixPtr<T> operator/(const T&number)const {
			return ptr->Cuda_ACC_number(3, number);
		}
		inline const MatrixPtr<T> operator+(const T&number)const {
			return ptr->Cuda_ACC_number(0, number);
		}
		inline const MatrixPtr<T> operator-(const T&number)const {
			return ptr->Cuda_ACC_number(1, number);
		}
		CUDA_ACC friend inline const MatrixPtr<T> operator*(const T&number,const MatrixPtr<T>&Right) {
			return Right.getMatrix()->Cuda_ACC_number(2, number, true);
		}
		CUDA_ACC friend inline const MatrixPtr<T> operator/(const T&number,const MatrixPtr<T>&Right) {
			return Right.getMatrix()->Cuda_ACC_number(3, number, true);
		}
		CUDA_ACC friend inline const MatrixPtr<T> operator+(const T&number,const MatrixPtr<T>&Right) {
			return Right.getMatrix()->Cuda_ACC_number(0, number, true);
		}
		CUDA_ACC friend inline const MatrixPtr<T> operator-(const T&number,const MatrixPtr<T>&Right) {
			return Right.getMatrix()->Cuda_ACC_number(1, number, true);
		}

		inline const MatrixPtr<T>& Order_Assign_Val(const MatrixPtr<T>&Order,T val)const {
			ptr->Order_Assign_Val(Order.getMatrix(), val);
			return *this;
		}
		inline const MatrixPtr<T>& RandMatrix(const MatrixPtr<T>&Right, const MatrixPtr<T>&Order, int Start_Row = 0,bool Right_Mat_random_Order=true,ui Max_Order=0)const {
			ptr->Cuda_ACC_RandMatrixs(Right.getMatrix(), Order.getMatrix(), Start_Row, Right_Mat_random_Order, Max_Order);
			return *this;
		}
		inline MatrixPtr<T>& RandomPositiveSample(const MatrixPtr<T>&Word_Context_Possibility, const MatrixPtr<T>&Word_Context_Context_Idx, const MatrixPtr<T>&Word_Context_Range, const MatrixPtr<T>&Negative_Sampling_Generator){
			ptr->RandomPositiveSample(Word_Context_Possibility.getMatrix(), Word_Context_Context_Idx.getMatrix(), Word_Context_Range.getMatrix(), Negative_Sampling_Generator.getMatrix());
			return *this;
		}
		inline MatrixPtr<T>& GenerateNegaOrder(const MatrixPtr<T>&Context, const MatrixPtr<T>&Context_Range, const MatrixPtr<T>&Order, const MatrixPtr<T>&Unigram_Table) {
			ptr->GenerateNegaOrder(Context.getMatrix(), Context_Range.getMatrix(), Order.getMatrix(), Unigram_Table.getMatrix());
			return *this;
		}
		inline MatrixPtr<T>& GenerateOutPutMask(const MatrixPtr<T>&Negative_Sampling_Order, const MatrixPtr<T>&Order, const MatrixPtr<T>&Context_Idx,const MatrixPtr<T>&Context_Range, const MatrixPtr<T>&OutPut_Mask,const MatrixPtr<T>&OutPut_Mask1) {
			ptr->GenerateOutPutMask(Negative_Sampling_Order.getMatrix(), Order.getMatrix(), Context_Idx.getMatrix(),Context_Range.getMatrix(), OutPut_Mask.getMatrix(),OutPut_Mask1.getMatrix());
			return *this;
		}
		inline MatrixPtr<T>& GenerateRandom(ui Mod=0xffffffff) {
			ptr->Cuda_ACC_generate_random_number(Mod);
			return *this;
		}
		inline MatrixPtr<T>& Normal_Sampling() {
			ptr->Normal_Sampling();
			return *this;
		}
		inline MatrixPtr<T>& OneHot_Sampling() {
			ptr->OneHot_Sampling();
			return *this;
		}
		inline constexpr Matrix<T>* getMatrix(const MatrixPtr<T>*ptr)const {
			return ptr ? ptr->getMatrix() : NULL;
		}
		inline MatrixPtr<T>& GomokuSimulation(const MatrixPtr<T>*Moves, const MatrixPtr<T>&Board_ID, bool InPut = true, MatrixPtr<T>*Reward = NULL, const MatrixPtr<T>*Random = NULL, MatrixPtr<T>*Value = NULL) {
			ptr->GomokuSimulation(getMatrix(Moves), Board_ID.getMatrix(), InPut, getMatrix(Reward), getMatrix(Random), getMatrix(Value));
			return *this;
		}
		inline MatrixPtr<T>& GomokuSimulation_Extend(const MatrixPtr<T>&Board, int Planes) {
			ptr->GomokuSimulation_Extend(Board.getMatrix(), Planes);
			return *this;
		}
		inline MatrixPtr<T>& ChessRepresentDecode(const MatrixPtr<T>&Board, int Planes) {
			ptr->ChessRepresentDecode(Board.getMatrix(), Planes);
			return *this;
		}
		inline const MatrixPtr<T>& Go_Action_Encode(const MatrixPtr<T>&Out_Sample,int Start_Row,ui ActionSpace) const {
			ptr->Go_Action_Encode(Out_Sample.getMatrix(), Start_Row, ActionSpace);
			return *this;
		}
		inline const MatrixPtr<T>& Chess_Action_Encode(const MatrixPtr<T>&Out_Sample, int Start_Row, ui ActionSpace, const MatrixPtr<T>&ActionMap,int W,int Plane, bool rotateAction) const {
			ptr->Chess_Action_Encode(Out_Sample.getMatrix(), Start_Row, ActionSpace, ActionMap.getMatrix(), W, Plane, rotateAction);
			return *this;
		}
		inline const MatrixPtr<T>& Mahjong_Action_Encode(const MatrixPtr<T>& Out_Sample, const MatrixPtr<T>& value, int Start_Row, ui ActionSpace, int Plane);
		inline const MatrixPtr<T>& Chess_Policy_Encode(const MatrixPtr<T>&Out_Sample, int Start_Row, const MatrixPtr<T>&ActionMap, bool rotateAction) const {
			ptr->Chess_Policy_Encode(Out_Sample.getMatrix(), Start_Row, ActionMap.getMatrix(), rotateAction);
			return *this;
		}
		inline MatrixPtr<T>& Mahjong_Policy_Encode(const MatrixPtr<T>& Out_Sample, int Start_Row);
		inline MatrixPtr<T>& Mahjong_Values_Encode();
		inline MatrixPtr<T>& MahjongRepresentDecode(const MatrixPtr<T>& Board, int Planes);
		inline MatrixPtr<T>& Mahjong_Reward_RepresentDecode(const MatrixPtr<T>& Board, int Planes);
		inline MatrixPtr<T>& Mahjong_Simplify_Policy_Encode(const MatrixPtr<T>& Out_Sample, int Start_Row);
		inline MatrixPtr<T>& Mahjong_Reward_softmax_Encode(int idx,const MatrixPtr<T>& final_reward);
		inline MatrixPtr<T>& Mahjong_Reward_Sample();

		inline MatrixPtr<T>& MahjongAgentRepresentDecode(const MatrixPtr<T>& Board, int Planes);


		inline const MatrixPtr<T> ScaleImage(int Image_W,int Image_Depth, const MatrixPtr<T>&Location, int Scale_Image_WH, int Scale_Num)const {
			return ptr->ScaleImage(Image_W, Image_Depth, Location.getMatrix(), Scale_Image_WH, Scale_Num);
		}
		inline const MatrixPtr<T> Conv_im2col(int Image_Depth, int Image_W, int W, int H, int Receptive, int Padding, int Stride, const MatrixPtr<T>*Gradient=NULL)const {
			return ptr->Conv_im2col(Image_Depth, Image_W, W, H, Receptive, Padding, Stride, Gradient ? Gradient->getMatrix() : NULL);
		}
		inline const MatrixPtr<T>& Conv_Image_Restore(int Image_W,MatrixPtr<T>&Restore_Image,bool bp=false)const {
			ptr->Conv_im2col_Restore(Image_W, Restore_Image.getMatrix(),bp);
			return *this;
		}
		inline const MatrixPtr<T> Image_Pooling(int Image_Depth, int Image_W, int W, int H, int Receptive, int Padding, int Stride,MatrixPtr<T>&Pool_idx, const MatrixPtr<T>*Gradient = NULL)const {
			return ptr->Image_Pooling(Image_Depth, Image_W, W, H, Receptive, Padding, Stride,Pool_idx.getMatrix(), Gradient ? Gradient->getMatrix() : NULL);
		}
		inline const MatrixPtr<T>& Image_SpatialConcatenate(MatrixPtr<T>&first, MatrixPtr<T>&second, int WH_Size, bool bp = false, bool first_Order = true) const {
			ptr->Image_SpatialConcatenate(first.getMatrix(), second.getMatrix(), WH_Size, bp, first_Order);
			return *this;
		}
		inline const MatrixPtr<T> BN_Normalization(MatrixPtr<T>&mean, MatrixPtr<T>&var)const;
		inline const MatrixPtr<T> operator!()const {
			return ptr->Cuda_ACC_T();
		}
		inline const MatrixPtr<T>& f(int fun_id, floatType param = -1, int Start_Row = 0)const {
			ptr->Cuda_ACC_function(fun_id, false, param, Start_Row);
			return *this;
		}
		//Create Temporary Variable
		inline const MatrixPtr<T> _f(int fun_id, floatType param = -1, int Start_Row = 0)const {
			return ptr->Cuda_ACC_function(fun_id, true, param, Start_Row);
		}
		inline const MatrixPtr<T> Sum()const {
			return ptr->Cuda_ACC_Sum();
		}
		inline const MatrixPtr<T>& SoftMax(bool Max = false, MatrixPtr<T>*Scale_One_Col = NULL)const {
			if (!Scale_One_Col)
				ptr->Cuda_ACC_SoftMaxFunction(Max);
			else ptr->Cuda_ACC_SoftMaxFunction(Max, Scale_One_Col->getMatrix());
			return *this;
		}
		inline const MatrixPtr<T>& ResetRows(int New_Row)const {
			ptr->Cuda_ACC_ResetRow(New_Row);
			return *this;
		}
		inline const MatrixPtr<T>& ResetCols(int New_Col)const {
			ptr->Cuda_ACC_ResetCol(New_Col);
			return *this;
		}
		inline const MatrixPtr<T> ScaleOneCol(bool OneRow = false)const {
			return ptr->Cuda_ACC_ScaleOneCol(OneRow);
		}
		inline void MinMax_Normalization(MatrixPtr<T>& result, MatrixPtr<T>& factor)const;

		//only for temporary variables
		inline MatrixPtr(const MatrixPtr<T>&Right) :MatrixPtr() {
			(*this) = Right;
		}
		inline MatrixPtr(MatrixPtr<T>&Right) : MatrixPtr() {
			(*this) = Right;
		}
		inline MatrixPtr<T>& operator=(const MatrixPtr<T>&Right) {
			swap(ptr, ((MatrixPtr<T>*)(void*)&Right)->getMatrix());
			return *this;
		}
		inline MatrixPtr<T>& operator=(MatrixPtr<T>&&Right) {
			swap(ptr, Right.getMatrix());
			return *this;
		}
		inline MatrixPtr<T>& operator=(MatrixPtr<T>&Right) {
			assert(Right.getMatrix());
			if (ptr == NULL)ptr = Heap[get_stm_id].malloc(Right.getMatrix()->row, Right.getMatrix()->col);//new Matrix<T>(Right.getMatrix()->row, Right.getMatrix()->col);
			assert(ptr->row == Right.getMatrix()->row&&ptr->col == Right.getMatrix()->col);
			ptr->Cuda_ACC_assign(Right.getMatrix(), Right.GetRow());
			return *this;
		}
		inline MatrixPtr<T>& Append(const MatrixPtr<T>&Right, int Start_Row, int R_Start_Col = 0) {
			ptr->Cuda_ACC_assign(Right.getMatrix(), Right.GetRow(), Start_Row, R_Start_Col);
			return *this;
		}
		inline MatrixPtr<T>& Append_(const MatrixPtr<T>&Right, int R_Start_Row){
			ptr->Cuda_ACC_assign(Right.getMatrix(), GetRow(), 0, 0, R_Start_Row);
			return *this;
		}
		inline MatrixPtr<T>& Append__(const MatrixPtr<T>&Right, int Start_col) {
			ptr->Cuda_ACC_assign_(Right.getMatrix(), Right.GetRow(), Start_col);  
			return *this;
		}
		inline T& operator[](ui idx)const {
			if (ptr->M == NULL)ptr->M = new T[ptr->row*ptr->col]{ 0 };
			assert(idx < ptr->row*ptr->col);
			//DEBUG(ptr->M == NULL, "error:Not have CPU memory\n");
			return ptr->M[idx];
		}
		inline constexpr ui GetRow()const { return ptr->row; };
		inline constexpr ui GetCol()const { return ptr->col; };
		//copy Cuda_M value
		//inline MatrixPtr<T>& operator=(const MatrixPtr<T>&Right);
		//copy ptr pointer
		//inline MatrixPtr<T>& operator=(MatrixPtr<T>&Right);
		//inline MatrixPtr<T>& operator=(MatrixPtr<T> Right);
		//compare M
		//inline bool operator==(MatrixPtr<T>&Right);


		inline void Print() const {
			ptr->ConsolePrint();
		}
		//Default(-0.1,0.1)*factor
		inline void RandData(floatType factor = 1.0)const {
			ptr->RandData(factor);
		}
		inline void WriteToDevice(int write_row = -1)const {
			if (ptr)ptr->WriteToCuda(write_row);
		}
		inline T* ReadFromDevice(bool pinned = false)const {
			if (ptr != NULL)
				return ptr->ReadFromCuda(pinned);
			return NULL;
		}
		inline T* GetMemData()const {
			if (ptr != NULL) {
				if (ptr->M == NULL)ptr->M = new T[ptr->row*ptr->col];
				return ptr->M;
			}
			return NULL;
		}


		//used for no-const variable
		//functions only return this
		inline MatrixPtr<T>& operator*=(const MatrixPtr<T>&Right) {
			((const MatrixPtr<T>*)this)->operator*=(Right);
			return *this;
		}
		inline MatrixPtr<T>& operator+=(const MatrixPtr<T>&Right) {
			((const MatrixPtr<T>*)this)->operator+=(Right);
			return *this;
		}
		inline MatrixPtr<T>& Order_Assign_Val(const MatrixPtr<T>&Order, T val) {
			((const MatrixPtr<T>*)this)->Order_Assign_Val(Order, val);
			return *this;
		}
		inline MatrixPtr<T>& RandMatrix(const MatrixPtr<T>&Right, const MatrixPtr<T>&Order, int Start_Row = 0, bool Right_Mat_random_Order = true, ui Max_Order = 0) {
			((const MatrixPtr<T>*)this)->RandMatrix(Right, Order, Start_Row, Right_Mat_random_Order, Max_Order);
			return *this;
		}
		inline MatrixPtr<T>& f(int fun_id, floatType param = -1, int Start_Row = 0) {
			((const MatrixPtr<T>*)this)->f(fun_id, param, Start_Row);
			return *this;
		}
		inline MatrixPtr<T>& SoftMax(bool Max = false, MatrixPtr<T>*Scale_One_Col = NULL) {
			((const MatrixPtr<T>*)this)->SoftMax(Max, Scale_One_Col);
			return *this;
		}
		inline MatrixPtr<T>& ResetRows(int New_Row) {
			((const MatrixPtr<T>*)this)->ResetRows(New_Row);
			return *this;
		}
		inline MatrixPtr<T>& ResetCols(int New_Col) {
			((const MatrixPtr<T>*)this)->ResetCols(New_Col);
			return *this;
		}
		inline MatrixPtr<T>& operator-=(const MatrixPtr<T>&Right) {
			((const MatrixPtr<T>*)this)->operator-=(Right);
			return *this;
		}

	};
	template<class T>
	inline std::ostream& operator<<(std::ostream&os, MatrixPtr<T>&Out) {
		if (Out.IsValid())
			Out.Print();
		return os;
	}
	template<class T>
	inline std::ostream& operator<<(std::ostream&os, const MatrixPtr<T>&Out) {
		if (Out.IsValid())
			Out.Print();
		return os;
	}
	template CUDA_ACC std::ostream& operator<<(std::ostream&os, MatrixPtr<floatType>&Out);
	template CUDA_ACC std::ostream& operator<<(std::ostream&os, const MatrixPtr<floatType>&Out);

	
	typedef MatrixPtr<floatType> Mat;
	template class CUDA_ACC MatrixPtr<floatType>;
}

#endif