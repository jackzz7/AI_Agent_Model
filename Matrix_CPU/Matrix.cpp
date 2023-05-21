#ifdef _WIN64
#define __GPU__
#endif

#ifndef __GPU__

#define CUDA_ACC_EXPORTS
#include"CUDA_ACC.h"

using namespace std;
using namespace _CUDA_;

//CPU


//#define type(x) decltype(x)
//gcc
//#define max(a,b) ({type(a) _a=(a);type(b) _b=(b);_a>_b?_a:_b;})
//#define min(a,b)((a)<(b)?(a):(b))
//#define max(a,b)((a)>(b)?(a):(b))
template<class T> const inline T min(T a, T b) { return a < b ? a : b; }
template<class T> const inline T max(T a, T b) { return a > b ? a : b; }


#define __device__

__device__ void _Sigmoid(double&In)
{
	double A = 1 + exp(-In);
	In = 1.0 / A;
}
__device__ void Sigmoid_Derivative(double&In)
{
	In = In * (1.0 - In);
	//double A = exp(-In);
	//In = A / ((1 + A)*(1 + A));
}
__device__ void valLog(double&In)
{
	//In>0
	In = log(In + 1e-16);
}
__device__ void _Pow2(double&In)
{
	In = In * In;
}
__device__ void div_Sqrt(double&In)
{
	//In>0
	In = 1.0 / (sqrt(In + 1e-16) + 1e-8);
}
__device__ void _One_Minus(double&In)
{
	In = 1.0 - In;
}
__device__ void _ReLU(double&In)
{
	In = max(0.0, In) + 0.01*min(0.0, In);
	//In = max(0.0, In);
}
__device__ void ReLU_Derivative(double&In)
{
	if (In <= 0)In = 0.01;
	else In = 1.0;
}
__device__ void Drop_Ori(double&In)
{
	if (In != 0.0)In = 1.0;
}
__device__ void Drop_Mod(double&In)
{
	if (In == 0.0)In = 1.0;
	else In = 0.0;
}
_fun host_f[10] = { NULL };
__device__ _fun d_f[10] = { _ReLU,ReLU_Derivative,valLog,_Pow2,div_Sqrt,_One_Minus,_Sigmoid,Sigmoid_Derivative,Drop_Ori };

void _CUDA_::Matrix_Set_Function(void**d_f, void**host_f, int MaxRandomStates_rows) {}

template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_mul(Matrix<T>*R) {
	//if (col != R->row)return NULL;
	Matrix<T>* C = new Matrix<T>(row, R->col);
	for (int i = 0; i < R->col; i++)
		for (int j = 0; j < row; j++) {
			double sum = 0;
			for (int k = 0; k < col; k++)
				sum += M[j*col + k] * R->M[k*R->col + i];
			C->M[j*C->col + i] = sum;
		}
	return C;
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_plus(Matrix<T>*R) {
	Matrix<T>* C = new Matrix<T>(row, max(col, R->col));
	if (col == R->col)
		for (ui i = 0; i < col; i++)
			for (ui j = 0; j < row; j++) {
				C->M[j*col + i] = M[j*col + i] + R->M[j*col + i];
			}
	else for (ui i = 0; i < C->col; i++)
		for (ui j = 0; j < C->row; j++) {
			C->M[j*C->col + i] = M[j*col + min(i, col - 1)] + R->M[j*R->col + min(i, R->col - 1)];
		}
	return C;
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_minus(Matrix<T>*R) {
	Matrix<T>* C = new Matrix<T>(row, max(col, R->col));
	for (int i = 0; i < col; i++)
		for (int j = 0; j < row; j++) {
			C->M[j*col + i] = M[j*col + i] - R->M[j*col + i];
		}
	return C;
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_div_by_number(const Matrix<T>*R, bool mul) {
	T number = R->M[0];
	bool flag = (mul && (row == 1 && col == 1));
	if (!mul)number = 1 / number;
	else if (flag)number = M[0];
	int col = flag ? R->col : this->col;
	int row = flag ? R->row : this->row;
	Matrix<T>* C = new Matrix<T>(row, col);
	for (int i = 0; i < col; i++)
		for (int j = 0; j < row; j++) {
			if (flag)
				C->M[j*col + i] = R->M[j*col + i] * number;
			else C->M[j*col + i] = M[j*col + i] * number;
		}
	return C;
}
template<class T>const Matrix<T>* Matrix<T>::Cuda_ACC_minus_(Matrix<T>*R) {
	//缩列
	if (col < R->col) {
		for (int i = 0; i < col; i++)
			for (int j = 0; j < row; j++) {
				for (int k = 0; k < R->col; k++)
					M[j*col + i] -= R->M[j*R->col + k];
			}
	}
	else for (int i = 0; i < col; i++)
		for (int j = 0; j < row; j++) {
			M[j*col + i] -= R->M[j*col + i];
		}
	return this;
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_plus_(Matrix<T>*R) {
	for (int i = 0; i < col; i++)
		for (int j = 0; j < row; j++) {
			M[j*col + i] += R->M[j*col + i];
		}
	return this;
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_T() {
	Matrix<T>* C = new Matrix<T>(col, row);
	for (int i = 0; i < col; i++)
		for (int j = 0; j < row; j++) {
			C->M[i*C->col + j] = M[j*col + i];
		}
	return C;
}
template<class T> const Matrix<T>* Matrix<T>::Cuda_ACC_ZeroMemory() {
	memset(M, 0, sizeof(T)*col*row);
	return this;
}
template<class T> const Matrix<T>* Matrix<T>::Cuda_ACC_mul_Matrix(Matrix<T>*R) {
	//同型
	if (row == R->row&&col == R->col) {
		for (int i = 0; i < col; i++)
			for (int j = 0; j < row; j++) {
				M[j*col + i] *= R->M[j*col + i];
			}
	}//一个数
	else if(R->row==1&&R->col==1){
		for (int i = 0; i < col; i++)
			for (int j = 0; j < row; j++) {
				M[j*col + i] *= R->M[0];
			}
	}
	//乘列
	else {
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++) {
				M[i*col + j] *= R->M[i];
			}
	}
	//乘行
	else {

	}
	return this;
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_function(int fun_id, bool TemporaryVariable,double param,int Start_Row) {
	Matrix<T>*Out = this;
	if (TemporaryVariable) {
		Out = new Matrix<T>(row, col,M);
	}
	_fun f = d_f[fun_id];
	for (int i = 0; i < col; i++)
		for (int j = Start_Row; j < row; j++) {
			f(Out->M[j*col + i]);
		}
	return Out;
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_Sum() {
	Matrix<T>* C = new Matrix<T>(1, 1);
	double Sum = 0;
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++) {
			Sum += M[i*col + j];
		}
	C->M[0] = Sum;
	return C;
}
template<class T>const Matrix<T>* Matrix<T>::Cuda_ACC_SoftMaxFunction(bool Max) {
	for (int i = 0; i < col; i++) {
		T mx = M[0 * col + i], Sum = 0;
		for (int j = 0; j < row; j++)
			if (M[j*col + i] > mx)mx = M[j*col + i];
		for (int j = 0; j < row; j++)
			Sum += exp(M[j*col + i] - mx);
		for (int j = 0; j < row; j++) {
			M[j*col + i] = exp(M[j*col + i] - mx) / Sum;
		}
	}
	return this;
}
template<class T> const Matrix<T>* Matrix<T>::Cuda_ACC_generate_random_number() {
	return this;
}
template<class T> const Matrix<T>* Matrix<T>::Cuda_ACC_RandMatrixs(Matrix<T>*R, Matrix<T>*Order, int Start_row) {
	return this;
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_ScaleOneCol(bool OneRow) { return this; }
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_assign(Matrix<T>*R, int dst_Start_row) {
	size_t sz = sizeof(T)*row*col;
	memcpy_s(M, sz, R->M, sz);
	return this;
}
template<class T> T* Matrix<T>::WriteToCuda() {
	return Cuda_M;
}
template<class T> T* Matrix<T>::ReadFromCuda() {
	return M;
}


template<class T> MatrixPtr<T>& MatrixPtr<T>::operator-=(const MatrixPtr<T>&Right) {

	/*if (ptr->col < Right.getMatrix()->col) {
		MatrixPtr<T>&R = MatrixPtr<T>(Right.getMatrix()->Cuda_ACC_ScaleOneCol());
		ptr->Cuda_ACC_minus_(R.getMatrix());
	}
	else ptr->Cuda_ACC_minus_(Right.getMatrix());*/
	ptr->Cuda_ACC_minus_(Right.getMatrix());

	/*if (CUDA_ACC_Enabled) {
		if (ptr->col < Right.getMatrix()->col) {
			MatrixPtr<T>&R = MatrixPtr<T>(Right.getMatrix()->Cuda_ACC_ScaleOneCol());
			ptr->Cuda_ACC_minus_(R.getMatrix());
		}
		else ptr->Cuda_ACC_minus_(Right.getMatrix());
	}
	else ptr->operator-=(Right.getMatrix());*/
	return *this;
}


#endif
