#pragma once

#ifndef MATRIX_H
#define MATRIX_H

#ifdef MATRIX_EXPORTS
#define MATRIX __declspec(dllexport)
#else
#define MATRIX __declspec(dllimport)

#endif


#include<iostream>

typedef unsigned int ui;

namespace Matrix_CPU {

	template<class T>  class CPU_Mat
	{
	public:
		//rows,cols
		ui row, col;
		//pointer to data
		T*M;

		void InitParam();

		CPU_Mat();
		CPU_Mat(ui row, ui col);
		CPU_Mat(ui row, ui col, T*Data);
		CPU_Mat(ui row, ui col, T**Data);
		CPU_Mat(ui row, ui col, ui Start_col, T**Data);

		inline CPU_Mat(const CPU_Mat<T>&R) {
			memcpy(this, &R, sizeof(CPU_Mat<T>));
		};

		~CPU_Mat();
		inline void Clear() {
			memset(this, 0, sizeof(CPU_Mat<T>));
		}

		void RandData();
		void ConsolePrint();

		CPU_Mat<T>* operator*(CPU_Mat<T>*R);
		CPU_Mat<T>* operator+(CPU_Mat<T>*R);
		CPU_Mat<T>* operator-(CPU_Mat<T>*R);
		CPU_Mat<T>* div(const CPU_Mat<T>*R, bool mul = false);
		CPU_Mat<T>* operator-=(CPU_Mat<T>*R);
		CPU_Mat<T>* operator+=(CPU_Mat<T>*R);
		CPU_Mat<T>* operator!();
		const CPU_Mat<T>* _ZeroMemory();
		const CPU_Mat<T>* mul_Matrix(CPU_Mat<T>*R);
		const CPU_Mat<T>* f(int fun_id);
		CPU_Mat<T>* Sum();
		const CPU_Mat<T>*Softmax();
		//compare M
		bool operator==(T*M);
		inline void Swap(const CPU_Mat<T>*R) {
			ui sz = sizeof(CPU_Mat<T>);
			void*dst = malloc(sz);
			memcpy(dst, this, sz);
			memcpy(this, R, sz);
			memcpy((void*)R, dst, sz);
			free(dst);
		}
		//move assignment op
		inline CPU_Mat<T>&operator=(const CPU_Mat<T>&&R) {
			memcpy(this, &R, sizeof(CPU_Mat<T>));
			return *this;
		}
		inline CPU_Mat<T>&operator=(const CPU_Mat<T>&R) {
			memcpy(this, &R, sizeof(CPU_Mat<T>));
			return *this;
		}
	};

	template<class T> std::ostream& operator<<(std::ostream&os, CPU_Mat<T>*a);
	template MATRIX std::ostream& operator<<(std::ostream&os, CPU_Mat<double>*a);



	CUDA_ACC bool MatrixEnd();
	//Inititalize func variables
	CUDA_ACC void Matrix_Set_Function(void**d_f, void**host_f);

	template<class T> class MatrixPtr {
	private:
		CPU_Mat<T>*ptr;
		inline MatrixPtr(CPU_Mat<T>*CPU_Mat) {
			ptr = CPU_Mat;
		}
	public:
		inline MatrixPtr() {
			ptr = NULL;
		}
		inline MatrixPtr(ui row, ui col) {
			ptr = new CPU_Mat<T>(row, col);
		}
		inline MatrixPtr(ui row, ui col, T*Data, bool WriteToDevice = false) {
			ptr = new CPU_Mat<T>(row, col, Data);
			if (WriteToDevice)
				this->WriteToDevice();
		}
		inline ~MatrixPtr() {
			delete ptr;
			ptr = NULL;
		}
		void Reset(ui row, ui col) {
			if (ptr)delete ptr;
			ptr = new CPU_Mat<T>(row, col);
		}
		void Reset(ui row, ui col, T*Data, bool WriteToDevice = false) {
			if (ptr)delete ptr;
			ptr = new CPU_Mat<T>(row, col, Data);
			if (WriteToDevice)
				this->WriteToDevice();
		}
		void Reset(ui row, ui col, T**Data) {
			if (ptr)delete ptr;
			ptr = new CPU_Mat<T>(row, col, Data);
		}
		void Reset(ui row, ui col, ui Start_col, T**Data) {
			if (ptr)delete ptr;
			ptr = new CPU_Mat<T>(row, col, Start_col, Data);
		}
		bool IsValid() {
			return ptr != NULL;
		}
		inline CPU_Mat<T>* getMatrix()const {
			return ptr;
		}
		inline MatrixPtr<T>& _ZeroMemory();

		inline MatrixPtr<T> operator*(const MatrixPtr<T>&Right);
		inline MatrixPtr<T> operator+(const MatrixPtr<T>&Right);
		inline MatrixPtr<T> operator-(const MatrixPtr<T>&Right);
		inline MatrixPtr<T> operator/(const MatrixPtr<T>&Right);
		inline MatrixPtr<T>& operator*=(const MatrixPtr<T>&Right);
		inline MatrixPtr<T>& operator-=(const MatrixPtr<T>&Right);
		inline MatrixPtr<T>& operator+=(const MatrixPtr<T>&Right);
		inline MatrixPtr<T> operator!();
		inline MatrixPtr<T> Sum();
		inline MatrixPtr<T>& SoftMax();


		inline MatrixPtr<T>& f(int fun_id);
		inline MatrixPtr<T> _f(int fun_id);

		inline T& operator[](ui idx) {
			return ptr->M[idx];
		}
		//copy Cuda_M value
		inline MatrixPtr<T>& operator=(const MatrixPtr<T>&Right);
		//copy ptr pointer
		inline MatrixPtr<T>& operator=(MatrixPtr<T>&Right);
		//inline MatrixPtr<T>& operator=(MatrixPtr<T> Right);
		//compare M
		//inline bool operator==(MatrixPtr<T>&Right);


		void Print() {
			if (CUDA_ACC_Enabled)
				ptr->ReadFromCuda();
			ptr->ConsolePrint();
		}
		void RandData() {
			ptr->RandData();
		}
		inline void WriteToDevice() {
			if (ptr)ptr->WriteToCuda();
		}
		inline T* ReadFromDevice() {
			if (ptr != NULL)
				return ptr->ReadFromCuda();
			return NULL;
		}
	};
	template<class T> std::ostream& operator<<(std::ostream&os, MatrixPtr<T>&Out);
	template CUDA_ACC std::ostream& operator<<(std::ostream&os, MatrixPtr<double>&Out);

	template class CUDA_ACC  MatrixPtr<double>;
	typedef CUDA_ACC MatrixPtr<double> Mat;
}

#endif