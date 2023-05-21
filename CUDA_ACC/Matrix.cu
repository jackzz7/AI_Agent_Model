
//#ifdef _WIN64
#define __GPU__
//#endif

#ifdef __GPU__

#define CUDA_ACC_EXPORTS
#include"CUDA_ACC.h"
#include<time.h>

#include<cuda_runtime.h>
#include<device_functions.h>
#include<cublas_v2.h>
#include<device_launch_parameters.h>
#include<curand.h>
#include<curand_kernel.h>

#include<map>
#include<vector>
#include<random>
#include<list>
#include<cfloat>

#include<mutex>


using namespace _CUDA_;
using namespace std;

#define cudaDeviceSynchronize()
__device__ const int FLT_Maximum_Int = float_Max_Int;//pow(10, FLT_DIG);
__device__ inline
void _syncthreads() { __syncthreads(); }
constexpr size_t sz = sizeof(floatType);
const int Maximum_Cuda_Block_Size = 512;


template<class T> Matrix<T>::~Matrix() {
	Disponse();
	row = col = 0;
}
template<class T> void Matrix<T>::Disponse() {
#ifdef __GPU__
	FreeCuda_M();
	ui flag;
	if (!(cudaHostGetFlags(&flag, M) == cudaErrorInvalidValue)) {
		cudaError_t err;
		if ((err = cudaFreeHost(M)) != cudaSuccess && err != cudaErrorCudartUnloading)cout << "cudaFreeHost error: " << err << endl, assert(false);
		M = NULL;
	}
#endif // __GPU__
	delete[]M; M = NULL;
}
template<class T>void Matrix<T>::FreeCuda_M() {
	if (Cuda_M)
		cudaFree(Cuda_M), Cuda_M = NULL;
	WidthStride = pitch = 0;
}

//#define type(x) decltype(x)
//gcc
//#define max(a,b) ({type(a) _a=(a);type(b) _b=(b);_a>_b?_a:_b;})
//#define min(a,b)((a)<(b)?(a):(b))
//#define max(a,b)((a)>(b)?(a):(b))
template<class T> inline constexpr T min(const T& a, const T& b) { return a < b ? a : b; }
template<class T> inline constexpr T max(const T& a, const T& b) { return a > b ? a : b; }
//int(n)/int(m) floor
__device__ constexpr inline int floor(const int& n, const int& m) {
	//assert(m != 0);
	return n / m + (n % m + m) / m - 1;
}

__device__ const floatType eps = 1e-8f;
__device__ floatType _Sigmoid(const floatType& In)
{
	return 1.0f / (1.0f + exp(-In));
}
__device__ floatType Sigmoid_Derivative(const floatType& In)
{
	return In * (1.0f - In);
}
__device__ floatType valLog(const floatType& In)
{
	//In>0
	assert(In + eps > 0);
	return log(In + eps);
}
__device__ floatType _Pow2(const floatType& In)
{
	return In * In;
}
__device__ floatType _Pow3(const floatType& In)
{
	return In * In * In;
}
__device__ floatType div_Sqrt(const floatType& In)
{
	//In>0
	assert(In + eps > 0);
	return 1.0f / sqrt(In + eps);
}
__device__ floatType _One_Minus(const floatType& In)
{
	return 1.0f - In;
}
__device__ floatType _ReLU(const floatType& In)
{
	assert(!isnan(In));
	return ::max((floatType)0.0f, In) + 0.01f * ::min((floatType)0.0f, In);
}
__device__ floatType ReLU_Derivative(const floatType& In)
{
	if (In <= 0.0f)return 0.01f;
	else return 1.0f;
}
__device__ floatType Bool(const floatType& In)
{
	if (abs(In) > eps)return 1.0f;
	else return 0.0f;
}
__device__ floatType Tanh(const floatType& In) {
	return 1.0f - 2.0f / (exp(2.0f * In) + 1.0f);
}
__device__ floatType D_Tanh(const floatType& In) {
	return 1.0f - In * In;
}
__device__ floatType Sqrt(const floatType& In) {
	assert(In + eps >= 0.0f);
	return sqrt(In + eps);
}
__device__ floatType HardTanh(const floatType& In) {
	if (In > 1.0f)return 1.0f;
	else if (In < -1.0f)return -1.0f;
	return In;
}
__device__ floatType D_HardTanh(const floatType& In) {
	if (In < -1.0f || In > 1.0f)return 0.0f;
	else return 1.0f;
}
__device__ floatType Leaky_HardTanh(const floatType& In) {
	if (In > 1.0f)return 0.01f * In + 0.99f;
	else if (In < -1.0f)return 0.01f * In - 0.99f;
	return In;
}
__device__ floatType D_Leaky_HardTanh(const floatType& In) {
	if (In < -1.0f || In > 1.0f)return 0.01f;
	else return 1.0f;
}
__device__ floatType Nor(const floatType& In) {
	if (In < 0)return -1;
	else if (In > 0)return 1;
	else return 0;
}
__device__ floatType DropOut_Bernoulli(const floatType& In, const floatType& random)
{
	return (random >= In) ? (1.0f / (1.0f - In)) : 0.0f;
}
__device__ floatType Assignment(const floatType& In, const floatType& param) {
	return param;
}
__device__ floatType Compare(const floatType& In, const floatType& param) {
	return (abs(In) >= param) ? 1.0f : 0.0f;
}
__device__ floatType Thresholding(const floatType& In, const floatType& param) {
	if (In > param)return param;
	else if (In < -param)return -param;
	else return In;
}
//(-1,1)
__device__ floatType Uniform(const floatType& In, const floatType& param) {
	return In * (param * 2.0f - 1.0f);
}
__device__ floatType Division(const floatType& In, const floatType& param) {
	return In / (param + eps);
}
__device__ floatType Number_div(const floatType& In, const floatType& param) {
	//In>0
	return param / (In + eps);
}
const int MaxFun = 20;
_fun dev_f[MaxFun] = { NULL };
_fun2 dev_f2[MaxFun] = { NULL };
__device__ _fun d_f[MaxFun] = { NULL, _ReLU,ReLU_Derivative,valLog,_Pow2,_Sigmoid,Sigmoid_Derivative,::Bool,::Tanh,::D_Tanh,::Sqrt,::HardTanh,::D_HardTanh,
::Leaky_HardTanh,::D_Leaky_HardTanh,::_Pow3,::Nor };
__device__ _fun2 d_f2[MaxFun] = { ::DropOut_Bernoulli,::Assignment,::Compare,::Thresholding,::Uniform,::Division,::Number_div };

__device__ floatType* Device_Zero = NULL;
__device__ floatType* Device_One = NULL;


#ifdef __GPU__

#define get_tid this_thread::get_id
//rand_states
curandState** states = new curandState * [Cuda_Max_Stream] {NULL};
size_t states_pitch = 0;
cudaStream_t* stms = new cudaStream_t[Cuda_Max_Stream]{ NULL };
cublasHandle_t handle[Cuda_Max_Stream]{ NULL };

map<thread::id, int>_CUDA_::stmID;
list<int>stms_space(1);

Matrix_Heap<floatType> _CUDA_::Heap[Cuda_Max_Stream];

mutex stm_lock;
void stmReset();
int _CUDA_::BindStm() {
	lock_guard<mutex>stm_locker(stm_lock);
	if (stms[0] == NULL)stmReset();
	assert(!stms_space.empty());
	if (stms_space.empty())printf("error:no more stms\n");
	auto ret = stmID[get_tid()] = stms_space.front();
	stms_space.pop_front();
	return ret;
}
void _CUDA_::unBindStm() {
	//ensure sync
	lock_guard<mutex>stm_locker(stm_lock);
	if (stmID.find(get_tid()) != stmID.end()) {
		stms_space.push_back(stmID[get_tid()]);
		stmID.erase(get_tid());
	}
}
void stmReset() {
	stmID.clear(); stms_space.clear();
	for (int i = 0; i < Cuda_Max_Stream; i++) {
		if (stms[i])cudaStreamDestroy(stms[i]); cudaStreamCreate(&stms[i]);
		if (handle[i])cublasDestroy(handle[i]); cublasCreate(&handle[i]);
		cublasSetStream(handle[i], stms[i]);
		Heap[i].clear();
		stms_space.push_back(i);
	}
}
void _CUDA_::StreamSynchronize() {
	cudaStreamSynchronize(get_stm);
}
Matrix<floatType>* malloc_Matrix(int row, int col) {
	return Heap[get_stm_id].malloc(row, col);
}

__global__ void Cuda_random_init(unsigned long long seed, int col, int states_pitch, curandState* states) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx)
		curand_init(seed, bix * col + i, 0, &states[bix * states_pitch + i]);
}
void _CUDA_::Matrix_Set_Function(void** d_f, void** host_f, int MaxRandomStates_rows) {
	//cudaDeviceReset();
	//stmReset();
	std::vector<std::uint32_t>seeds;
	seeds.resize(Cuda_Max_Stream);
	std::seed_seq seq{ time(0) };
	seq.generate(seeds.begin(), seeds.end());

	for (int i = 0; i < Cuda_Max_Stream; i++) {
		//ensure different seeds
		if (states[i])cudaFree(states[i]);
		DEBUG(cudaMallocPitch(&states[i], &states_pitch, sizeof(curandState) * Maximum_Cuda_Block_Size, MaxRandomStates_rows) != cudaSuccess, "curandState cudaMalloc error\n");
		states_pitch /= sizeof(curandState);

		dim3 grid(MaxRandomStates_rows, 1, 1), block(256, 1, 1);
		Cuda_random_init << < grid, block >> > (seeds[i], Maximum_Cuda_Block_Size, states_pitch, states[i]);
	}

	size_t sz;
	cudaGetSymbolSize(&sz, ::d_f);
	cudaMemcpyFromSymbolAsync(dev_f, ::d_f, sz, 0, cudaMemcpyDeviceToHost);
	cudaGetSymbolSize(&sz, ::d_f2);
	cudaMemcpyFromSymbolAsync(dev_f2, ::d_f2, sz, 0, cudaMemcpyDeviceToHost);
	if (Device_One) {
		cudaFree(Device_Zero);
		cudaFree(Device_One);
	}
	cudaMalloc(&Device_One, ::sz);
	cudaMalloc(&Device_Zero, ::sz);
	cudaMemsetAsync(Device_Zero, 0, ::sz);
	cudaMemcpyAsync(Device_One, &(const floatType&)(1.0f), ::sz, cudaMemcpyHostToDevice);
}
template<class T> T* Matrix<T>::alloc_Cuda_M() {
	if (Cuda_M == NULL) {
		cudaError_t err;
		if ((err = cudaMallocPitch(&Cuda_M, &pitch, WidthStride, row)) != cudaSuccess)
			cout << "cudaMalloc errorCode: " << err << '\n', assert(false);
	}return Cuda_M;
}

template<class T> T* Matrix<T>::WriteToCuda(int write_row) {
	alloc_Cuda_M();
	if (M != NULL)
		cudaMemcpy2DAsync(Cuda_M, pitch, M, WidthStride, WidthStride, write_row == -1 ? row : write_row, cudaMemcpyHostToDevice, get_stm);
	return Cuda_M;
}
template<class T> T* Matrix<T>::ReadFromCuda(bool pinned) {
	if (M == NULL) {
		if (pinned) {
			if (cudaHostAlloc(&M, sz * row * col, cudaHostAllocPortable) != cudaSuccess)
				cout << "cudaHostAlloc error\n", assert(false);
		}
		else M = new T[row * col];
	}
	else if (pinned) {
		ui flag; if (cudaHostGetFlags(&flag, M) == cudaErrorInvalidValue) {
			delete[]M;
			if (cudaHostAlloc(&M, sz * row * col, cudaHostAllocPortable) != cudaSuccess)
				cout << "cudaHostAlloc error\n", assert(false);
		}
	}
	cudaMemcpy2DAsync(M, WidthStride, Cuda_M, pitch, WidthStride, row, cudaMemcpyDeviceToHost, get_stm);
	StreamSynchronize();
	return M;
}

//__global__ void Cuda_Matrix_mul(int A_pitch, int B_pitch, double*A, double*B, double*Out, int A_col, int B_col) {
//	extern __shared__ double _shared[];
//
//	ui tix = threadIdx.x;
//	ui tiy = threadIdx.y;
//	ui bix = blockIdx.x;
//	//int biy = blockIdx.y;
//	ui bdx = blockDim.x;
//	ui bdy = blockDim.y;
//
//	double&sum = _shared[tix*bdy + tiy];
//	for (int i = tix; i < B_col; i += bdx) {
//		sum = 0;
//		for (int j = tiy; j < A_col; j += bdy) {
//			sum += A[bix*A_pitch + j] * B[j*B_pitch + i];
//		}
//		if (bdy > 1) {
//			__syncthreads();
//			if (tiy == 0) {
//				double*A = &sum;
//				for (int j = 1; j < bdy; j++)
//					sum += A[j];
//				Out[bix*B_pitch + i] = sum;
//			}
//			__syncthreads();
//		}
//		else Out[bix*B_pitch + i] = sum;
//	}
//}
__global__ void Cuda_Matrix_plus_minus_rnumber(int pitch, floatType* Out, floatType* A, const floatType number, int col, const int sign) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		Out[bix * pitch + i] = A[bix * pitch + i] + sign * number;
	}
}
__global__ void Cuda_Matrix_plus_minus_lnumber(int pitch, floatType* Out, floatType* A, const floatType number, int col, const int sign) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		Out[bix * pitch + i] = number + sign * A[bix * pitch + i];
	}
}
__global__ void Cuda_Matrix_div_Matrix(int Out_pitch, int A_pitch, int B_pitch, floatType* A, floatType* B, floatType* Out, ui Out_col, ui A_row, ui A_col, ui B_row, ui B_col) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (ui i = tix; i < Out_col; i += bdx) {
		Out[bix * Out_pitch + i] = A[::min(bix, A_row - 1) * A_pitch + ::min(i, A_col - 1)] / B[::min(bix, B_row - 1) * B_pitch + ::min(i, B_col - 1)];
	}
}
__global__ void Cuda_Matrix_mul_Matrix(int Out_pitch, int A_pitch, int B_pitch, floatType* A, floatType* B, floatType* Out, ui Out_col, ui A_row, ui A_col, ui B_row, ui B_col) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (ui i = tix; i < Out_col; i += bdx) {
		Out[bix * Out_pitch + i] = A[::min(bix, A_row - 1) * A_pitch + ::min(i, A_col - 1)] * B[::min(bix, B_row - 1) * B_pitch + ::min(i, B_col - 1)];
	}
}
__global__ void Cuda_Matrix_plus_Matrix(int Out_pitch, int A_pitch, int B_pitch, floatType* A, floatType* B, floatType* Out, ui Out_col, ui A_row, ui A_col, ui B_row, ui B_col) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (ui i = tix; i < Out_col; i += bdx) {
		Out[bix * Out_pitch + i] = A[::min(bix, A_row - 1) * A_pitch + ::min(i, A_col - 1)] + B[::min(bix, B_row - 1) * B_pitch + ::min(i, B_col - 1)];
	}
}
__global__ void Cuda_Matrix_minus_Matrix(int Out_pitch, int A_pitch, int B_pitch, floatType* A, floatType* B, floatType* Out, ui Out_col, ui A_row, ui A_col, ui B_row, ui B_col) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (ui i = tix; i < Out_col; i += bdx) {
		Out[bix * Out_pitch + i] = A[::min(bix, A_row - 1) * A_pitch + ::min(i, A_col - 1)] - B[::min(bix, B_row - 1) * B_pitch + ::min(i, B_col - 1)];
	}
}
__global__ void Cuda_Matrix_Order_Assign(int A_pitch, int B_pitch, floatType* A, floatType* B, int B_col, floatType val) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < B_col; i += bdx) {
		A[i * A_pitch + (ui)B[bix * B_pitch + i]] = val;
	}
}
const unsigned long long Var = 0xffffffLL * 0xffffff;
__global__ void Cuda_Matrix_Normal_Sampling(int pitch, floatType* A, int col, int row, curandState* states, int states_pitch) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	ui No = bix * states_pitch + tix;
	//Marsaglia polar method
	bool even = ((bix * 2 + 1) < row);
	for (int i = tix; i < col; i += bdx) {
		long long x, y; unsigned long long S;
		double _x, _y;
		do {
			x = (long long)(curand(&states[No]) % (0x1ffffff)) - 0xffffff, y = (long long)(curand(&states[No]) % (0x1ffffff)) - 0xffffff;
		} while ((S = x * x + y * y) > Var);
		if (S != 0) {
			double _S = 1.0 * S / Var;
			double C = sqrt(-2.0f * log(_S) / _S) / 0xffffff;
			_x = x * C;
			_y = y * C;
		}
		else {
			_x = _y = 0.0f;
		}
		A[bix * 2 * pitch + i] = _x;
		if (even)
			A[(bix * 2 + 1) * pitch + i] = _y;
	}
}
__device__ void swap(int& x, int& y) {
	int t = x;
	x = y;
	y = t;
}
__device__ void Rotate(int& x, int& y, const int type) {
	switch (type)
	{
	case 0:
		break;
	case 1:
		swap(x, y);
		break;
	case 2:
		swap(x, y);
		x *= -1;
		break;
	case 3:
		x *= -1;
		break;
	case 4:
		x *= -1, y *= -1;
		break;
	case 5:
		swap(x, y);
		x *= -1, y *= -1;
		break;
	case 6:
		swap(x, y);
		y *= -1;
		break;
	case 7:
		y *= -1;
		break;
	default:
		assert(false);
		break;
	}
}
__device__ void BoardRotated(const int& cx, const int& cy, int& mx, int& my, const int type) {
	mx -= cx, my -= cy;
	Rotate(mx, my, type);
	mx += cx, my += cy;
}
__device__ int DeRotation[8] = { 0,1,6,3,4,5,2,7 };
//Gomoku Board Image Depth
const int Gomoku_Planes = 4;// 12 + 8 + 8 + 4;// +1;
__global__ void Cuda_Matrix_Board_Simulation(int pitch, int Moves_pitch, floatType* Board, int col, floatType* Moves, floatType* board_id, floatType* Reward, const floatType* Random) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		int gameid = (int)board_id[i] / 1000, stepid = (int)board_id[i] % 1000;
		if (bix < stepid) {
			int mov = Moves[bix * Moves_pitch + gameid] - 1;
			//random rotation
			if (Random) {
				assert(0 <= Random[i] && Random[i] < 8);
				int mx = mov % 15, my = mov / 15;
				BoardRotated(7, 7, mx, my, Random[i]);
				mov = my * 15 + mx;
			}
			assert(0 <= mov && mov < 15 * 15);
			//player/opponent
			Board[(mov * Gomoku_Planes + (((stepid % 2) == (bix % 2)) ? 0 : 1)) * pitch + i] = 1;
			//Turns Since
			//Board[(mov*Gomoku_Planes + min(3 + (stepid - bix), 11))*pitch + i] = 1;
			//distance to boundary(0,1,2,3,>3)
			/*int _x = mov % 15, _y = mov / 15;
			int dis = min(min(_x, _y), min(14 - _x, 14 - _y));
			if (dis < 4) {
				assert(0 <= dis && dis < 4);
				Board[(mov*Gomoku_Planes + 28 + dis)*pitch + i] = 1;
			}*/
			//OutPut mask
			if (Reward)Reward[mov * pitch + i] = -1e9f;
		}
		//board boundary
		Board[(bix * Gomoku_Planes + 3) * pitch + i] = 1;
	}
}
__device__ const int dir[8][2] = { {1,0},{0,1},{1,1},{1,-1},{-1,0},{0,-1},{-1,-1},{-1,1 } };
__global__ void Cuda_Matrix_Board_Simulation_cal(int pitch, int Moves_pitch, floatType* Board, int col, floatType* Moves, floatType* board_id, floatType* Reward) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	int x = bix % 15, y = bix / 15;
	for (int i = tix; i < col; i += bdx) {
		//Liberties(1,2.....8)
		/*int lib = 0;
		for (int j = 0; j < 8; j++) {
			int _x = dir[j][0] + x, _y = dir[j][1] + y;
			if (_x < 0 || _y < 0 || _x >= 15 || _y >= 15)continue;
			if (Board[((_y * 15 + _x)*Gomoku_Planes + 2)*pitch + i] == 0)
				lib++;
		}
		if (lib > 0)
			Board[((y * 15 + x)*Gomoku_Planes + 11 + lib)*pitch + i] = 1;*/
			//empty point
		if (Board[(bix * Gomoku_Planes + 0) * pitch + i] == 0 && Board[(bix * Gomoku_Planes + 1) * pitch + i] == 0)
			Board[(bix * Gomoku_Planes + 2) * pitch + i] = 1;
	}
}
__global__ void Cuda_Matrix_Board_Simulation_(int pitch, int Moves_pitch, floatType* Board, int col, floatType* Moves, floatType* board_id, floatType* Reward, int Reward_pitch, const floatType* Random,
	floatType* Value, bool One_Encode) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		int gameid = (int)board_id[i] / 1000, stepid = (int)board_id[i] % 1000;
		int mov = One_Encode ? (Moves[stepid * Moves_pitch + gameid] - 1) : bix;
		//policy
		if (Random) {
			assert(0 <= Random[i] && Random[i] < 8);
			int mx = mov % 15, my = mov / 15;
			BoardRotated(7, 7, mx, my, Random[i]);
			mov = my * 15 + mx;
		}
		else assert(false);
		assert(0 <= mov && mov < 225);
		Board[mov * pitch + i] = One_Encode ? 1 : board_id[(bix + 1) * pitch + i];
		//value
		if (Value) {
			if (bix == 0)
				Value[i] = (stepid % 2) ? (-1 * Reward[gameid]) : Reward[gameid];
			assert(Value[i] == 0 || Value[i] == 1 || Value[i] == -1);
		}
	}
}
__global__ void Cuda_Matrix_Board_Simulation_Extend(int pitch, floatType* Out, floatType* Board, int col, int Gomoku_Planes) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		assert(Board[bix * pitch + i] == 0 || Board[bix * pitch + i] == 1 || Board[bix * pitch + i] == -1);
		//player/opponent/empty
		Out[(bix * Gomoku_Planes + (int)Board[bix * pitch + i] + 1) * pitch + i] = 1;
		//board boundary
		Out[(bix * Gomoku_Planes + 3) * pitch + i] = 1;
	}
}
const int type_num = 56 + 8 + 9;
__device__ const int DIR[8][2] = { {1,0},{0,1},{1,1},{-1,-1},{-1,0},{0,-1},{1,-1},{-1,1} };
__device__ const int KN_DIR[8][2] = { {1,-2},{2,-1},{2,1},{1,2},{-1,2},{-2,1},{-2,-1},{-1,-2} };
__device__ const int DIR_reverse[8] = { 4,5,3,2,0,1,7,6 };
__device__ const int KN_DIR_reverse[8] = { 4,5,6,7,0,1,2,3 };
__device__ void ACT(ui Action, floatType* Board, int pitch, int i) {
	const int W = 8;
	int type = Action % type_num, pos = Action / type_num;
	floatType& stone = Board[pos * pitch + i];
	//move
	if (type < 56) {
		int dir = type % 8, len = type / 8 + 1;
		int _x = pos % W + DIR[dir][0] * len, _y = pos / W + DIR[dir][1] * len;
		//underpromotions to queen
		if (stone == 1 && _x == 0)stone = 5;
		//castling
		if (stone == 6 && len == 2) {
			//left
			if (_y < pos / W)
				Board[7 * pitch + i] = 0, Board[(7 + W * (_y + 1)) * pitch + i] = 4;
			//right
			else Board[(7 + W * 7) * pitch + i] = 0, Board[(7 + W * (_y - 1)) * pitch + i] = 4;
		}
		//En passant
		if (stone == 1 && _y != pos / W && Board[(_x + W * _y) * pitch + i] == 0) {
			assert(Board[(_x + 1 + W * _y) * pitch + i] == -1);
			Board[(_x + 1 + W * _y) * pitch + i] = 0;
		}
		//move/capture
		Board[(_x + W * _y) * pitch + i] = stone;
	}
	//knight move
	else if (type < 56 + 8) {
		int dir = type - 56;
		int _x = pos % W + KN_DIR[dir][0], _y = pos / W + KN_DIR[dir][1];
		Board[(_x + W * _y) * pitch + i] = stone;
	}
	//underpromotions to other
	else {
		int ptype = (type - 56 - 8) % 3 + 2, d = (type - 56 - 8) / 3;
		int _x = pos % W - 1, _y = pos / W;
		if (d == 0);
		else if (d == 1) _y--;
		else _y++;
		assert(_x == 0);
		Board[(_x + W * _y) * pitch + i] = ptype;
	}
	stone = 0;
}
__device__ void StoneRotated(floatType& x, floatType& y) {
	x -= 3.5f, y -= 3.5f;
	x *= -1, y *= -1;
	x += 3.5f, y += 3.5f;
}
__device__ inline
void _swap(floatType& _Left, floatType& _Right)
{	// exchange values stored at _Left and _Right
	floatType _Tmp = _Left;
	_Left = _Right;
	_Right = _Tmp;
}
__device__ void rotateView(floatType* Board, int pitch, int k) {
	for (int i = 0; i < 64; i++)
		Board[i * pitch + k] *= -1;
	for (int i = 0; i < 8 / 2; i++)
		for (int j = 0; j < 8; j++) {
			floatType x = i, y = j;
			StoneRotated(x, y);
			_swap(Board[(i + 8 * j) * pitch + k], Board[(int)(x + 8 * y) * pitch + k]);
		}
}
__device__ void Chess_Board(int pitch, floatType* Out, floatType* Board, int Chess_Planes, int i, int t) {
	/*if (i == 0) {
		printf("\nt:%d\n", t);
	}*/
	for (int bix = 0; bix < 64; bix++) {
		int stone = Board[bix * pitch + i];
		//player 6 type stone
		if (stone > 0)
			Out[(bix * Chess_Planes + t * 14 + stone - 1) * pitch + i] = 1;
		//opponent 6 type stone
		else if (stone < 0)
			Out[(bix * Chess_Planes + t * 14 + 6 + stone + 6) * pitch + i] = 1;
		assert(abs(stone) <= 6);
		//Repetition count
		int repeat = Board[(64 + 2 * t) * pitch + i];
		assert(repeat >= 0);
		Out[(bix * Chess_Planes + t * 14 + 12 + ::min(repeat, 1)) * pitch + i] = 1;

		/*if (i == 0) {
			printf("%d ", stone);
			if (bix % 8 == 7)printf("\n");
		}*/
	}
}
__global__ void Cuda_Matrix_Chess_Representation_Decode(int pitch, floatType* Out, floatType* Board, int col, int Chess_Planes) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		for (int t = 0; t < 8; t++) {
			if (t > 0) {
				int act = Board[(64 + 1 + 2 * (t - 1)) * pitch + i];
				if (act != -1) {
					ACT(act, Board, pitch, i);
					rotateView(Board, pitch, i);
				}
			}
			int rep = Board[(64 + 2 * t) * pitch + i];
			if (rep != -1) {
				//rotate to host
				if (t % 2 == 0)rotateView(Board, pitch, i);
				Chess_Board(pitch, Out, Board, Chess_Planes, i, t);
				//restore
				if (t % 2 == 0)rotateView(Board, pitch, i);
			}
		}
		ui extra = Board[(64 + 8 * 2 - 1) * pitch + i];
		assert(extra <= FLT_Maximum_Int);
		for (int k = 0; k < 64; k++) {
			//colour
			Out[(k * Chess_Planes + 8 * 14) * pitch + i] = (bool)(extra & 1);
			//castling
			for (int j = 0; j < 4; j++)
				Out[(k * Chess_Planes + 8 * 14 + 1 + j) * pitch + i] = (bool)(extra & (1 << j + 1));
			//move count
			Out[(k * Chess_Planes + 8 * 14 + 5) * pitch + i] = ((extra >> 5) & ((1 << 10) - 1)) / 100.0f;
			//no progress count
			Out[(k * Chess_Planes + 8 * 14 + 6) * pitch + i] = (extra >> 15) / 100.0f;
		}
	}
}
__global__ void Cuda_Matrix_Mahjong_Reward_Representation_Decode(int pitch, floatType* Out, floatType* Board, int col,int Planes) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		const int Scr_Size = 41;
		int idx = 0, depth = 0;
		//base round info
		//extend to more previous round
		for (int k = 0; k < 2; k++) {
			//round wind number
			//-1 when the last round
			int round_number = Board[idx++ * pitch + i] + 1;
			assert(0 <= round_number && round_number < 9);
			for (int pos = 0; pos < Scr_Size; pos++)
				Out[(pos * Planes + depth + round_number) * pitch + i] = 1.0;
			//Out[(depth + round_number) * pitch + i] = 1.0;
			depth += 9;
			//four players scores
			for (int t = 0; t < 4; t++) {
				int val = Board[idx++ * pitch + i];
				//consider large range score later
				int Score = val / 25.0; floatType remain = val % 25;
				//printf("%d ", Score);
				assert(-20 < Score && Score < 40);
				Score = ::min(Score, 39);
				Score = ::max(Score, 0);
				remain = ::max(remain, 0.0f);
				remain /= 25.0;
				//printf("val:%d %f %f %f ", val, 1 - remain, remain, (1 - remain) * Score * 25 + remain * (Score + 1) * 25);
				Out[(Score * Planes + depth + 0) * pitch + i] = 1.0 - remain;
				Out[((Score + 1) * Planes + depth + 0) * pitch + i] = remain;
				depth++;
				/*Out[(depth + Score) * pitch + i] = 1.0 - remain;
				Out[(depth + Score + 1) * pitch + i] = remain;
				depth += 41;*/
				//remain /= 25.0;
				assert((val < 0 && Score == 0 && remain == 0) || (val >= 0 && abs((1 - remain) * Score * 25 + remain * (Score + 1) * 25 - val) < 1e-4));
			}
			//honba sticks
			int honba_sticks = Board[idx++ * pitch + i] / 2.0;
			assert(0<= honba_sticks);
			honba_sticks = ::min(honba_sticks, 10);
			//Out[(depth + honba_sticks) * pitch + i] = 1.0;
			for (int pos = 0; pos < Scr_Size; pos++)
				Out[(pos * Planes + depth + honba_sticks) * pitch + i] = 1.0;
			depth += 11;
			//riichi sticks
			int riichi_sticks = Board[idx++ * pitch + i];
			assert(0 <= riichi_sticks && riichi_sticks < 10);
			riichi_sticks = ::min(riichi_sticks, 10);
			//Out[(depth + riichi_sticks) * pitch + i] = 1.0;
			for (int pos = 0; pos < Scr_Size; pos++)
				Out[(pos * Planes + depth + riichi_sticks) * pitch + i] = 1.0;
			depth += 11;
		}
		assert(depth == 2 * (9 + 4 + 11 * 2));
		//assert(depth == 2 * (9 + 4 * 41 + 11 * 2));
		assert(idx == 2 * 7);
	}
}
#define Mahjong_ValuePlane(Value) \
{for (int pos = 0; pos < Scr_Size; pos++)Out[(pos * Planes + depth) * pitch + i] = (Value); \
depth++;} \

#define Mahjong_tiles_UnOrder_Plane(tiles_Count)\
{\
memset(Tile2Count, 0, sizeof(Tile2Count));\
for (int j = 0; j < (tiles_Count); j++) {\
	int tile = Board[idx++ * pitch + i];\
	assert(-1 <= tile && tile < 34);\
	if (tile != -1)Tile2Count[tile]++;\
}\
for (int j = 0; j < Scr_Size; j++) {\
	int Count = Tile2Count[j];\
	assert(0 <= Count && Count <= 4);\
	if (Count > 0)\
		for (int t = 0; t < Count; t++)\
			Out[(j * Planes + depth + t) * pitch + i] = 1.0;\
}\
depth += 4;}\


__global__ void Cuda_Matrix_Mahjong_Representation_Decode(int pitch, floatType* Out, floatType* Board, int col, int Planes) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		ui depth = 0, idx = 0;
		const int Scr_Size = (3 * 9 + 7) * 1;
		int Tile2Count[Scr_Size] = { 0 };
		for (int k = 0; k < 4; k++) {
			//players full drop tiles
			Mahjong_tiles_UnOrder_Plane(25);
			//tiles set
			Mahjong_tiles_UnOrder_Plane(16);
			//player with riichi
			floatType riichi = Board[idx++ * pitch + i];
			Mahjong_ValuePlane(riichi);
			//score
			//0 1 2 3 4 5 6 7 8 9 10 
			//more depth(x2)
			int Score = Board[idx++ * pitch + i] / 50.0, bucket = -1;
			assert(Score >= 0);
			////0-10
			//if (20 <= Score && Score <= 30) {
			//	bucket = Score - 20;
			//}
			////11-15
			//else if (Score < 20) {
			//	bucket = Score / 4 + 11;
			//}
			////16-20
			//else if (Score > 30) {
			//	bucket = (Score - 30) / 4 + 16;
			//	bucket = ::min(bucket, 20);
			//}
			//else assert(false);
			//printf("%d ", Score);
			assert(-10 < Score && Score < 20);
			Score = ::min(Score, 10);
			Score = ::max(Score, 0);
			for (int pos = 0; pos < Scr_Size; pos++)
				Out[(pos * Planes + depth + Score) * pitch + i] = 1.0;
			depth += 11;
		}
		//full player hand tiles
		for (int k = 0; k < 4; k++) {
			Mahjong_tiles_UnOrder_Plane(14);
		}
		////remain tiles count
		int remain_tiles = Board[idx++ * pitch + i] / 10.0;
		assert(0 <= remain_tiles && remain_tiles < 7);
		for (int pos = 0; pos < Scr_Size; pos++)
			Out[(pos * Planes + depth + remain_tiles) * pitch + i] = 1.0;
		depth += 7;
		//check tile
		int check_tile = Board[idx++ * pitch + i];
		assert(0 <= check_tile && check_tile < Scr_Size);
		Out[(check_tile * Planes + depth) * pitch + i] = 1.0;
		depth++;
		//chan kan
		floatType chan_kan = Board[idx++ * pitch + i];
		Mahjong_ValuePlane(chan_kan);
		//lastest PlayerID
		int Player = Board[idx++ * pitch + i];
		assert(0 <= Player && Player < 4);
		for (int pos = 0; pos < Scr_Size; pos++)
			Out[(pos * Planes + depth + Player) * pitch + i] = 1.0;
		depth += 4;
		//InsPlayer ID
		Player = Board[idx++ * pitch + i];
		assert(0 <= Player && Player < 4);
		//dealer ID
		Player = Board[idx++ * pitch + i];
		assert(0 <= Player && Player < 4);
		for (int pos = 0; pos < Scr_Size; pos++)
			Out[(pos * Planes + depth + Player) * pitch + i] = 1.0;
		depth += 4;
		//dora indicators(remove)
		memset(Tile2Count, 0, sizeof(Tile2Count));
		for (int j = 0; j < 5; j++) {
			int tile = Board[idx++ * pitch + i];
			if (tile != -1)Tile2Count[tile]++;
		}
		//round wind[0,4)
		int wind = Board[idx++ * pitch + i];
		assert(0 <= wind && wind < 4);
		for (int pos = 0; pos < Scr_Size; pos++)
			Out[(pos * Planes + depth + wind) * pitch + i] = 1.0;
		depth += 4;
		////remain least round number
		int remain_rounds = Board[idx++ * pitch + i];
		assert(0 <= remain_rounds && remain_rounds < 8);
		for (int pos = 0; pos < Scr_Size; pos++)
			Out[(pos * Planes + depth + ::min(remain_rounds, 7)) * pitch + i] = 1.0;
		depth += 8;
		//printf("%d ", remain_rounds);
		////riichi sticks
		int riichi_sticks = Board[idx++ * pitch + i];
		//assert(0 <= riichi_sticks);
		//for (int pos = 0; pos < Scr_Size; pos++)
		//	Out[(pos * Planes + depth + ::min(riichi_sticks, 4)) * pitch + i] = 1.0;
		//depth += 5;
		////honba sticks
		int honba_sticks = Board[idx++ * pitch + i];
		//assert(0 <= honba_sticks);
		//for (int pos = 0; pos < Scr_Size; pos++)
		//	Out[(pos * Planes + depth + ::min(honba_sticks, 4)) * pitch + i] = 1.0;
		//depth += 5;
		////furiten
		floatType riichi_furiten = Board[idx++ * pitch + i];
		Mahjong_ValuePlane(riichi_furiten);
		floatType temporary_furiten = Board[idx++ * pitch + i];
		Mahjong_ValuePlane(temporary_furiten);

		//recently wall tiles
		//dead wall dora tiles
		//rinshan tiles
		for (int j = 0; j < 24; j++) {
			int tile = Board[idx++ * pitch + i];
			assert(-1 <= tile && tile < 34);
			if (tile != -1)
				Out[(tile * Planes + depth + 0) * pitch + i] = 1.0;
			depth++;
		}
		//rinshan pop wall tiles
		Mahjong_tiles_UnOrder_Plane(4);
		//special action
		int action_type = Board[idx++ * pitch + i];
		//printf("%d ", action_type);
		assert(0 <= action_type && action_type < 4);
		for (int pos = 0; pos < Scr_Size; pos++)
			Out[(pos * Planes + depth + action_type) * pitch + i] = 1.0;
		depth += 4;

		assert(depth == 4 * (4 + 4 + 4 + 12) + 25 + 10 + 6 + 22);
		assert(idx == 4 * (25 + 4 * 4 + 2 + 14) + 6 + 9 + 2 + 10 + 10 + 4 + 4 + 1);
	}
}

__global__ void Cuda_Go_Action_Encode(int pitch, floatType* Out, floatType* A, int col, int A_Start_Row, ui ActionSpace, curandState* states, int states_pitch) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	ui No = bix * states_pitch + tix;
	for (int i = tix; i < col; i += bdx)
		if (A[A_Start_Row * pitch + i] != -1)
			Out[(int)A[A_Start_Row * pitch + i] * pitch + i] = 1;
	    //Uniform policy
		else Out[(curand(&states[No]) % ActionSpace) * pitch + i] = 1;
}
__device__ void PositionRotated(int&pos) {
	floatType x = pos % 8, y = pos / 8;
	StoneRotated(x, y);
	pos = x + y * 8;
}
__device__ void ActionRotated(int& action) {
	const int W = 8;
	assert(action < 64 * 73);
	int type = action % type_num, pos = action / type_num;
	floatType x = pos % 8, y = pos / 8;
	StoneRotated(x, y);
	pos = x + y * 8;
	assert(0 <= pos && pos < W* W);
	if (type < 56) {
		int dir = type % 8, len = type / 8 + 1;
		action = pos * type_num + (len - 1) * 8 + DIR_reverse[dir];
	}
	else if (type < 56 + 8) {
		int dir = type - 56;
		action = pos * type_num + KN_DIR_reverse[dir] + 56;
	}
	//underpromotions to other
	else {
		int ptype = (type - 56 - 8) % 3; int d = (type - 56 - 8) / 3, _d;
		if (d == 0)_d = 0;
		else if (d == 1) _d = 2;
		else if (d == 2) _d = 1;
		else assert(false);
		action = pos * type_num + _d * 3 + ptype + 56 + 8;
	}
	assert(action < 64 * 73);
}
__global__ void Cuda_Chess_Action_Encode(int pitch, floatType* Out, floatType* A, int col, ui simActionSpace, int Out_Plane, int W, floatType* SimActS2ActS, bool rotateAction, bool IsTrain, curandState* states, int states_pitch) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	ui No = bix * states_pitch + tix;
	for (int i = tix; i < col; i += bdx) {
		int act = A[i];
		//Uniform policy
		if (act == -1)act = SimActS2ActS[curand(&states[No]) % simActionSpace];
		assert(0 <= act < 64 * 73);
		//remove this code after train!
		if (rotateAction && IsTrain) {
			ActionRotated(act);
			assert(false);
		}

		//train with debug,to ensure data correct

		int type = act % type_num, pos = act / type_num, _pos, ptype = 5;
		if (type < 56) {
			int dir = type % 8, len = type / 8 + 1;
			int _x = pos % W + DIR[dir][0] * len, _y = pos / W + DIR[dir][1] * len;
			_pos = _x + _y * W;
		}
		else if (type < 56 + 8) {
			int dir = type - 56;
			int _x = pos % W + KN_DIR[dir][0], _y = pos / W + KN_DIR[dir][1];
			_pos = _x + _y * W;
		}
		//underpromotions to other
		else {
			ptype = (type - 56 - 8) % 3 + 2; int d = (type - 56 - 8) / 3;
			int _x = pos % W, _y = pos / W;
			if (rotateAction)_x++;// , assert(_x == 7);
			else _x--;// , assert(_x == 0);
			if (d == 0);
			else if (d == 1) _y--;
			else if (d == 2)_y++;
			else assert(false);
			_x = ::min(_x, 7), _y = ::min(_y, 7);
			_x = ::max(_x, 0), _y = ::max(_y, 0);
			_pos = _x + _y * W;
		}
		assert(0 <= pos && pos < W* W);
		assert(0 <= _pos && _pos < W* W);
		//move from
		Out[(pos * Out_Plane) * pitch + i] = 1;
		//move to
		Out[(_pos * Out_Plane + 1) * pitch + i] = 1;
		//underpromotion type(none,queen,knight,bishop,rook)
		//W==H
		for (int j = 0; j < W * W; j++)
			Out[(j * Out_Plane + ptype) * pitch + i] = 1;
		assert(2 <= ptype && ptype <= 5);
		/*if (i == 0)
			printf("%d f:%d,%d t:%d,%d--", act, pos % W, pos / W, _pos % W, _pos / W);*/
	}
}
__global__ void Cuda_Mahjong_Action_Encode(int pitch, floatType* Out, floatType* A, floatType* Values,int col, ui ActionSpace, int Out_Plane, curandState* states, int states_pitch) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	ui No = bix * states_pitch + tix;
	//only nextplayer==0 Action
	//const ui Count = 34 * (3 * 4 * 4 + 9);
	const int Scr_Size = (3 * 9 + 7) * 1;
	for (int i = tix; i < col; i += bdx) {
		int Action = A[i];
		assert(-1 <= Action < 34 * (3 * 4 * 4 + 9));
		//Uniform policy
		if (Action == -1) {
			Action = curand(&states[No]) % ActionSpace;
		}
		//generalize action
		//only contain tile information with hand tiles
		//try and test informly
		//to solve always pon/chi problem(special action Value problem)

		int type = Action % (3 * 4 * 4 + 9), tile = Action / (3 * 4 * 4 + 9);
		assert(0 <= tile && tile < 34);
		//Action tile
		//1discard+1pass+1shouminkan+9other
		int action_type = (type < 3 * 4 * 4) ? (type % 3) : (3 + type - 3 * 4 * 4);
		assert(0 <= action_type && action_type < 12);
		//Out[(tile * Out_Plane + 0) * pitch + i] = (action_type + 1) / 12.0;
		Out[(tile * Out_Plane + 0) * pitch + i] = 1;

		//action type
		for (int u = 0; u < Scr_Size; u++)Out[(u * Out_Plane + 1 + action_type) * pitch + i] = 1;

		//?
		//test use binary planes for special actions
		//extra riichi binary planes instead
		
		//check
		int check = (type < 3 * 4 * 4) ? (type % (3 * 4) / 3) : 0;
		assert(0 <= check && check < 4);
		for (int u = 0; u < Scr_Size; u++)Out[(u * Out_Plane + 13 + check) * pitch + i] = 1;
		//next player
		int next = (type < 3 * 4 * 4) ? (type / (3 * 4)) : 0;
		assert(0 <= next && next < 4);
		for (int u = 0; u < Scr_Size; u++)Out[(u * Out_Plane + 17 + next) * pitch + i] = 1;

		//draw flag
		/*bool draw_tile = false;
		if (check == 0 && (action_type <= 3 || action_type == 8))
			draw_tile = true;
		for (int u = 0; u < Scr_Size; u++)Out[(u * Out_Plane + (14 + draw_tile)) * pitch + i] = 1;*/

		//dynamic rotate values
		if (Values) {
			floatType vals[4];
			for (int j = 0; j < 4; j++)vals[j] = Values[j * pitch + i];
			for (int j = 0; j < 4; j++)Values[j * pitch + i] = vals[(j + next) % 4];
		}
	}

}
__global__ void Cuda_Mahjong_Policy_Encode(int pitch, floatType* Out, floatType* A, int col, int A_row) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	//only nextplayer==0 Action
	ui gdx = gridDim.x;
	for (int i = tix; i < col; i += bdx) {
		//Uniform Policy
		if (A[i] == -1) {
			//if (3 * 4 <= bix % (3 * 4 * 4 + 9) && bix % (3 * 4 * 4 + 9) < 3 * 4 * 2)
			Out[bix * pitch + i] = 1.0f / gdx;
		}
		else if (bix < A_row && A[(bix * 2 + 1) * pitch + i] != 0) {
			//used for first train
			int _i = 0;
			while (A[(_i * 2) * pitch + i] != 0)_i++;
			assert(0 < _i && _i < A_row&& _i>bix);
			Out[(ui)A[(bix * 2) * pitch + i] * pitch + i] = 1.0f / _i;// A[(bix * 2 + 1) * pitch + i];
		}
	}
}
__global__ void Cuda_Mahjong_Simplify_Policy_Encode(int pitch, floatType* Out, floatType* A, int col, int A_row) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	ui gdx = gridDim.x;
	for (int i = tix; i < col; i += bdx) {
		assert(A[i] != -1);
		if (bix < A_row && A[(bix * 2 + 1) * pitch + i] != 0) {
			int idx = A[(bix * 2) * pitch + i];
			int type = idx % 57, tile = idx / 57;
			assert(0 <= tile && tile < 34);
			if (type < 48) {
				type = type % 3;
			}
			else type = 3 + type - 48;
			Out[(tile * 12 + type) * pitch + i] = A[(bix * 2 + 1) * pitch + i];
		}
	}
}
__global__ void Cuda_Chess_Policy_Encode(int pitch, floatType* Out, floatType* A, int col, floatType* SimActS2ActS, int A_row,bool rotateAction) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	int gdx = gridDim.x;
	for (int i = tix; i < col; i += bdx) {
		//Uniform Policy
		if (A[i] == -1)Out[(ui)SimActS2ActS[bix] * pitch + i] = 1.0f / gdx;
		else if (bix < A_row && A[(bix * 2 + 1) * pitch + i] != 0) {
			int act = A[(bix * 2) * pitch + i];
			if (rotateAction)
				ActionRotated(act);
			Out[act * pitch + i] = A[(bix * 2 + 1) * pitch + i];
			assert(act < 64 * 73);
		}
	}
}
__global__ void Cuda_Mahjong_Values_Encode(int pitch, floatType* Out, int col) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;

	//for (int i = tix; i < col; i += bdx)Out[bix * pitch + i] = Out[bix * pitch + i] == 0 ? 0 : (Out[bix * pitch + i] / abs(Out[bix * pitch + i]));
	__shared__ extern floatType _shared[];
	_shared[tix * 2] = -1e9; _shared[tix * 2 + 1] = 1e9;
	for (int i = tix; i < col; i += bdx)_shared[tix * 2] = ::max(_shared[tix * 2], Out[bix * pitch + i]), _shared[tix * 2 + 1] = ::min(_shared[tix * 2 + 1], Out[bix * pitch + i]);
	_syncthreads();
	if (tix == 0) {
		for (int i = 1; i < bdx; i++)_shared[0] = ::max(_shared[0], _shared[i * 2]), _shared[1] = ::min(_shared[1], _shared[i * 2 + 1]);
	}
	_syncthreads();
	assert(_shared[0] >= 25000 && _shared[1] <= 25000);
	//if (_shared[0] > eps) {
	for (int i = tix; i < col; i += bdx) {
		floatType& val = Out[bix * pitch + i];
		val -= 25000;
		if (val > 0)val /= abs(_shared[0] - 25000);
		else if (val < 0)val /= abs(_shared[1] - 25000);
		assert(-1 <= val && val <= 1);
	}
}
__global__ void Cuda_Mahjong_reward_Softmax_Encode(int pitch, floatType* Out, floatType* A, int col, int idx) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		floatType reward = A[idx * pitch + i];
		//printf("idx:%d %f$", idx, reward);
		//win
		if (abs(reward - 1) <= 1e-8) {
			Out[1 * pitch + i] = 1;
			Out[0 * pitch + i] = 0;
		}
		//loss
		else if (abs(reward + 1) <= 1e-8) {
			Out[1 * pitch + i] = 0;
			Out[0 * pitch + i] = 1;
		}
		//between win and draw
		else {
			assert(abs(reward) < 1);
			Out[1 * pitch + i] = 0.5;
			Out[0 * pitch + i] = 0.5;
		}
	}
}
__global__ void Cuda_Matrix_Mahjong_Agent_Representation_Decode(int pitch, floatType* Out, floatType* Board, int col, int Planes) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		ui depth = 0, idx = 0;
		const int Scr_Size = (3 * 9 + 7) * 1;
		int Tile2Count[Scr_Size] = { 0 };
		for (int k = 0; k < 4; k++) {
			//players full drop tiles
			Mahjong_tiles_UnOrder_Plane(25);
			//tiles set
			Mahjong_tiles_UnOrder_Plane(16);
			//player with riichi
			floatType riichi = Board[idx++ * pitch + i];
			Mahjong_ValuePlane(riichi);
			//score
			//0 1 2 3 4 5 6 7 8 9 10 
			//more depth(x2)
			int Score = Board[idx++ * pitch + i] / 50.0, bucket = -1;
			assert(Score >= 0);
			assert(-10 < Score && Score < 20);
			Score = ::min(Score, 10);
			Score = ::max(Score, 0);
			/*for (int pos = 0; pos < Scr_Size; pos++)
				Out[(pos * Planes + depth + Score) * pitch + i] = 1.0;*/
			depth += 11;
		}
		//full player hand tiles
		for (int k = 0; k < 4; k++) {
			if (k == 0) {
				Mahjong_tiles_UnOrder_Plane(14);
			}
			else {
				idx += 14; depth += 4;
			}
		}
		////remain tiles count
		int remain_tiles = Board[idx++ * pitch + i] / 10.0;
		assert(0 <= remain_tiles && remain_tiles < 7);
		for (int pos = 0; pos < Scr_Size; pos++)
			Out[(pos * Planes + depth + remain_tiles) * pitch + i] = 1.0;
		depth += 7;
		//check tile
		int check_tile = Board[idx++ * pitch + i];
		assert(0 <= check_tile && check_tile < Scr_Size);
		Out[(check_tile * Planes + depth) * pitch + i] = 1.0;
		depth++;
		//chan kan
		floatType chan_kan = Board[idx++ * pitch + i];
		Mahjong_ValuePlane(chan_kan);
		//lastest PlayerID
		int Player = Board[idx++ * pitch + i];
		assert(0 <= Player && Player < 4);
		for (int pos = 0; pos < Scr_Size; pos++)
			Out[(pos * Planes + depth + Player) * pitch + i] = 1.0;
		depth += 4;
		//InsPlayer ID
		Player = Board[idx++ * pitch + i];
		assert(0 <= Player && Player < 4);
		//dealer ID
		Player = Board[idx++ * pitch + i];
		assert(0 <= Player && Player < 4);
		for (int pos = 0; pos < Scr_Size; pos++)
			Out[(pos * Planes + depth + Player) * pitch + i] = 1.0;
		depth += 4;
		//dora indicators(remove)
		Mahjong_tiles_UnOrder_Plane(5);
		//round wind[0,4)
		int wind = Board[idx++ * pitch + i];
		assert(0 <= wind && wind < 4);
		/*for (int pos = 0; pos < Scr_Size; pos++)
			Out[(pos * Planes + depth + wind) * pitch + i] = 1.0;*/
		depth += 4;
		////remain least round number
		int remain_rounds = Board[idx++ * pitch + i];
		assert(0 <= remain_rounds && remain_rounds < 8);
		/*for (int pos = 0; pos < Scr_Size; pos++)
			Out[(pos * Planes + depth + ::min(remain_rounds, 7)) * pitch + i] = 1.0;*/
		depth += 8;
		//printf("%d ", remain_rounds);
		////riichi sticks
		int riichi_sticks = Board[idx++ * pitch + i];
		//assert(0 <= riichi_sticks);
		//for (int pos = 0; pos < Scr_Size; pos++)
		//	Out[(pos * Planes + depth + ::min(riichi_sticks, 4)) * pitch + i] = 1.0;
		//depth += 5;
		////honba sticks
		int honba_sticks = Board[idx++ * pitch + i];
		//assert(0 <= honba_sticks);
		//for (int pos = 0; pos < Scr_Size; pos++)
		//	Out[(pos * Planes + depth + ::min(honba_sticks, 4)) * pitch + i] = 1.0;
		//depth += 5;
		////furiten
		floatType riichi_furiten = Board[idx++ * pitch + i];
		Mahjong_ValuePlane(riichi_furiten);
		floatType temporary_furiten = Board[idx++ * pitch + i];
		Mahjong_ValuePlane(temporary_furiten);

		//recently wall tiles
		//dead wall dora tiles
		//rinshan tiles
		for (int j = 0; j < 28; j++) {
			int tile = Board[idx++ * pitch + i];
			assert(-1 <= tile && tile < 34);
			/*if (tile != -1)
				Out[(tile * Planes + depth + 0) * pitch + i] = 1.0;*/
			depth++;
		}
		//rinshan pop wall tiles
		//Mahjong_tiles_UnOrder_Plane(4);
		//special action
		int action_type = Board[idx++ * pitch + i];
		//printf("%d ", action_type);
		assert(0 <= action_type && action_type < 4);
		for (int pos = 0; pos < Scr_Size; pos++)
			Out[(pos * Planes + depth + action_type) * pitch + i] = 1.0;
		depth += 4;

		assert(depth == 4 * (4 + 4 + 4 + 12) + 25 + 10 + 6 + 22 + 4);
		assert(idx == 4 * (25 + 4 * 4 + 2 + 14) + 6 + 9 + 2 + 10 + 10 + 4 + 4 + 1);
	}
}
__global__ void Cuda_Matrix_reward_sample(int pitch, floatType* Out, int col, curandState* states, int states_pitch) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	ui No = bix * states_pitch + tix;
	const ui MOD = 1000;
	for (int i = tix; i < col; i += bdx) {
		double val = 1.0 * (curand(&states[No]) % MOD) / MOD;
		floatType& reward = Out[bix * pitch + i];
		if (val < reward)reward = 1.0;
		else reward = -1.0;
	}
}
__global__ void Cuda_Matrix_random(int pitch, floatType* A, int col, curandState* states, int states_pitch, ui Mod) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	ui No = bix * states_pitch + tix;
	for (int i = tix; i < col; i += bdx) {
		assert(Mod <= FLT_Maximum_Int);
		A[bix * pitch + i] = curand(&states[No]) % Mod;
	}
}
__global__ void Cuda_Matrix_rand_order(int A_pitch, int B_pitch, floatType* A, floatType* B, int A_col, int B_col, floatType* Order) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < A_col; i += bdx) {
		assert(Order[i] < FLT_Maximum_Int);
		A[bix * A_pitch + i] = B[bix * B_pitch + ((ui)Order[i] % B_col)];
	}
}
__global__ void Cuda_Matrix_rand_order_(int A_pitch, int B_pitch, floatType* A, floatType* B, int A_col, int B_col, floatType* Order) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < B_col; i += bdx) {
		assert(Order[i] < FLT_Maximum_Int);
		A[bix * A_pitch + ((ui)Order[i] % A_col)] = B[bix * B_pitch + i];
	}
}
__global__ void Cuda_Matrix_Negative_Sampling(int A_pitch, int B_pitch, int C_pitch, floatType* A, floatType* B, floatType* Range, int A_col, int Range_col, floatType* Order, int Order_col, floatType* Unigram_Table, int U_col, curandState* states, int states_pitch) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	ui No = bix * states_pitch + tix;
	for (int i = tix + Order_col; i < A_col; i += bdx) {
		ui rand = curand(&states[No]);
		//binary search find random negative sample
		double tar = 1.0f * rand / 0xffffffff;
		long long l = 0, r = Range_col - 1;
		while (l < r) {
			long long M = l + r >> 1;
			if (Unigram_Table[M] < tar)l = M + 1;
			else r = M;
		}
		A[bix * A_pitch + i] = r;
	}
}
__global__ void Cuda_Matrix_Positive_Sampling(floatType* Order, floatType* P, floatType* Word_Context, floatType* Range, floatType* Neg, int Order_col, int Range_col, curandState* states, int states_pitch) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	ui No = bix * states_pitch + tix;
	for (int i = tix; i < Order_col; i += bdx) {
		unsigned long long rand = ((unsigned long long)curand(&states[No]) << 32) + curand(&states[No]);
		//binary search find random Word-Context sample
		double tar = 1.0f * rand / 0xffffffffffffffff;
		long long l = 0, r = Range[Range_col - 1] - 1;
		while (l < r) {
			long long M = l + r >> 1;
			if (P[M] < tar)l = M + 1;
			else r = M;
		}
		long long idx = r;
		//Word
		Order[i] = (long long)Word_Context[idx] / Range_col;
		//Context(Positive)
		Neg[i] = (long long)Word_Context[idx] % Range_col;
	}
}
__global__ void Cuda_Matrix_Generate_OutPutMask(int A_pitch, floatType* A, floatType* mask, floatType* mask1, floatType* Neg, int A_col, floatType* Order, floatType* Word_Context, int C_pitch, int C_col, int C_row, floatType* Range, int Range_col) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < A_col; i += bdx) {
		long long tar = Neg[bix]; //ui _col = ((ui)Order[i] % D_col);
		long long l = 0, r = Range[(ui)Order[i]]; if ((ui)Order[i] > 0)l = Range[(ui)Order[i] - 1];
		while (l < r) {
			long long M = l + r >> 1;
			if (((long long)Word_Context[M] % Range_col) <= tar)l = M + 1;
			else r = M;
		}
		if (l - 1 < 0 || ((ui)Order[i] > 0 && l - 1 < Range[(ui)Order[i] - 1]) || ((long long)Word_Context[l - 1] % Range_col) != tar)A[bix * A_pitch + i] = 0, mask[bix * A_pitch + i] = mask1[bix * A_pitch + i] = 1;
		else A[bix * A_pitch + i] = 1, mask[bix * A_pitch + i] = 0, mask1[bix * A_pitch + i] = -1;
	}
}
__global__ void Cuda_Matrix_mul_Matrix_(int A_pitch, int B_pitch, floatType* A, floatType* B, int col) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		A[bix * A_pitch + i] *= B[bix * B_pitch];
	}
}
__global__ void Cuda_Matrix_mul_Matrix__(int A_pitch, int B_pitch, floatType* A, floatType* B, int col) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		A[bix * A_pitch + i] *= B[i];
	}
}
__global__ void Cuda_Matrix_plus(int pitch, floatType* A, floatType* B, floatType* Out, int col) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	//double*_out = &Out[bix*pitch], *_a = &A[bix*pitch], *_b = &B[bix*pitch];
	for (int i = tix; i < col; i += bdx) {
		Out[bix * pitch + i] = A[bix * pitch + i] + B[bix * pitch + i];
	}
}

//col expand
__global__ void Cuda_Matrix_col_expand(int A_pitch, int B_pitch, int Max_pitch, floatType* A, floatType* B, floatType* Out, int A_col, int B_col, int Max_col, const int sign) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < Max_col; i += bdx) {
		Out[bix * Max_pitch + i] = A[bix * A_pitch + ::min(i, A_col - 1)] + sign * B[bix * B_pitch + ::min(i, B_col - 1)];
	}
}
//row expand
__global__ void Cuda_Matrix_row_expand(int A_pitch, int B_pitch, int Out_pitch, floatType* A, floatType* B, floatType* Out, int A_row, int B_row, int Out_col, const int sign) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < Out_col; i += bdx) {
		Out[bix * Out_pitch + i] = A[::min(bix, (ui)A_row - 1) * A_pitch + i] + sign * B[::min(bix, (ui)B_row - 1) * B_pitch + i];
	}
}
__global__ void Cuda_Matrix_plus__(int pitch, floatType* A, floatType* B, int col) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		A[bix * pitch + i] += B[bix * pitch + i];
	}
}
__global__ void Cuda_Matrix_minus(int pitch, floatType* A, floatType* B, floatType* Out, int col) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		Out[bix * pitch + i] = A[bix * pitch + i] - B[bix * pitch + i];
	}
}
__global__ void Cuda_Matrix_minus_(int pitch, floatType* A, floatType* B, int col) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		A[bix * pitch + i] -= B[bix * pitch + i];
	}
}
//303ms -->21ms
__global__ void Cuda_Matrix_scale_col(int A_pitch, int Out_pitch, floatType* A, floatType* Out, int A_col) {
	//double sum = 0;
	__shared__ extern floatType _shared[];
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	_shared[tix] = 0;
	for (int i = tix; i < A_col; i += bdx) {
		_shared[tix] += A[bix * A_pitch + i];
	}
	_syncthreads();

	if (tix == 0) {
		floatType& sum = _shared[tix];
		for (int i = 1; i < bdx; i++)
			sum += _shared[i];
		Out[bix * Out_pitch] = sum;
	}
}
//270ms
//__global__ void Cuda_Matrix_T(int A_pitch, int Out_pitch, double*A, double*Out, int Out_col) {
//	//extern __shared__ double _shared[];
//	ui tix = threadIdx.x;
//	//int tiy = threadIdx.y;
//	ui bix = blockIdx.x;
//	//int biy = blockIdx.y;
//	ui bdx = blockDim.x;
//	//int bdy = blockDim.y;
//
//	/*
//	int gdx = gridDim.x;
//	int gdy = gridDim.y;
//
//
//	int _Col = min(bdy, A_col - bix * bdy);
//	int _Row = min(bdy, Out_col - biy * bdy);
//
//	if (_Col <= 0 || _Row <= 0)return;
//	//起始坐标
//	int _x = bix * bdy, _y = biy * bdy;
//
//	if (_y + tiy < Out_col)
//		for (int i = 0; i < _Col; i++)
//			_shared[tiy*_Col + i] = A[(_y + tiy)*A_pitch + _x+i];
//
//	__syncthreads();
//
//
//	if (_x + tiy < A_col)
//		for (int i = 0; i < _Row; i++)
//			Out[(_x + tiy)*Out_pitch + _y + i] = _shared[i*_Col + tiy];
//
//	return;*/
//
//	double*out = &Out[bix*Out_pitch];
//	for (int i = tix; i < Out_col; i += bdx) {
//		out[i] = A[i*A_pitch + bix];
//	}
//}
__global__ void Cuda_Matrix_function(int pitch, floatType* A, floatType* Out, int col, _fun device_f) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (ui i = tix; i < col; i += bdx) {
		Out[bix * pitch + i] = device_f(A[bix * pitch + i]);
	}
}
__global__ void Cuda_Matrix_function_param(int pitch, floatType* A, floatType* Out, int col, _fun2 device_f, const floatType param) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (ui i = tix; i < col; i += bdx) {
		Out[bix * pitch + i] = device_f(A[bix * pitch + i], param);
	}
}
__global__ void Cuda_Matrix_Scale_Image(int pitch, floatType* dst, int col, floatType* image, int Image_W, int Image_H, int Image_Depth, floatType* Location, int Scale_Image_WH) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	int sx = (bix / Image_Depth) % Scale_Image_WH, sy = bix / (Image_Depth * Scale_Image_WH) % Scale_Image_WH;
	int Scale = 1, shift = Scale_Image_WH;
	int scales = bix / (Scale_Image_WH * Scale_Image_WH * Image_Depth);
	while (scales--)Scale *= 2, shift *= 2;
	shift = (shift - 1) / 2;
	int depth = bix % Image_Depth;
	//faster
	for (ui i = tix; i < col; i += bdx) {
		floatType Sum = 0;
		int cx = Location[i] - shift, cy = Location[pitch + i] - shift;
		for (int j = 0; j < Scale; j++)
			for (int k = 0; k < Scale; k++) {
				int x1 = cx + sx * Scale + j, y1 = cy + sy * Scale + k;
				if (x1 < 0 || y1 < 0 || x1 >= Image_W || y1 >= Image_H)continue;
				Sum += image[pitch * ((y1 * Image_W + x1) * Image_Depth + depth) + i];
			}
		Sum /= Scale * Scale;
		dst[pitch * bix + i] = Sum;// (Sum == 0) ? 0 : (Sum > 0 ? 1.0 : -1.0);
	}
}
//small Matrix ===> large Matrix(usually OK,overlapping wrong)
//large Matrix ===> small Matrix(usually not correct)!!! (same region one thread)
__global__ void Cuda_Max_Pooling(int Image_pitch, int Out_pitch, floatType* Image, int Image_Depth, int Image_W, int Image_H, int F, int col, int dst_W, int dst_H, int stride, int padding, floatType* Out, floatType* Out_idx) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	int x = (bix / Image_Depth) % dst_W, y = (bix / Image_Depth) / dst_W;
	int depth = bix % Image_Depth;
	for (ui i = tix; i < col; i += bdx) {
		floatType& tar = Out[bix * Out_pitch + i];
		floatType& idx = Out_idx[bix * Out_pitch + i];
		for (int j = 0; j < F; j++)
			for (int k = 0; k < F; k++) {
				int _x = x * stride - padding + k, _y = y * stride - padding + j;
				if (-1 < _x && -1 < _y && _x < Image_W && _y < Image_H) {
					int _idx = Image_Depth * Image_W * _y + Image_Depth * _x + depth;
					floatType& img = Image[_idx * Image_pitch + i];
					//max value pooling
					if (img > tar || idx == -1) {
						tar = img;
						//save idx
						idx = _idx;
					}
				}
			}
	}
}
__global__ void Cuda_Max_Pooling_bp(int Image_pitch, int Out_pitch, floatType* Image, int Image_Depth, int Image_W, int Image_H, int F, int col, int dst_W, int dst_H, int stride, int padding, floatType* Out, floatType* Out_idx) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	int x = (bix / Image_Depth) % Image_W, y = (bix / Image_Depth) / Image_W;
	int depth = bix % Image_Depth;
	int lx = floor(x + padding - F, stride) + 1, rx = floor(x + padding, stride);
	lx = lx > 0 ? lx : 0; rx = rx < (dst_W - 1) ? rx : (dst_W - 1);
	int ly = floor(y + padding - F, stride) + 1, ry = floor(y + padding, stride);
	ly = ly > 0 ? ly : 0; ry = ry < (dst_H - 1) ? ry : (dst_H - 1);
	for (ui i = tix; i < col; i += bdx) {
		for (int j = ly; j <= ry; j++)
			for (int k = lx; k <= rx; k++) {
				ui idx = (Image_Depth * dst_W * j + Image_Depth * k + depth) * Out_pitch + i;
				//max value pooling
				assert(Out_idx[idx] < FLT_Maximum_Int);
				if (Out_idx[idx] == bix) {
					floatType& img = Image[bix * Image_pitch + i];
					floatType& tar = Out[idx];
					img += tar;
				}
			}
	}
}
//__global__ void Cuda_im2col(int Image_pitch, int Out_pitch, double* Image, int Image_Depth, int Image_W, int Image_H, int F, int col, int dst_W, int dst_H, int stride, int padding, double* Out) {
//	float reg[4]; int _i = 0;
//	ui tix = threadIdx.x + blockIdx.y * blockDim.x;
//	ui bix = blockIdx.x;
//	ui bd = blockDim.x * gridDim.y;
//	int lx = (bix / Image_Depth) % F, ly = (bix / Image_Depth) / F;
//	int WH = dst_W * dst_H;
//	for (ui i = tix; i < col; i += bd) {
//		int x = i % dst_W, y = (i % WH) / dst_W;
//		int _x = x * stride - padding + lx, _y = y * stride - padding + ly;
//		int b = i / WH;
//		//faster?
//		if (-1 < _x && -1 < _y && _x < Image_W && _y < Image_H) {
//			//Out[bix * Out_pitch + i] = 
//			int _a = Image_Depth * Image_W * _y, _b = Image_Depth * _x;
//			reg[_i++] = Image[(_a + _b + bix % Image_Depth) * Image_pitch + b];
//		}//out of boundary zero padding
//		else {
//			//Out[bix * Out_pitch + i] = 0;
//			reg[_i++] = 0;
//		}
//	}
//	_i = 0; for (ui i = tix; i < col; i += bd, _i++)Out[bix * Out_pitch + i] = reg[_i];
//}
__global__ void Cuda_im2col(int Image_pitch, int Out_pitch, floatType* Image, int Image_Depth, int Image_W, int Image_H, int F, int col, int dst_W, int dst_H, int stride, int padding, floatType* Out) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	int lx = (bix / Image_Depth) % F, ly = (bix / Image_Depth) / F;
	int depth = bix % Image_Depth;
	for (ui i = tix; i < col; i += bdx) {
		int x = (i % (dst_W * dst_H)) % dst_W, y = (i % (dst_W * dst_H)) / dst_W;
		int _x = x * stride - padding + lx, _y = y * stride - padding + ly;
		int b = i / (dst_W * dst_H);
		//faster?
		if (-1 < _x && -1 < _y && _x < Image_W && _y < Image_H) {
			Out[bix * Out_pitch + i] = Image[(Image_Depth * Image_W * _y + Image_Depth * _x + depth) * Image_pitch + b];
		}//out of boundary zero padding
		else {
			Out[bix * Out_pitch + i] = 0.0f;
		}
	}
}
//__global__ void Cuda_im2col(int Image_pitch, int Out_pitch, double*Image, int Image_Depth, int Image_W, int Image_H, int F, int col, int dst_W, int dst_H, int stride, int padding, double*Out) {
//	ui tix = threadIdx.x;
//	ui bix = blockIdx.x;
//	ui bdx = blockDim.x;
//	int lx = (bix / Image_Depth) % F, ly = (bix / Image_Depth) / F;
//	int depth = bix % Image_Depth, Size = dst_W * dst_H, cnt = 0;
//	static const int register_num = 8;
//	float reg[register_num];
//	//fetch 4 to register
//	for (ui i = tix; i < col; i += bdx) {
//		int x = i % dst_W, y = (i % Size) / dst_W;
//		int _x = x * stride - padding + lx, _y = y * stride - padding + ly;
//		//faster?
//		if (-1 < _x && -1 < _y && _x < Image_W && _y < Image_H)
//			reg[cnt++] = Image[(Image_Depth * Image_W * _y + Image_Depth * _x + depth) * Image_pitch + i / Size];
//		//out of boundary zero padding
//		else reg[cnt++] = 0;
//		//register to memory
//		if (cnt == register_num || i + bdx >= col) {
//			for (int j = 0; j < cnt; j++)
//				Out[bix * Out_pitch + i - bdx * j] = reg[cnt - 1 - j];
//			cnt = 0;
//		}
//	}
//}
__global__ void Cuda_im2col_bp(int Image_pitch, int Out_pitch, floatType* Image, int Image_Depth, int Image_W, int Image_H, int F, int col, int dst_W, int dst_H, int stride, int padding, floatType* Out) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	int x = (bix / Image_Depth) % Image_W, y = (bix / Image_Depth) / Image_W;
	int depth = bix % Image_Depth;
	int lx = floor(x + padding - F, stride) + 1, rx = floor(x + padding, stride);
	lx = lx > 0 ? lx : 0; rx = rx < (dst_W - 1) ? rx : (dst_W - 1);
	int ly = floor(y + padding - F, stride) + 1, ry = floor(y + padding, stride);
	ly = ly > 0 ? ly : 0; ry = ry < (dst_H - 1) ? ry : (dst_H - 1);
	for (ui i = tix; i < col; i += bdx)
		for (int j = ly; j <= ry; j++)
			for (int k = lx; k <= rx; k++) {
				int _x = k * stride - padding, _y = j * stride - padding;
				_x = x - _x, _y = y - _y;
				Image[bix * Image_pitch + i] += Out[(Image_Depth * F * _y + Image_Depth * _x + depth) * Out_pitch + dst_W * dst_H * i + dst_W * j + k];
			}
}
__global__ void Cuda_im2col_Restore(int pitch, int Image_pitch, int row, floatType* im2col, floatType* Image, int Image_col, int Image_W, int Image_H) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	ui x = (bix / row) % Image_W, y = (bix / row) / Image_W * Image_W + x;
	ui depth = bix % row;
	for (ui i = tix; i < Image_col; i += bdx) {
		Image[bix * Image_pitch + i] = im2col[depth * pitch + i * Image_W * Image_H + y];
	}
}
__global__ void Cuda_im2col_Restore_bp(int pitch, int Image_pitch, int row, floatType* im2col, floatType* Image, int Image_col, int Image_W, int Image_H) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	ui x = (bix / row) % Image_W, y = (bix / row) / Image_W * Image_W + x;
	ui depth = bix % row;
	for (ui i = tix; i < Image_col; i += bdx) {
		im2col[depth * pitch + i * Image_W * Image_H + y] = Image[bix * Image_pitch + i];
	}
}
__global__ void Cuda_Image_SpatialConcatenate(int pitch, int col, floatType* A, floatType* B, floatType* Out, int Out_filters, int A_filters) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	int pos = bix / Out_filters, f = bix % Out_filters;
	int B_filters = Out_filters - A_filters;
	for (ui i = tix; i < col; i += bdx) {
		Out[bix * pitch + i] = f < A_filters ? A[(pos * A_filters + f) * pitch + i] : B[(pos * B_filters + (f - A_filters)) * pitch + i];
	}
}
__global__ void Cuda_Image_SpatialConcatenate_bp(int pitch, int col, floatType* A, floatType* Out, int Out_filters, int A_filters, bool first) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	int pos = bix / Out_filters, f = bix % Out_filters;
	int B_filters = A_filters - Out_filters;
	for (ui i = tix; i < col; i += bdx) {
		Out[bix * pitch + i] = first ? A[(pos * A_filters + f) * pitch + i] : A[(pos * A_filters + (f + B_filters)) * pitch + i];
	}
}
__global__ void Cuda_BN_Normalization(int pitch, int col, floatType* A, floatType* Mean, floatType* Var, floatType* Out, int Mean_pitch) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (ui i = tix; i < col; i += bdx) {
		Out[bix * pitch + i] = (A[bix * pitch + i] - Mean[bix * Mean_pitch]) / sqrt(Var[bix * Mean_pitch]);
	}
}
__global__ void Cuda_Matrix_function_random(int pitch, floatType* A, floatType* Out, int col, _fun2 device_f, floatType param, curandState* states, int states_pitch) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	ui No = bix * states_pitch + tix;
	for (ui i = tix; i < col; i += bdx) {
		Out[bix * pitch + i] = device_f(param, 1.0f * (curand(&states[No]) % (ui)0xffffff) / 0xfffffe);
	}
}
__global__ void Cuda_Matrix_One_To_No(int pitch, floatType* A, int col) {
	__shared__ floatType _shared;
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (ui i = tix; i < col; i += bdx) {
		if (A[bix * pitch + i] == 1)
			_shared = i;
	}
	_syncthreads();
	if (tix == 0)A[bix * pitch] = _shared;
}
__global__ void Cuda_Matrix_mul_Row(int A_pitch, floatType* A, floatType* B, int col) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	for (int i = tix; i < col; i += bdx) {
		A[bix * A_pitch + i] *= B[i];
	}
}
__global__ void Cuda_Matrix_SoftMax_(int pitch, floatType* A, int col, bool Max) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	__shared__ extern floatType _shared[];
	_shared[tix] = -FLT_MAX;
	for (int i = tix; i < col; i += bdx) {
		if (A[bix * pitch + i] > _shared[tix])
			_shared[tix] = A[bix * pitch + i];
	}
	_syncthreads();
	//find col Max
	if (tix == 0) {
		for (int i = 1; i < bdx; i++)
			if (_shared[i] > _shared[0])_shared[0] = _shared[i];
	}
	_syncthreads();
	if (Max)
		for (ui i = tix; i < col; i += bdx) {
			A[bix * pitch + i] = (A[bix * pitch + i] == _shared[0]) ? 1 : 0;
		}
	else
		for (ui i = tix; i < col; i += bdx) {
			A[bix * pitch + i] = exp(A[bix * pitch + i] - _shared[0]);
		}
}
__global__ void Cuda_Matrix_SoftMax(int pitch, floatType* A, int col, floatType* col_Sum, int Sum_pitch) {
	__shared__ extern floatType Sum[];
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	Sum[0] = col_Sum[bix * Sum_pitch];
	assert(Sum[0] >= 1);
	for (ui i = tix; i < col; i += bdx) {
		A[bix * pitch + i] /= Sum[0];
		assert(abs(A[bix * pitch + i]) <= 1);
	}
}
__global__ void Cuda_Matrix_MinMax_Normalization(int pitch, floatType* A, int col, floatType* factor) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	__shared__ extern floatType _shared[];
	_shared[tix * 2] = -FLT_MAX; _shared[tix * 2 + 1] = FLT_MAX;
	for (int i = tix; i < col; i += bdx)_shared[tix * 2] = ::max(_shared[tix * 2], A[bix * pitch + i]), _shared[tix * 2 + 1] = ::min(_shared[tix * 2 + 1], A[bix * pitch + i]);
	_syncthreads();
	//find col Max
	if (tix == 0) {
		for (int i = 1; i < bdx; i++)_shared[0] = ::max(_shared[0], _shared[i * 2]), _shared[1] = ::min(_shared[1], _shared[i * 2 + 1]);
	}
	_syncthreads();
	if (abs(_shared[0] - _shared[1]) > eps) {
		for (int i = tix; i < col; i += bdx)A[bix * pitch + i] = (A[bix * pitch + i] - _shared[1]) / (_shared[0] - _shared[1]);
		if (tix == 0)factor[bix] = 1.0f / (_shared[0] - _shared[1]);// , printf("max:%lf min:%lf\n", _shared[0], _shared[1]);
	}
	else {
		//printf("smaller than eps\n");
		assert(_shared[0] == 0 && _shared[1] == 0);
		for (int i = tix; i < col; i += bdx)A[bix * pitch + i] = 0.5f;
	}
}
__global__ void Cuda_Matrix_OneHot_Sampling(int pitch, floatType* A, int col, curandState* states, int states_pitch) {
	ui tix = threadIdx.x;
	ui bix = blockIdx.x;
	ui bdx = blockDim.x;
	__shared__ extern floatType _shared[];
	_shared[tix] = 0;
	for (int i = tix; i < col; i += bdx) {
		_shared[tix] += A[bix * pitch + i];
	}
	_syncthreads();
	//Summary
	if (tix == 0) {
		for (int i = 1; i < bdx; i++)
			_shared[i] += _shared[i - 1];
		//random number
		assert(_shared[bdx - 1] > 0);
		_shared[bdx] = _shared[bdx - 1] * 1.0 * ((curand(&states[bix * states_pitch]) % ((ui)0xfffffffe)) + 1) / 0xffffffff;
	}
	_syncthreads();
	int tar = -1;
	if (_shared[bdx] <= _shared[tix])
		for (ui i = tix; i < col; i += bdx) {
			_shared[tix] -= A[bix * pitch + i];
			if (_shared[bdx] > _shared[tix]) {
				A[bix * pitch + i] = 1;
				tar = i;
				break;
			}
		}
	for (int i = tix; i < col; i += bdx)
		if (i != tar)
			A[bix * pitch + i] = 0;
}
//C=a*(A)+b*(B)
template<class T> Matrix<T>* _geam(const T* alpha, const Matrix<T>* A, const T* beta, const Matrix<T>* B, Matrix<T>* Out = NULL, cublasOperation_t A_ops = CUBLAS_OP_N) {
	if (Out == NULL) {
		if (A_ops == CUBLAS_OP_N)
			Out = malloc_Matrix(A->row, A->col);// new Matrix<T>(A->row, A->col);
		else Out = malloc_Matrix(A->col, A->row);//new Matrix<T>(A->col, A->row);
		Out->alloc_Cuda_M();
	}
	cublasSgeam(handle[get_stm_id], A_ops, A_ops, Out->col, Out->row, alpha, A->Cuda_M, A->pitch / sz, beta, B->Cuda_M,
		B->pitch / sz, Out->Cuda_M, Out->pitch / sz);
	cudaDeviceSynchronize();
	return Out;
}
//y=a*(A)x+by
template<class T> Matrix<T>* _gemv(T alpha, Matrix<T>* A, T* x, T beta, Matrix<T>* y_Out = NULL, int incx = 1) {
	if (y_Out == NULL) {
		y_Out = malloc_Matrix(A->row, 1);//new Matrix<T>(A->row, 1);
		y_Out->alloc_Cuda_M();
	}
	cublasSgemv(handle[get_stm_id], CUBLAS_OP_T, A->col, A->row, &alpha, A->Cuda_M, A->pitch / sz, x, incx, &beta, y_Out->Cuda_M, y_Out->pitch / sz);
	cudaDeviceSynchronize();
	return y_Out;
}
//A=a*(x)(y)T+A
template<class T> Matrix<T>* _ger(const T& alpha, const T* x, int incx, const T* y, int incy, Matrix<T>* A_Out) {
	//incx,incy>0
	cublasSger(handle[get_stm_id], A_Out->col, A_Out->row, &alpha, y, incy, x, incx, A_Out->Cuda_M, A_Out->pitch / sz);
	cudaDeviceSynchronize();
	return A_Out;
}
//C=a*(A)*(B)+b*C
template<class T> Matrix<T>* _gemm(T alpha, const Matrix<T>* A, const Matrix<T>* B, T beta, Matrix<T>* Out = NULL) {
	if (Out == NULL) {
		Out = malloc_Matrix(A->row, B->col);//new Matrix<T>(A->row, B->col);
		Out->alloc_Cuda_M();
	}
	cublasStatus_t status = cublasSgemm(handle[get_stm_id], CUBLAS_OP_N, CUBLAS_OP_N, B->col, A->row, A->col, &alpha, B->Cuda_M, B->pitch / sz, A->Cuda_M,
		A->pitch / sz, &beta, Out->Cuda_M, Out->pitch / sz);
	if (status != CUBLAS_STATUS_SUCCESS)printf("cublas errorCode: %d\n", status), assert(false);
	cudaDeviceSynchronize();
	return Out;
}
//y[j]=x[k],i=1,...,n,k=1+(i-1)*incx,j=1+(i-1)*incy
//incx,incy>=0
template<class T> Matrix<T>* _copy(int row, int col, int incx, int incy, const T* x, Matrix<T>* Out = NULL) {
	if (Out == NULL) {
		Out = malloc_Matrix(row, col);//new Matrix<T>(row, col);
		Out->alloc_Cuda_M();
	}
	cublasScopy(handle[get_stm_id], row * col, x, incx, Out->Cuda_M, incy);
	cudaDeviceSynchronize();
	return Out;
}

template<class T> Matrix<T>* Matrix<T>::Matrix_row_col_expand(Kernel MatrixKernel, Matrix<T>* R, bool TemporaryVariable) {
	Matrix<T>* tmp = NULL;
	if (!TemporaryVariable) {
#ifdef __GPU__
		//number+=Matrix
		if (col == 1 && row == 1 && col < R->col && row < R->row) {
			tmp = R = R->Cuda_ACC_Sum();
		}
		//col+=Matrix
		else if (col == 1 && col < R->col) {
			tmp = R = R->Cuda_ACC_ScaleOneCol();
		}
		//row+=Matrix
		else if (row == 1 && row < R->row) {
			tmp = R = R->Cuda_ACC_ScaleOneCol(true);
		}
#else
		//ptr->Cuda_ACC_minus_(Right.getMatrix());
#endif
	}
	//Mat,Mat col,Mat row,Mat number,Mat
	assert((row == R->row && col == R->col) || (row == R->row && (col == 1 || R->col == 1)) || (col == R->col && (row == 1 || R->row == 1)) ||
		(row == 1 && col == 1) || (R->row == 1 && R->col == 1));
	int _row = max(row, R->row), _col = max(col, R->col);
	int _sz = (_col + 32 - 1) / 32 * 32;
	dim3 grid(_row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Matrix<T>* Out = TemporaryVariable ? malloc_Matrix(_row, _col) : this;
	Out->alloc_Cuda_M();
	MatrixKernel << <grid, block, 0, get_stm >> > (Out->pitch / sz, pitch / sz, R->pitch / sz, Cuda_M, R->Cuda_M, Out->Cuda_M, Out->col, row, col, R->row, R->col);
	Heap[get_stm_id].free(tmp);
	return Out;
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_Basic_Cal(int Cal_Ops, Matrix<T>* R, bool TemporaryVariable) {
	switch (Cal_Ops)
	{
		//+
	case 0:return Matrix_row_col_expand(Cuda_Matrix_plus_Matrix, R, TemporaryVariable);
		//-
	case 1:return Matrix_row_col_expand(Cuda_Matrix_minus_Matrix, R, TemporaryVariable);
		//*
	case 2:return Matrix_row_col_expand(Cuda_Matrix_mul_Matrix, R, TemporaryVariable);
		//\/
	case 3:return Matrix_row_col_expand(Cuda_Matrix_div_Matrix, R, TemporaryVariable);
	default:
		break;
	}
	assert(false);
	return NULL;
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_mul(const Matrix<T>* R) {
	assert(col == R->row);
	return _gemm(1.0f, this, R, 0.0f);
}
//Matrix<double>*OneVector = NULL;
//Matrix+Col/Matrix+Row
//template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_plus_minus(Matrix<T>*R, const int& sign, bool TemporaryVariable) {
//	//plus/minus col/row
//	if (col != R->col || row != R->row) {
//		const Matrix<T>*_A = row == R->row ? (col > R->col ? this : R) : (row > R->row ? this : R);
//		const Matrix<T>*_B = row == R->row ? (col > R->col ? R : this) : (row > R->row ? R : this);
//		assert((row == R->row&&_B->col == 1) || (col == R->col&&_B->row == 1));
//		//assert(sign == 1 || (sign == -1 && _B == R));
//		Matrix<T>* Out = TemporaryVariable ? malloc_Matrix(_A->row, _A->col)/*new Matrix<T>(_A->row, _A->col)*/ : this;
//		Out->alloc_Cuda_M();
//		//cudaMemcpy2DAsync(Out->Cuda_M, Out->pitch, _A->Cuda_M, _A->pitch, _A->WidthStride, _A->row, cudaMemcpyDeviceToDevice);
//		////One vector
//		//if (!OneVector || _A->col > OneVector->col || _A->row > OneVector->col) {
//		//	delete OneVector;
//		//	OneVector = _copy(1, max(_A->col, _A->row), 0, 1, Device_One);
//		//}
//		int _sz = (Out->col + 32 - 1) / 32 * 32;
//		dim3 grid(Out->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
//		if (row == R->row) {
//			Cuda_Matrix_col_expand << <grid, block, 0, get_stm >> > (pitch / sz, R->pitch / sz, Out->pitch / sz, Cuda_M, R->Cuda_M, Out->Cuda_M, col, R->col, Out->col, sign);
//			//return _ger(sign, _B->Cuda_M, _B->pitch / sz, OneVector->Cuda_M, 1, Out);
//		}
//		else Cuda_Matrix_row_expand << <grid, block, 0, get_stm >> > (pitch / sz, R->pitch / sz, Out->pitch / sz, Cuda_M, R->Cuda_M, Out->Cuda_M, row, R->row, Out->col, sign);
//		return Out;
//		//else return _ger(sign, OneVector->Cuda_M, 1, _B->Cuda_M, 1, Out);
//	}
//	else return _geam(&(const T&)1.0, this, &(const T&)sign, R, TemporaryVariable ? NULL : this);
//}
//template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_plus_(Matrix<T>*Right,const int& sign) {
//	//assert(col == R->col&&row == R->row);
//	//return _geam(&(const T&)1.0, this, &(const T&)1.0, R, this);
//
//	return this;
//}
//template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_minus(Matrix<T>*R) {
//	assert(row == R->row);
//	//拓展列
//	if (col != R->col) {
//		Matrix<T>* Out = new Matrix<T>(row, max(col, R->col));
//		Out->alloc_Cuda_M();
//		int _sz = (Out->col + 32 - 1) / 32 * 32;
//		dim3 grid(Out->row, 1, 1), block(min(_sz, 512), 1, 1);
//		//Cuda_Matrix_col_expand << <grid, block >> > (pitch / sz, R->pitch / sz, Out->pitch / sz, Cuda_M, R->Cuda_M, Out->Cuda_M, col, R->col, Out->col);
//		cudaDeviceSynchronize();
//		return Out;
//	}
//	return _geam(&(const T&)1.0, this, &(const T&)-1.0, R);
//}
//template<class T> const Matrix<T>* Matrix<T>::Cuda_ACC_minus_(Matrix<T>*R) {
//	/*
//	ui sz = sizeof(T);
//	dim3 grid(row, 1, 1), block(min(col, 512), 1, 1);
//	Cuda_Matrix_minus_ << <grid, block >> >(pitch / sz, Cuda_M, R->Cuda_M, col);
//	return this;*/
//	assert(col == R->col&&row == R->row);
//	return _geam(&(const T&)1.0, this, &(const T&)-1.0, R, this);
//}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_number(int Ops, const T& number, bool lnumber) {
	switch (Ops)
	{
		//+，-
	case 0:
	case 1:
	{
		Matrix<T>* Out = malloc_Matrix(row, col);//new Matrix<T>(row, col);
		Out->alloc_Cuda_M();
		int _sz = (Out->col + 32 - 1) / 32 * 32;
		dim3 grid(Out->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
		if (lnumber)
			Cuda_Matrix_plus_minus_lnumber << <grid, block, 0, get_stm >> > (pitch / sz, Out->Cuda_M, Cuda_M, number, col, Ops ? -1 : 1);
		else Cuda_Matrix_plus_minus_rnumber << <grid, block, 0, get_stm >> > (pitch / sz, Out->Cuda_M, Cuda_M, number, col, Ops ? -1 : 1);
		return Out;
	}
	//*
	case 2:return _geam(&number, this, &(const T&)0.0f, this);
		//\/
	case 3:
		if (lnumber)
			return Cuda_ACC_function(number_div, true, number);
		else return _geam(&(const T&)(1.0f / number), this, &(const T&)0.0f, this);
	default:
		break;
	}
	assert(false);
	return NULL;
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_assign(const Matrix<T>* R, int Copy_Row_Num, int dst_Start_row, int src_Start_col, int src_Start_row) {
	assert(R->Cuda_M);
	assert(Copy_Row_Num <= row - dst_Start_row);
	alloc_Cuda_M();
	cudaMemcpy2DAsync(Cuda_M + pitch * dst_Start_row / sz, pitch, R->Cuda_M + src_Start_col + R->pitch * src_Start_row / sz, R->pitch, WidthStride, Copy_Row_Num, cudaMemcpyDeviceToDevice, get_stm);
	return this;
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_assign_(const Matrix<T>* R, int Copy_Row_Num, int dst_Start_col) {
	assert(R->Cuda_M);
	assert(col >= R->col);
	alloc_Cuda_M();
	cudaMemcpy2DAsync(Cuda_M + dst_Start_col, pitch, R->Cuda_M, R->pitch, R->WidthStride, Copy_Row_Num, cudaMemcpyDeviceToDevice, get_stm);
	return this;
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_function(int fun_id, bool TemporaryVariable, const T& param, int Start_Row) {
	alloc_Cuda_M();
	Matrix<T>* Out = this;
	if (TemporaryVariable) {
		Out = malloc_Matrix(row, col);//new Matrix<T>(row, col);
		Out->alloc_Cuda_M();
	}
	int _sz = (col + 32 - 1) / 32 * 32;
	dim3 grid(row - Start_Row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	if (fun_id == Null_Func) {
		if (TemporaryVariable)assert(false);
	}
	else if (fun_id < DropOut_Bernoulli)Cuda_Matrix_function << <grid, block, 0, get_stm >> > (pitch / sz, Cuda_M + Start_Row * pitch / sz, Out->Cuda_M + Start_Row * pitch / sz, col, dev_f[fun_id]);
	//contain random number
	else if (fun_id == DropOut_Bernoulli || fun_id == Uniform)
		Cuda_Matrix_function_random << <grid, block, 0, get_stm >> > (pitch / sz, Cuda_M + Start_Row * pitch / sz, Out->Cuda_M + Start_Row * pitch / sz, col, dev_f2[fun_id - DropOut_Bernoulli], param, states[get_stm_id], states_pitch);
	else Cuda_Matrix_function_param << <grid, block, 0, get_stm >> > (pitch / sz, Cuda_M + Start_Row * pitch / sz, Out->Cuda_M + Start_Row * pitch / sz, col, dev_f2[fun_id - DropOut_Bernoulli], param);
	cudaDeviceSynchronize();
	return Out;
}
template<class T> Matrix<T>* Matrix<T>::ScaleImage(int Image_W, int Image_Depth, Matrix<T>* Location, int Scale_Image_WH, int Scale_Num) {
	Matrix<T>* Out = malloc_Matrix(Scale_Image_WH * Scale_Image_WH * Image_Depth * Scale_Num, col);
	Out->alloc_Cuda_M();
	int _sz = (Out->col + 32 - 1) / 32 * 32;
	dim3 grid(Out->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_Scale_Image << <grid, block, 0, get_stm >> > (pitch / sz, Out->Cuda_M, col, Cuda_M, Image_W, row / (Image_W * Image_Depth), Image_Depth, Location->Cuda_M, Scale_Image_WH);
	cudaDeviceSynchronize();
	return Out;
}
template<class T> const Matrix<T>* Matrix<T>::Normal_Sampling() {
	alloc_Cuda_M();
	int _sz = (col + 32 - 1) / 32 * 32;
	dim3 grid((row + 1) / 2, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_Normal_Sampling << < grid, block, 0, get_stm >> > (pitch / sz, Cuda_M, col, row, states[get_stm_id], states_pitch);
	cudaDeviceSynchronize();
	return this;
}
template<class T> const Matrix<T>* Matrix<T>::OneHot_Sampling() {
	Matrix<T>* M = Cuda_ACC_T();
	//find col max and cal
	int _sz = ((int)pow(M->col, 0.55) + 32 - 1) / 32 * 32;
	dim3 grid(M->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_OneHot_Sampling << <grid, block, sz* (block.x + 1), get_stm >> > (M->pitch / sz, M->Cuda_M, M->col, states[get_stm_id], states_pitch);
	cudaDeviceSynchronize();
	_geam(&(const T&)1.0f, M, &(const T&)0.0f, M, this, CUBLAS_OP_T);

	Heap[get_stm_id].free(M);
	return this;
}
template<class T> const Matrix<T>* Matrix<T>::GomokuSimulation(Matrix<T>* Moves, Matrix<T>* BoardID, bool InPut, Matrix<T>* Reward, const Matrix<T>* Random, Matrix<T>* Value) {
	int _sz = (col + 32 - 1) / 32 * 32;
	bool One_Encode = BoardID->row == 1;
	dim3 grid(InPut ? (row / Gomoku_Planes) : (One_Encode ? 1 : row), 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	if (InPut) {
		Cuda_Matrix_Board_Simulation << < grid, block, 0, get_stm >> > (pitch / sz, Moves ? Moves->pitch / sz : pitch / sz, Cuda_M, col, Moves ? Moves->Cuda_M : (BoardID->Cuda_M + pitch / sz), BoardID->Cuda_M, Reward ? Reward->Cuda_M : NULL, Random ? Random->Cuda_M : NULL);
		Cuda_Matrix_Board_Simulation_cal << < grid, block, 0, get_stm >> > (pitch / sz, Moves ? Moves->pitch / sz : pitch / sz, Cuda_M, col, Moves ? Moves->Cuda_M : (BoardID->Cuda_M + pitch / sz), BoardID->Cuda_M, Reward ? Reward->Cuda_M : NULL);
	}
	else Cuda_Matrix_Board_Simulation_ << < grid, block, 0, get_stm >> >
		(pitch / sz, Moves->pitch / sz, Cuda_M, col, Moves->Cuda_M, BoardID->Cuda_M, Reward ? Reward->Cuda_M : NULL, Reward ? Reward->pitch / sz : NULL, Random ? Random->Cuda_M : NULL, Value ? Value->Cuda_M : NULL, One_Encode);
	cudaDeviceSynchronize();
	return this;
}
template<class T> const Matrix<T>* Matrix<T>::GomokuSimulation_Extend(Matrix<T>* Board, int Planes) {
	int _sz = (col + 32 - 1) / 32 * 32;
	dim3 grid(row / Planes, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_Board_Simulation_Extend << < grid, block, 0, get_stm >> > (pitch / sz, Cuda_M, Board->Cuda_M, col, Planes);
	cudaDeviceSynchronize();
	return this;
}
template<class T> const Matrix<T>* Matrix<T>::ChessRepresentDecode(Matrix<T>* Board, int Planes) {
	int _sz = (col + 32 - 1) / 32 * 32;
	dim3 grid(1, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_Chess_Representation_Decode << < grid, block, 0, get_stm >> > (pitch / sz, Cuda_M, Board->Cuda_M, col, Planes);
	cudaDeviceSynchronize();
	return this;
}
template<class T>
MatrixPtr<T>& MatrixPtr<T>::MahjongRepresentDecode(const MatrixPtr<T>& Board, int Planes) {
	int _sz = (ptr->col + 32 - 1) / 32 * 32;
	dim3 grid(1, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_Mahjong_Representation_Decode << < grid, block, 0, get_stm >> > (ptr->pitch / sz, ptr->Cuda_M, Board.ptr->Cuda_M, ptr->col, Planes);
	return *this;
}
template<class T>
MatrixPtr<T>& MatrixPtr<T>::MahjongAgentRepresentDecode(const MatrixPtr<T>& Board, int Planes) {
	int _sz = (ptr->col + 32 - 1) / 32 * 32;
	dim3 grid(1, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_Mahjong_Agent_Representation_Decode << < grid, block, 0, get_stm >> > (ptr->pitch / sz, ptr->Cuda_M, Board.ptr->Cuda_M, ptr->col, Planes);
	return *this;
}
template<class T>
MatrixPtr<T>& MatrixPtr<T>::Mahjong_Reward_RepresentDecode(const MatrixPtr<T>& Board, int Planes) {
	int _sz = (ptr->col + 32 - 1) / 32 * 32;
	dim3 grid(1, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_Mahjong_Reward_Representation_Decode << < grid, block, 0, get_stm >> > (ptr->pitch / sz, ptr->Cuda_M, Board.getMatrix()->Cuda_M, ptr->col, Planes);
	return *this;
}

template<class T> const Matrix<T>* Matrix<T>::Go_Action_Encode(Matrix<T>* Out_Sample, int Start_Row, ui ActionSpace) {
	alloc_Cuda_M();
	int _sz = (col + 32 - 1) / 32 * 32;
	dim3 grid(1, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Go_Action_Encode << < grid, block, 0, get_stm >> > (pitch / sz, Cuda_M, Out_Sample->Cuda_M, col, Start_Row, ActionSpace, states[get_stm_id], states_pitch);
	cudaDeviceSynchronize();
	return this;
}
template<class T> const Matrix<T>* Matrix<T>::Chess_Action_Encode(Matrix<T>* Out_Sample, int Start_Row, ui ActionSpace, Matrix<T>* ActionMap, int W, int Plane, bool rotateAction) {
	alloc_Cuda_M();
	int _sz = (col + 32 - 1) / 32 * 32;
	dim3 grid(1, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Chess_Action_Encode << < grid, block, 0, get_stm >> > (pitch / sz, Cuda_M, Out_Sample->Cuda_M + Start_Row * pitch / sz, col, ActionSpace, Plane, W, ActionMap->Cuda_M, rotateAction, Start_Row!=0, states[get_stm_id], states_pitch);
	cudaDeviceSynchronize();
	return this;
}
template<class T>
const MatrixPtr<T>& MatrixPtr<T>::Mahjong_Action_Encode(const MatrixPtr<T>& Out_Sample, const MatrixPtr<T>& value, int Start_Row, ui ActionSpace, int Plane) {
	ptr->alloc_Cuda_M();
	int _sz = (ptr->col + 32 - 1) / 32 * 32;
	dim3 grid(1, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Mahjong_Action_Encode << < grid, block, 0, get_stm >> > (ptr->pitch / sz, ptr->Cuda_M, Out_Sample.getMatrix()->Cuda_M + Start_Row * ptr->pitch / sz, value.getMatrix()?value.getMatrix()->Cuda_M:NULL, ptr->col, ActionSpace, Plane, states[get_stm_id], states_pitch);
	return *this;
}
template<class T> 
MatrixPtr<T>& MatrixPtr<T>::Mahjong_Policy_Encode(const MatrixPtr<T>& Out_Sample, int Start_Row){
	ptr->alloc_Cuda_M();
	int _sz = (ptr->col + 32 - 1) / 32 * 32;
	dim3 grid(ptr->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Mahjong_Policy_Encode << < grid, block, 0, get_stm >> > (ptr->pitch / sz, ptr->Cuda_M, Out_Sample.getMatrix()->Cuda_M + Start_Row * ptr->pitch / sz, ptr->col, (Out_Sample.getMatrix()->row - Start_Row) / 2);
	return *this;
}
template<class T>
MatrixPtr<T>& MatrixPtr<T>::Mahjong_Simplify_Policy_Encode(const MatrixPtr<T>& Out_Sample, int Start_Row) {
	ptr->alloc_Cuda_M();
	int _sz = (ptr->col + 32 - 1) / 32 * 32;
	dim3 grid(ptr->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Mahjong_Simplify_Policy_Encode << < grid, block, 0, get_stm >> > (ptr->pitch / sz, ptr->Cuda_M, Out_Sample.getMatrix()->Cuda_M + Start_Row * ptr->pitch / sz, ptr->col, (Out_Sample.getMatrix()->row - Start_Row) / 2);
	return *this;
}
template<class T>
MatrixPtr<T>& MatrixPtr<T>::Mahjong_Values_Encode() {
	Matrix<T>* M = ptr->Cuda_ACC_T();
	int _sz = ((int)pow(M->col, 0.55) + 32 - 1) / 32 * 32;
	dim3 grid(M->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Mahjong_Values_Encode << <grid, block, sz* block.x * 2, get_stm >> > (M->pitch / sz, M->Cuda_M, M->col);
	_geam(&(const T&)1.0f, M, &(const T&)0.0f, M, ptr, CUBLAS_OP_T);

	Heap[get_stm_id].free(M);
	return *this;
}
template<class T>
MatrixPtr<T>& MatrixPtr<T>::Mahjong_Reward_softmax_Encode(int idx, const MatrixPtr<T>& final_reward) {
	ptr->alloc_Cuda_M();
	int _sz = (ptr->col + 32 - 1) / 32 * 32;
	dim3 grid(1, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Mahjong_reward_Softmax_Encode << < grid, block, 0, get_stm >> > (ptr->pitch / sz, ptr->Cuda_M, final_reward.getMatrix()->Cuda_M, ptr->col, idx);
	return *this;
}
template<class T>
MatrixPtr<T>& MatrixPtr<T>::Mahjong_Reward_Sample() {
	ptr->alloc_Cuda_M();
	int _sz = (ptr->col + 32 - 1) / 32 * 32;
	dim3 grid(ptr->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_reward_sample << < grid, block, 0, get_stm >> > (ptr->pitch / sz, ptr->Cuda_M, ptr->col, states[get_stm_id], states_pitch);
	return *this;
}
template<class T> const Matrix<T>* Matrix<T>::Chess_Policy_Encode(Matrix<T>* Out_Sample, int Start_Row, Matrix<T>* ActionMap, bool rotateAction) {
	alloc_Cuda_M();
	int _sz = (col + 32 - 1) / 32 * 32;
	dim3 grid(ActionMap->col, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Chess_Policy_Encode << < grid, block, 0, get_stm >> > (pitch / sz, Cuda_M, Out_Sample->Cuda_M + Start_Row * pitch / sz, col, ActionMap->Cuda_M, (Out_Sample->row - Start_Row) / 2, rotateAction);
	cudaDeviceSynchronize();
	return this;
}
template<class T> Matrix<T>* Matrix<T>::Conv_im2col(int Image_Depth, int Image_W, int W, int H, int Receptive, int Padding, int Stride, Matrix<T>* bp) {
	int Image_H = row / (Image_Depth * Image_W);
	Matrix<T>* Out = NULL;
	if (!bp) {
		Out = malloc_Matrix(Receptive * Receptive * Image_Depth, W * H * col);
		Out->alloc_Cuda_M();
	}
	else Out = bp;
	int _sz = ((bp ? col : Out->col) + 32 - 1) / 32 * 32;
	//dim3 grid(bp ? row : Out->row, bp ? 1 : ceil(1.0 * _sz / 4 / min(_sz, 64)), 1), block(min(_sz, 64), 1, 1);
	dim3 grid(bp ? row : Out->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	if (!bp)
		Cuda_im2col << < grid, block, 0, get_stm >> > (pitch / sz, Out->pitch / sz, Cuda_M, Image_Depth, Image_W, Image_H, Receptive, Out->col, W, H, Stride, Padding, Out->Cuda_M);
	else Cuda_im2col_bp << < grid, block, 0, get_stm >> > (pitch / sz, Out->pitch / sz, Cuda_M, Image_Depth, Image_W, Image_H, Receptive, col, W, H, Stride, Padding, Out->Cuda_M);
	cudaDeviceSynchronize();
	return (!bp) ? Out : NULL;
}
template<class T> Matrix<T>* Matrix<T>::Image_Pooling(int Image_Depth, int Image_W, int W, int H, int Receptive, int Padding, int Stride, Matrix<T>* Pool_idx, Matrix<T>* bp) {
	int Image_H = row / (Image_Depth * Image_W);
	Matrix<T>* Out = NULL;
	if (!bp) {
		Out = malloc_Matrix(W * H * Image_Depth, col);
		Out->Cuda_ACC_ZeroMemory();
		Pool_idx->Cuda_ACC_function(Assignment, false, -1);
	}
	else Out = bp;
	int _sz = (Out->col + 32 - 1) / 32 * 32;
	dim3 grid((!bp) ? Out->row : row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	if (!bp)
		Cuda_Max_Pooling << < grid, block, 0, get_stm >> > (pitch / sz, Out->pitch / sz, Cuda_M, Image_Depth, Image_W, Image_H, Receptive, Out->col, W, H, Stride, Padding,
			Out->Cuda_M, Pool_idx->Cuda_M);
	else Cuda_Max_Pooling_bp << < grid, block, 0, get_stm >> > (pitch / sz, Out->pitch / sz, Cuda_M, Image_Depth, Image_W, Image_H, Receptive, Out->col, W, H, Stride, Padding, Out->Cuda_M, Pool_idx->Cuda_M);
	cudaDeviceSynchronize();
	return bp ? NULL : Out;
}
template<class T> Matrix<T>* Matrix<T>::Image_SpatialConcatenate(Matrix<T>* first, Matrix<T>* second, int size, bool bp, bool first_Order) {
	alloc_Cuda_M();
	int _sz = (col + 32 - 1) / 32 * 32;
	dim3 grid(row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	if (!bp)
		Cuda_Image_SpatialConcatenate << < grid, block, 0, get_stm >> > (pitch / sz, col, first->Cuda_M, second->Cuda_M, Cuda_M, row / size, first->row / size);
	else Cuda_Image_SpatialConcatenate_bp << < grid, block, 0, get_stm >> > (pitch / sz, col, first->Cuda_M, Cuda_M, row / size, first->row / size, first_Order);
	cudaDeviceSynchronize();
	return this;
}
template<class T> const Matrix<T>* Matrix<T>::Conv_im2col_Restore(int Image_W, Matrix<T>* Restore_Image, bool bp) {
	Restore_Image->alloc_Cuda_M();
	int _sz = (Restore_Image->col + 32 - 1) / 32 * 32;
	dim3 grid(Restore_Image->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	if (!bp)
		Cuda_im2col_Restore << < grid, block, 0, get_stm >> > (pitch / sz, Restore_Image->pitch / sz, row, Cuda_M, Restore_Image->Cuda_M, Restore_Image->col, Image_W, col / (Restore_Image->col * Image_W));
	else Cuda_im2col_Restore_bp << < grid, block, 0, get_stm >> > (pitch / sz, Restore_Image->pitch / sz, row, Cuda_M, Restore_Image->Cuda_M, Restore_Image->col, Image_W, col / (Restore_Image->col * Image_W));
	cudaDeviceSynchronize();
	return this;
}

template<class T> const MatrixPtr<T> MatrixPtr<T>::BN_Normalization(MatrixPtr<T>& mean, MatrixPtr<T>& var)const {
	Matrix<T>* Out = NULL;
	Out = malloc_Matrix(ptr->row, ptr->col);
	Out->alloc_Cuda_M();
	int _sz = (ptr->col + 32 - 1) / 32 * 32;
	dim3 grid(ptr->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_BN_Normalization << < grid, block, 0, get_stm >> > (ptr->pitch / sz, ptr->col, ptr->Cuda_M, mean.ptr->Cuda_M, var.ptr->Cuda_M, Out->Cuda_M, mean.ptr->pitch / sz);
	cudaDeviceSynchronize();
	return Out;
}
template<class T> void MatrixPtr<T>::MinMax_Normalization(MatrixPtr<T>& result, MatrixPtr<T>& factor)const {
	result.getMatrix()->alloc_Cuda_M();
	Matrix<T>* M = ptr->Cuda_ACC_T();
	//find col max and cal
	int _sz = ((int)pow(M->col, 0.55) + 32 - 1) / 32 * 32;
	dim3 grid(M->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_MinMax_Normalization << <grid, block, sz* block.x * 2, get_stm >> > (M->pitch / sz, M->Cuda_M, M->col, factor.getMatrix()->Cuda_M);
	_geam(&(const T&)1.0f, M, &(const T&)0.0f, M, result.getMatrix(), CUBLAS_OP_T);

	Heap[get_stm_id].free(M);
}
template<class T> const Matrix<T>* Matrix<T>::Cuda_ACC_generate_random_number(ui Mod_Value) {
	alloc_Cuda_M();
	int _sz = (col + 32 - 1) / 32 * 32;
	dim3 grid(row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_random << < grid, block, 0, get_stm >> > (pitch / sz, Cuda_M, col, states[get_stm_id], states_pitch, Mod_Value);
	cudaDeviceSynchronize();
	return this;
}
template<class T> const Matrix<T>* Matrix<T>::Order_Assign_Val(Matrix<T>* Order, T val) {
	int _sz = (Order->col + 32 - 1) / 32 * 32;
	dim3 grid(Order->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_Order_Assign << < grid, block, 0, get_stm >> > (pitch / sz, Order->pitch / sz, Cuda_M, Order->Cuda_M, Order->col, val);
	cudaDeviceSynchronize();
	return this;
}

template<class T> const Matrix<T>* Matrix<T>::Cuda_ACC_RandMatrixs(Matrix<T>* R, Matrix<T>* Order, int Start_row, bool Right_Mat_random_Order, ui Max_Order) {
	assert(row - Start_row == R->row);
	alloc_Cuda_M();
	int _sz = ((Right_Mat_random_Order ? col : R->col) + 32 - 1) / 32 * 32;
	dim3 grid(row - Start_row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	if (Right_Mat_random_Order)
		Cuda_Matrix_rand_order << <grid, block, 0, get_stm >> > (pitch / sz, R->pitch / sz, Cuda_M + Start_row * pitch / sz, R->Cuda_M, col, Max_Order == 0 ? R->col : Max_Order, Order->Cuda_M);
	else Cuda_Matrix_rand_order_ << <grid, block, 0, get_stm >> > (pitch / sz, R->pitch / sz, Cuda_M + Start_row * pitch / sz, R->Cuda_M, col, R->col, Order->Cuda_M);
	cudaDeviceSynchronize();
	return this;
}
template<class T> const Matrix<T>* Matrix<T>::RandomPositiveSample(Matrix<T>* Word_Context_Possibility, Matrix<T>* Word_Context_Context_Idx, Matrix<T>* Word_Context_Range, Matrix<T>* Negative_Sampling_Generator) {
	int _sz = (col + 32 - 1) / 32 * 32;
	dim3 grid(row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_Positive_Sampling << <grid, block, 0, get_stm >> > (Cuda_M, Word_Context_Possibility->Cuda_M, Word_Context_Context_Idx->Cuda_M, Word_Context_Range->Cuda_M, Negative_Sampling_Generator->Cuda_M, col, Word_Context_Range->col, states[get_stm_id], states_pitch);
	cudaDeviceSynchronize();
	return this;
}
template<class T> const Matrix<T>* Matrix<T>::GenerateNegaOrder(Matrix<T>* Context, Matrix<T>* Context_Range, Matrix<T>* Order, Matrix<T>* Unigram_Table) {
	ui sz = sizeof(T);
	int _sz = (col + 32 - 1) / 32 * 32;
	dim3 grid(row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_Negative_Sampling << <grid, block, 0, get_stm >> > (pitch / sz, Context->pitch / sz, Context_Range->pitch / sz, Cuda_M, Context->Cuda_M, Context_Range->Cuda_M, col, Context_Range->col, Order->Cuda_M, Order->col, Unigram_Table->Cuda_M, Unigram_Table->col, states[get_stm_id], states_pitch);
	cudaDeviceSynchronize();
	return this;
}
template<class T> const Matrix<T>* Matrix<T>::GenerateOutPutMask(Matrix<T>* Negative_Sampling_Order, Matrix<T>* Order, Matrix<T>* Context_Idx, Matrix<T>* Context_Range, Matrix<T>* OutPut_Mask, Matrix<T>* OutPut_Mask1) {
	int _sz = (col + 32 - 1) / 32 * 32;
	dim3 grid(row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
	Cuda_Matrix_Generate_OutPutMask << <grid, block, 0, get_stm >> > (pitch / sz, Cuda_M, OutPut_Mask->Cuda_M, OutPut_Mask1->Cuda_M, Negative_Sampling_Order->Cuda_M, col, Order->Cuda_M, Context_Idx->Cuda_M, Context_Idx->pitch / sz, Context_Idx->col, Context_Idx->row, Context_Range->Cuda_M, Context_Range->col);
	cudaDeviceSynchronize();
	return this;
}
template<class T>const Matrix<T>* Matrix<T>::Cuda_ACC_SoftMaxFunction(bool Max, Matrix<T>* One_Col) {
	if (!One_Col) {
		Matrix<T>* M = Cuda_ACC_T();
		//find col max and cal
		int _sz = ((int)pow(M->col, 0.55) + 32 - 1) / 32 * 32;
		dim3 grid(M->row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
		Cuda_Matrix_SoftMax_ << <grid, block, sz* (block.x), get_stm >> > (M->pitch / sz, M->Cuda_M, M->col, Max);
		cudaDeviceSynchronize();
		if (!Max) {
			//Col Sum
			Matrix<T>* Sum = M->Cuda_ACC_ScaleOneCol();
			Cuda_Matrix_SoftMax << <grid, block, sz, get_stm >> > (M->pitch / sz, M->Cuda_M, M->col, Sum->Cuda_M, Sum->pitch / sz);
			cudaDeviceSynchronize();
			Heap[get_stm_id].free(Sum);
			//delete Sum;
		}
		_geam(&(const T&)1.0f, M, &(const T&)0.0f, M, this, CUBLAS_OP_T);

		Heap[get_stm_id].free(M);
		//delete M;
		return this;
	}
	else {
		int _sz = ((int)pow(col, 0.55) + 32 - 1) / 32 * 32;
		dim3 grid(row, 1, 1), block(min(_sz, Maximum_Cuda_Block_Size), 1, 1);
		Cuda_Matrix_mul_Row << < grid, block, 0, get_stm >> > (pitch / sz, Cuda_M, One_Col->Cuda_M, col);
		cudaDeviceSynchronize();
		Cuda_Matrix_SoftMax_ << <grid, block, sz* (block.x), get_stm >> > (pitch / sz, Cuda_M, col, Max);
		cudaDeviceSynchronize();
		Cuda_Matrix_One_To_No << <grid, block, 0, get_stm >> > (pitch / sz, Cuda_M, col);
		cudaDeviceSynchronize();
		return this;
	}
}
Matrix<floatType>* One_row[Cuda_Max_Stream] = { NULL };// = new Matrix<double>(1, 1);
ui max_col_size[Cuda_Max_Stream] = { 0 };
void Row_Reset(int new_size) {
	int id = get_stm_id;
	if (new_size > max_col_size[id]) {
		max_col_size[id] = new_size;
		delete One_row[id];
		One_row[id] = new Matrix<floatType>(1, new_size);
		One_row[id]->alloc_Cuda_M();
		One_row[id]->Cuda_ACC_function(Device_Func::Assignment, false, 1);
	}
	else One_row[id]->col = new_size;
}

template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_ScaleOneCol(bool OneRow) {
	ui One_size = OneRow ? row : col;
	Row_Reset(One_size);
	if (OneRow) {
		return _gemm(1.0f, One_row[get_stm_id], this, 0.0f);
	}
	else {
		return _gemv(1.0f, this, One_row[get_stm_id]->Cuda_M, 0.0f);
	}
}

template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_T() {
	return _geam(&(const T&)1.0f, this, &(const T&)0.0f, this, (Matrix<T>*)NULL, CUBLAS_OP_T);
}
template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_Sum() {
	Matrix<T>* Out = Cuda_ACC_ScaleOneCol();
	Row_Reset(Out->row);
	auto ret = _gemv(1.0f, One_row[get_stm_id], Out->Cuda_M, 0.0f, (Matrix<T>*)NULL, Out->pitch / sz);

	Heap[get_stm_id].free(Out);
	return ret;
}

template<class T> const Matrix<T>* Matrix<T>::Cuda_ACC_ZeroMemory() {
	assert(row > 0 && col > 0);
	alloc_Cuda_M();
	cudaMemset2DAsync(Cuda_M, pitch, 0, WidthStride, row, get_stm);
	return this;
}

template<class T> void MatrixPtr<T>::Reset(ui row, ui col, bool WriteToDevice, bool pinned, bool auto_alloc_free) {
	if (ptr == NULL || ptr->row != row || ptr->col != col) {
		if (auto_alloc_free)
			Heap[get_stm_id].free(ptr), ptr = Heap[get_stm_id].malloc(row, col);
		else {
			delete ptr; ptr = new Matrix<T>(row, col);
		}
	}
	//only Write faster
	if (pinned && !ptr->M) {
		if (cudaHostAlloc(&ptr->M, Matrix<T>::sz * row * col, cudaHostAllocWriteCombined) != cudaSuccess)
			cout << "cudaHostAlloc error\n", assert(false);
	}
	if (WriteToDevice)
		ptr->alloc_Cuda_M();
}

#endif
//#else
////CPU
//
//
//template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_mul(Matrix<T>*R) {
//	//if (col != R->row)return NULL;
//	Matrix<T>* C = new Matrix<T>(row, R->col);
//	for (int i = 0; i < R->col; i++)
//		for (int j = 0; j < row; j++) {
//			double sum = 0;
//			for (int k = 0; k < col; k++)
//				sum += M[j*col + k] * R->M[k*R->col + i];
//			C->M[j*C->col + i] = sum;
//		}
//	return C;
//}
//template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_plus(Matrix<T>*R) {
//	Matrix<T>* C = new Matrix<T>(row, max(col, R->col));
//	if (col == R->col)
//		for (ui i = 0; i < col; i++)
//			for (ui j = 0; j < row; j++) {
//				C->M[j*col + i] = M[j*col + i] + R->M[j*col + i];
//			}
//	else for (ui i = 0; i < C->col; i++)
//		for (ui j = 0; j < C->row; j++) {
//			C->M[j*C->col + i] = M[j*col + min(i, col - 1)] + R->M[j*R->col + min(i, R->col - 1)];
//		}
//	return C;
//}
//template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_minus(Matrix<T>*R) {
//	Matrix<T>* C = new Matrix<T>(row, max(col, R->col));
//	for (int i = 0; i < col; i++)
//		for (int j = 0; j < row; j++) {
//			C->M[j*col + i] = M[j*col + i] - R->M[j*col + i];
//		}
//	return C;
//}
//template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_div_by_number(const Matrix<T>*R, bool mul) {
//	T number = R->M[0];
//	bool flag = (mul && (row == 1 && col == 1));
//	if (!mul)number = 1 / number;
//	else if (flag)number = M[0];
//	int col = flag ? R->col : this->col;
//	int row = flag ? R->row : this->row;
//	Matrix<T>* C = new Matrix<T>(row, col);
//	for (int i = 0; i < col; i++)
//		for (int j = 0; j < row; j++) {
//			if (flag)
//				C->M[j*col + i] = R->M[j*col + i] * number;
//			else C->M[j*col + i] = M[j*col + i] * number;
//		}
//	return C;
//}
//template<class T>const Matrix<T>* Matrix<T>::Cuda_ACC_minus_(Matrix<T>*R) {
//	//缩列
//	if (col < R->col) {
//		for (int i = 0; i < col; i++)
//			for (int j = 0; j < row; j++) {
//				for (int k = 0; k < R->col; k++)
//					M[j*col + i] -= R->M[j*R->col + k];
//			}
//	}
//	else for (int i = 0; i < col; i++)
//		for (int j = 0; j < row; j++) {
//			M[j*col + i] -= R->M[j*col + i];
//		}
//	return this;
//}
//template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_plus_(Matrix<T>*R) {
//	for (int i = 0; i < col; i++)
//		for (int j = 0; j < row; j++) {
//			M[j*col + i] += R->M[j*col + i];
//		}
//	return this;
//}
//template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_T() {
//	Matrix<T>* C = new Matrix<T>(col, row);
//	for (int i = 0; i < col; i++)
//		for (int j = 0; j < row; j++) {
//			C->M[i*C->col + j] = M[j*col + i];
//		}
//	return C;
//}
//template<class T> const Matrix<T>* Matrix<T>::Cuda_ACC_ZeroMemory() {
//	memset(M, 0, sizeof(T)*col*row);
//	return this;
//}
//template<class T> const Matrix<T>* Matrix<T>::Cuda_ACC_mul_Matrix(Matrix<T>*R) {
//	//同型
//	if (row == R->row&&col == R->col) {
//		for (int i = 0; i < col; i++)
//			for (int j = 0; j < row; j++) {
//				M[j*col + i] *= R->M[j*col + i];
//			}
//	}//一个数
//	else {
//		for (int i = 0; i < col; i++)
//			for (int j = 0; j < row; j++) {
//				M[j*col + i] *= R->M[0];
//			}
//	}
//	return this;
//}
//template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_function(int fun_id, bool TemporaryVariable) {
//	Matrix<T>*Out = this;
//	if (TemporaryVariable) {
//		Out = new Matrix<T>(row, col);
//	}
//	_fun f = d_f[fun_id];
//	for (int i = 0; i < col; i++)
//		for (int j = 0; j < row; j++) {
//			f(Out->M[j*col + i]);
//		}
//	return Out;
//}
//template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_Sum() {
//	Matrix<T>* C = new Matrix<T>(1, 1);
//	double Sum = 0;
//	for (int i = 0; i < row; i++)
//		for (int j = 0; j < col; j++) {
//			Sum += M[i*col + j];
//		}
//	C->M[0] = Sum;
//	return C;
//}
//template<class T>const Matrix<T>* Matrix<T>::Cuda_ACC_SoftMaxFunction() {
//	for (int i = 0; i < col; i++) {
//		T mx = M[0 * col + i], Sum = 0;
//		for (int j = 0; j < row; j++)
//			if (M[j*col + i] > mx)mx = M[j*col + i];
//		for (int j = 0; j < row; j++)
//			Sum += exp(M[j*col + i] - mx);
//		for (int j = 0; j < row; j++) {
//			M[j*col + i] = exp(M[j*col + i] - mx) / Sum;
//		}
//	}
//	return this;
//}
//template<class T> Matrix<T>* Matrix<T>::Cuda_ACC_assign(Matrix<T>*R) {
//	size_t sz = sizeof(T)*row*col;
//	memcpy_s(M, sz, R->M, sz);
//	return this;
//}
//template<class T> T* Matrix<T>::WriteToCuda() {
//	return Cuda_M;
//}
//template<class T> T* Matrix<T>::ReadFromCuda() {
//	return M;
//}
//
//#endif





/*
template<class T> MatrixPtr<T>& MatrixPtr<T>::operator=(MatrixPtr<T> Right) {
	//if(Right.getMatrix()->col!=Right.getMatrix()->col)
	if (CUDA_ACC_Enabled) {
		ptr->Cuda_ACC_assign(Right.getMatrix());
	}
	else {
		size_t sz = sizeof(T)*ptr->row*ptr->col;
		memcpy_s(ptr->M, sz, Right.getMatrix()->M, sz);
	}
	return *this;
}*/
#include<time.h>

//namespace Test {

void TestOps(Mat& A, Mat& B, Mat& C, int ops_id) {
	switch (ops_id)
	{
		//矩阵乘
	case 0:C = A % B;
		break;
		//同型矩阵自乘矩阵
	case 1:C = A *= B;
		break;
		//矩阵自乘数字
	case 2:
		C = A *= B;
		break;
		//矩阵加
	case 3:
		C = A + B;
		break;
		//矩阵加列
	case 4:
		C = A + B;
		break;
		//矩阵减矩阵
	case 5:
		C = A - B;
		break;
		//矩阵自减矩阵
	case 6:
		C = A -= B;
		break;
		//列自减矩阵
	case 7:
		C = A -= B;
		break;
		//矩阵除以数
	case 8:
		C = A / B;
		break;
		//矩阵函数
	case 9:
		C = A.f(0);
		break;
		//矩阵转置
	case 10:
		C = !A;
		//矩阵求和
		break;
	case 11:
		C = A.Sum();
		break;
		//矩阵自加矩阵
	case 12:
		C = A += B;
		break;
		//矩阵乘数
	case 13:
		C = A * B;
		break;
		//SoftMax_Func
	case 14:
		C = A.SoftMax();
		break;
	default:
		break;
	}
}
//#include<assert.h>
bool Console = true;
int randNum() {
	if (Console)
		return ::max(rand() % 5, 1);
	else return ::max(rand() % 5000, 1);
}
void randrc(int& a_r, int& a_c, int& b_r, int& b_c, int ops_id) {
	switch (ops_id)
	{
	case 0:a_r = randNum();
		a_c = b_r = randNum();
		b_c = randNum();
		break;
		//同型
	case 1:
	case 3:
	case 5:
	case 6:
	case 12:
		a_r = b_r = randNum();
		a_c = b_c = randNum();
		break;
	case 2:
	case 8:
	case 9:
	case 10:
	case 11:
	case 13:
	case 14:
		a_r = randNum();
		a_c = randNum();
		b_r = b_c = 1;
		break;
	case 4:
		a_r = b_r = randNum();
		a_c = randNum();
		b_c = 1;
		break;
	case 7:
		a_r = b_r = randNum();
		b_c = randNum();
		a_c = 1;
		break;
	default:
		break;
	}
}
void Test() {
	for (int id = 0; id < 1; id++) {
		int tar = -1;
		clock_t Sum[2000];
		memset(Sum, 0, sizeof(Sum));
		for (int t = 0; t < 2; t++) {
			int a_r, a_c, b_r, b_c;
			Console = true;
			randrc(a_r, a_c, b_r, b_c, id);
			//id = 14; a_r = 4000, a_c = 4883, b_r = 1, b_c = 1;
			//id = 7; a_r = 2, a_c = 1, b_r = 1, b_c = 3;
			Mat A(a_r, a_c), B(b_r, b_c);


			clock_t Start = clock();
			clock_t sum = 0, sum1 = 0, sum2 = 0;

			Mat C = Mat();
			Mat D = Mat();
			for (int i = 0; i < 1; i++) {
				A.RandData();
				B.RandData();
				//除以非0
				if (id == 8)
					B.getMatrix()->M[0]++;
				if (Console) {
					A.getMatrix()->ConsolePrint();
					B.getMatrix()->ConsolePrint();
				}
				A.getMatrix()->WriteToCuda();
				B.getMatrix()->WriteToCuda();
				/*
				Start = clock();
				double alpha = 1, beta = 1;
				ui row = A.getMatrix()->row, col = A.getMatrix()->col;
				Mat Out = Mat(row, col);
				Out.getMatrix()->alloc_Cuda_M();
				ui sz = sizeof(double);
				cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, col, row, &alpha, B.getMatrix()->Cuda_M, B.getMatrix()->pitch / sz, &beta, A.getMatrix()->Cuda_M,
				A.getMatrix()->pitch / sz,  Out.getMatrix()->Cuda_M, Out.getMatrix()->pitch / sz);
				cudaDeviceSynchronize();
				sum2 = clock() - Start;*/



				//CUDA_ACC_Enabled = false;
				Start = clock();
				TestOps(A, B, C, id);
				sum1 = clock() - Start;


				/*CUDA_ACC_Enabled = true;
				Start = clock();
				TestOps(A, B, D, id);
				sum = clock() - Start;*/


				C.getMatrix()->ReadFromCuda();
				//E.getMatrix()->ReadFromCuda();

				//if (C.getMatrix() && D.getMatrix()->operator==(C.getMatrix()->M)) {
				//	//cout << "ok\n";
				//}
				//else cout << "wrong\n";
				//if (C.getMatrix() && E.getMatrix()->operator==(C.getMatrix()->M)) {
					//cout << "ok\n";
				//}
				//else cout << "CPU wrong\n";
				if (Console) {
					if (C.getMatrix())C.getMatrix()->ConsolePrint();
					//D.getMatrix()->ConsolePrint();
					//E.getMatrix()->ConsolePrint();
				}
			}
			/*double res = 0;
			if (sum != 0)res = 1.0*sum1 / sum;
			printf("op:%d a_r:%d a_c:%d b_r:%d b_c:%d %0.2lfms %0.2lfms x%lf\n", id, a_r, a_c, b_r, b_c, 1000.0*sum / CLOCKS_PER_SEC, 1000.0*sum1 / CLOCKS_PER_SEC, res);
			printf("cublas:%0.2lfms\n", 1000.0*sum2 / CLOCKS_PER_SEC);*/
			//getchar();
			cout << endl;
		}
		getchar();
		continue;
		clock_t mi = 1e9 + 7; tar = -1;
		for (ui i = 32; i < 1024; i += 32)
			if (Sum[i] < mi)mi = Sum[i], tar = i;
		cout << tar << '\n';
	}
}
////#include<thread>
///*void StartThread(cudaStream_t*stm, Mat*A, Mat*B, int id) {
//	cublasSetStream(handle, stm[id]);
//	for (int i = 0; i < 100; i++)
//		*A * *B;
//}*/
using namespace _CUDA_;
__global__ void Cuda_Test() {
#define N 1000000
#pragma unroll 2
	register int a = 1, b = 7, c = 11, d = 2, e = 4, f = 3;
	for (int i = 0; i < N; i++) {
		a = a * b + c;
		d = d * b + c;
		//e = e * b + c;
		//f = f * b + c;
	}
}
int main() {
	dim3 grid(1024, 1, 1), block(1024, 1, 1);
	Cuda_Test << < grid, block >> > ();
#undef cudaDeviceSynchronize
	cudaDeviceSynchronize();

	return;
	Matrix_Set_Function(0, 0, 124);
	double _A[] = {
		1,2,
		3,4
	}, _B[] = {
		-1,-2,-3,
		-2,-3,-4
	};
	for (int i = 0; i < 2; i++) {
		Mat A(2, 2, _A), B(2, 3, _B);
		B = Mat(2, 2, _A);// % B;// -A;
		{
			Mat C = A.f(ReLU)._f(ReLU).f(ReLU);
			C.Print();
		}
	}
	//Mat One(100, 100, 1.0), Two(100, 100, 2.0);
	//clock_t Start = clock();
	//for (int i = 0; i < 10000; i++)
	//	One.ReadFromDevice();
	//cout << 1000.0*(clock() - Start) / CLOCKS_PER_SEC << '\n';
	//Start = clock();
	//Matrix<double>*p = One.getMatrix(), *p1 = Two.getMatrix();
	//for (int i = 0; i < 10000; i++) {
	//	cublasSetPointerMode(handle, cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE);
	//	cudaMemcpy2D(p1->Cuda_M, p->WidthStride, p->Cuda_M, p->pitch, p->WidthStride, p->row, cudaMemcpyDeviceToDevice);
	//	cublasSetPointerMode(handle, cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
	//}
	//	//One._f(Device_Func::One_div);
	//cout << 1000.0*(clock() - Start) / CLOCKS_PER_SEC << '\n';

	////CUDA_ACC_Enabled = false;
	//cudaDeviceProp prop;
	//int count;
	//cudaGetDeviceCount(&count);//获取设备数目，比如GTX295 有两个GPU（也就是双核） count为2
	//for (int i = 0; i<count; i++)
	//{
	//cudaGetDeviceProperties(&prop, i);//将第i个GPU数据放到prop中
	//std::cout << "显卡名称：" << prop.name << std::endl;
	//std::cout << "显存大小：" << prop.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
	//std::cout << "一个block的共享内存大小：" << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
	//std::cout << "block最大线程数：" << prop.maxThreadsPerBlock << std::endl;
	//}
	//system("pause");
	//
	////ptr-test set best core block param
	//srand(time(0));


#ifdef __GPU__
	//Matrix_Set_Function(NULL, NULL);
#endif


	//cudaOccupancyMaxActiveBlocksPerMultiprocessor()
	//cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceMapHost | cudaDeviceLmemResizeToMax);
	//Mat A(100, 100), B(100, 100);
	//A.RandData(); B.RandData();
	//A.WriteToDevice(); B.WriteToDevice();
	//int stm_cnt = 10;
	//cudaStream_t*stm =(cudaStream_t*)malloc(stm_cnt*sizeof(cudaStream_t));
	//for (int i = 0; i < stm_cnt; i++) {
	//	cudaStreamCreate(&stm[i]);
	//}
	//clock_t Start = clock();
	////StartThreads << <{1,1,1}, { 1000,1,1 } >> >(stm, A, B);
	//for (int i = 0; i < stm_cnt; i++) {
	//	cublasSetStream(handle, stm[i]);
	//	for (int j = 0; j < 100; j++) {
	//		A * B;
	//	}
	//}
	////280ms
	//for (int i = 0; i < stm_cnt; i++)
	//	cudaStreamSynchronize(stm[i]);
	//cudaDeviceSynchronize();
	//cout << 1000.0*(clock() - Start) / CLOCKS_PER_SEC << '\n';
	//Start = clock();
	////thread* T[10000];
	//for (int i = 0; i < stm_cnt; i++) {
	//	thread(StartThread, stm, &A, &B, i).detach();
	//	//T[i]->detach();
	//}
	////int cnt = 0;
	////cudaGetDeviceCount(&cnt);
	//for (int i = 0; i < stm_cnt; i++)
	//	cudaStreamSynchronize(stm[i]);
	//cudaDeviceSynchronize();
	////7864ms
	//cout << 1000.0*(clock() - Start) / CLOCKS_PER_SEC << '\n';


	//cudaDeviceReset();
	//cublasCreate(&handle);

	//Matrix_Set_Function((void**)&__f, (void**)_f1);

	//void*p;
	//if (cudaGetSymbolAddress(&p, __f) == cudaSuccess) {
		//Matrix_Set_Function((void**)tmp, (void**)_f1);
	//}
	/*
	Mat A(3, 3); A.RandData();
	A.getMatrix()->ConsolePrint();
	A.WriteToDevice();
	A.ZeroMemory();
	A.ReadFromDevice();
	A.getMatrix()->ConsolePrint();*/

	//Test();
	//getchar();






	//Matrix_Set_Function(f1);
	/*
	Mat G(3, 3), G1(3, 3);
	G.RandData();
	//G.getMatrix()->ConsolePrint();
	G.getMatrix()->WriteToCuda();
	//Matrix<double>*P=G.getMatrix()->Cuda_ACC_ScaleOneCol();
	//P->ReadFromCuda();
	G.getMatrix()->ConsolePrint();
	G1.ZeroMemory();
	rsize_t sz= sizeof(double);
	cudaMemcpy2D(G1.getMatrix()->Cuda_M, G1.getMatrix()->pitch, G.getMatrix()->Cuda_M, G.getMatrix()->pitch, sz, G.getMatrix()->row, cudaMemcpyDeviceToDevice);

	G1.getMatrix()->ReadFromCuda();
	G1.getMatrix()->ConsolePrint();
	*/
	/*
	Fun op;

	Func  p3 = &Fun::_f;
	void* p4 = (void*)((size_t&)p3);

	void*p = (void*)&((size_t&)(Fun::_f));
	_fun f = (_fun)( + 0x10);*/


	return 0;
}

#endif