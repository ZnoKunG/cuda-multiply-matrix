
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void multiplyMatrix(float* A, float* B, float* C, int N)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0.0f;

	if (row >= N || col >= N) return;

	for (int i = 0; i < N; i++)
	{
		sum += A[N * row + i] * B[N * i + col];
	}

	C[row * N + col] = sum;
}

int main()
{
	int N = 1024;
	float* a = new float[N * N];
	float* b = new float[N * N];
	float* c = new float[N * N];
	float* c_CPU = new float[N * N];

	// Generate dummy data in metrix
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a[i * N + j] = sinf(i);
			b[i * N + j] = cosf(i);
		}
	}

	// Initialize timer
	clock_t start, end;
	double cpu_time_used;
	double gpu_time_used;

	// CPU execution
	//---------------------------
	start = clock();
	std::cout << "Performing " << N << "x" << N << " matrix multiplation on CPU ..." << std::endl;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < N; k++)
			{
				sum += a[N * i + k] * b[N * k + j];
			}

			c_CPU[i * N + j] = sum;
		}
	}
	end = clock();
	std::cout << "Task completed" << std::endl;
	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
	std::cout << "CPU Execution time: " << cpu_time_used << std::endl;
	// ---------------------------

	// GPU Execution
	// -------------------------------------
	start = clock();
	std::cout << "Performing " << N << "x" << N << " matrix multiplation on GPU ..." << std::endl;

	// Initialize pointers in GPU
	float* cudaA = 0;
	float* cudaB = 0;
	float* cudaC = 0;

	// Allocate memory by size of the matrix
	cudaMalloc(&cudaA, sizeof(a));
	cudaMalloc(&cudaB, sizeof(b));
	cudaMalloc(&cudaC, sizeof(c));

	// Copy data into the memory
	cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaB, b, sizeof(b), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	int nBlocks = ceil(N / BLOCK_SIZE);
	dim3 blocksPerGrid(nBlocks, nBlocks);

	multiplyMatrix <<< blocksPerGrid, threadsPerBlock >>> (cudaA, cudaB, cudaC, N);

	cudaMemcpy(c, cudaC, sizeof(c), cudaMemcpyDeviceToHost);
	end = clock();
	std::cout << "Task completed" << std::endl;

	gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
	std::cout << "GPU Execution time: " << gpu_time_used << std::endl;

	int i;
	std::cin >> i;
	return 0;
}
