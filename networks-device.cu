#include "networks-device.hu"

__global__ void gpu_Matsum(double* input1, double* input2, double* output, int N, int M) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N*M) {
        output[idx] = input1[idx] + input2[idx];
    }
}

__global__ void gpu_Dim1Sum(double* input, double* output, int N, int M) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < M) {
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += input[i * M + idx];
        }
        output[idx] = sum;
    }
}

__global__ void gpu_Matmul_element(double* input1, double* input2, double* output, int N, int M) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N * M) {
        output[idx] = input1[idx] * input2[idx];
    }
}

__global__ void gpu_Mat_scale(double* input1, double scale, double* output, int N, int M) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N * M) {
        output[idx] = input1[idx] * scale;
    }
}

__global__ void gpu_Matmul(double* input, double* weight, double* output, int N, int K, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += input[row * K + k] * weight[k * M + col];
        }
        output[row * M + col] = sum;
    }
}

__global__ void gpu_Dense_Bias_Calc(double* output, double* weight, int N, int K, int M) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N * M) {
        output[idx] += weight[K * M + idx % M];
    }
}

void gpu_Dense(double* input, double* weight, double* output, int N, int K, int M, bool bias) {

    dim3 block_size(32, 32);
    dim3 grid_size((M + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    gpu_Matmul<<<grid_size, block_size>>>(input, weight, output, N, K, M);
    cudaDeviceSynchronize();
    if (bias) {
        int bias_block_size = 1024;
        int bias_grid_size = (N * M + bias_block_size - 1) / bias_block_size;

        gpu_Dense_Bias_Calc<<<bias_grid_size, bias_block_size>>>(output, weight, N, K, M);     
        cudaDeviceSynchronize();
    }
}

__global__ void gpu_ReLU(double* input, double* output, double* masked, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        if (input[idx] > 0) {
            output[idx] = input[idx];
            masked[idx] = 1.0;
        } else {
            output[idx] = 0.0;
            masked[idx] = 0.0;
        }
    }
}

__global__ void gpu_Softmax(double* input, double* output, int batch, int N) {
    int b = blockIdx.x;
    int idx = threadIdx.x;

    if (b < batch && idx < N) {
        double max_val = -1e20;
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            max_val = max(max_val, input[b * N + i]);
        }
        for (int i = 0; i < N; i++) {
            sum += exp(input[b * N + i] - max_val);
        }
        output[b * N + idx] = exp(input[b * N + idx] - max_val) / sum;
    }
}

__global__ void gpu_NLL_Loss(double* pred, uint8_t* labels, double* output, int numclass, int batch_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch_size * numclass) {
        int b = index / numclass; 
        int i = index % numclass; 
        output[index] = pred[index] - (i == labels[b] ? 1.0 : 0.0);
    }
}