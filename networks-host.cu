#include "networks-host.hu"

__host__ void cpu_Matsum(double* input1, double* input2, double* output, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            output[i * M + j] = input1[i * M + j] + input2[i * M + j];
        }
    }
}

__host__ void cpu_Dim1Sum(double* input, double* output, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            output[j] += input[i * M + j];
        }
    }
}

__host__ void cpu_Matmul_element(double* input1, double* input2, double* output, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            output[i * M + j] = input1[i * M + j] * input2[i * M + j];
        }
    }
}

__host__ void cpu_Mat_scale(double* input1, double scale, double* output, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            output[i * M + j] = input1[i * M + j] * scale;
        }
    }
}

__host__ void cpu_Matmul(double* input, double* weight, double* output, int N, int K, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            output[i * M + j] = 0.0f;
            for (int k = 0; k < K; k++) {
                output[i * M + j] += input[i * K + k] * weight[k * M + j];
            }
        }
    }
}

__host__ void cpu_Dense(double* input, double* weight, double* output, int N, int K, int M, bool bias) {
    cpu_Matmul(input, weight, output, N, K, M);

    if (bias)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            output[i * M + j] += weight[K * M + j];
        }
    }
}

__host__ void cpu_ReLU(double* input, double* output, double* masked, int N) {
    for (int i = 0; i < N; i++) {
        if (input[i] > 0) {
            output[i] = input[i];
            masked[i] = 1;
        } else {
            output[i] = masked[i] = 0;
        }
    }
}

__host__ void cpu_Softmax(double* input, double* output, int batch, int N) {
    for (int b = 0; b < batch; b++) {
        double sum = 1e-6;
        for (int i = 0; i < N; i++) {
            sum += exp(input[i + b * N]);
        }
        for (int i = 0; i < N; i++) {
            output[i + b * N] = exp(input[i + b * N]) / sum;
        }
    }
}

__host__ void cpu_NLL_Loss(double* pred, uint8_t* labels, double* output, int numclass, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < numclass; i++) {
            output[i + b * numclass] = pred[i + b * numclass] - (i == labels[b] ? 1.0 : 0.0);
        }
    }
}
