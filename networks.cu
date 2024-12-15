#include "networks.hu"

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}


void input_transform(uint8_t* data, double* &input, int N) {
    for (int i = 0; i < N; i++) {
        input[i] = 1.0f * data[i] / 255;
    }
}

__global__ void call(double*item, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%.3f ", item[i * M + j]);
        }
        printf("\n");
    }
}

void fit(int mode, uint8_t* data, double** &weights, uint8_t* labels, uint32_t dataSize, double mu) {

    // flatten & transform
    double* input;
    input = (double*) malloc(dataSize * 28 * 28 * sizeof(double));
    input_transform(data, input, 28 * 28 * dataSize);

    if (mode == 0) { // Proceed CPU mode

        // Forward

        // layer 1
        double* output1;
        output1 = (double*) malloc(dataSize * 128 * sizeof(double));
        double* masked1;
        masked1 = (double*) malloc(dataSize * 128 * sizeof(double));

        cpu_Dense(input, weights[0], output1, dataSize, 28 * 28, 128);
        cpu_ReLU(output1, output1, masked1, dataSize * 128);

        // layer 2
        double* output2;
        output2 = (double*) malloc(dataSize * 128 * sizeof(double));
        double* masked2;
        masked2 = (double*) malloc(dataSize * 128 * sizeof(double));

        cpu_Dense(output1, weights[1], output2, dataSize, 128, 128);
        cpu_ReLU(output2, output2, masked2, dataSize * 128);

        // layer 3
        double* output3;
        output3 = (double*) malloc(dataSize * 10 * sizeof(double));

        cpu_Dense(output2, weights[2], output3, dataSize, 128, 10);
        cpu_Softmax(output3, output3, dataSize, 10);

        // Loss calculation
        double* loss;
        loss = (double*) malloc(dataSize * 10 * sizeof(double));

        cpu_NLL_Loss(output3, labels, loss, 10, dataSize);

        for (int b = 0; b < 1; b++) {
            for (int i = 0; i < 10; i++) {
                printf("%f ", output3[i + 10 * b]);
            }
            printf("\n");
        }
        printf("--------\n");

        // Backwards
        cpu_Mat_scale(loss, -mu, loss, dataSize, 10); // scaling loss with lr
        // layer 3 backprob

        double* w3_grad;
        w3_grad = (double*) malloc(128 * 10 * sizeof(double));

        cpu_Matmul(output2, loss, w3_grad, 128, dataSize, 10);

        cpu_Matsum(weights[2], w3_grad, weights[2], 128, 10);

        cpu_Dim1Sum(loss, weights[2] + 128 * 10, dataSize, 10);

        // layer 2 backprob
        double* w2_grad;
        double* tmp;

        w2_grad = (double*) malloc(128 * 128 * sizeof(double));
        tmp = (double*) malloc(dataSize * 128 * sizeof(double));

        cpu_Matmul(loss, weights[2], tmp, dataSize, 10, 128);
        cpu_Matmul(output1, tmp, w2_grad, 128, dataSize, 128);
        cpu_Matmul_element(w2_grad, masked2, w2_grad, 128, 128);

        cpu_Matsum(weights[1], w2_grad, weights[1], 128, 128);
        cpu_Dim1Sum(tmp, weights[1] + 128 * 128, dataSize, 128);

        // layer 1 backprob
        double* w1_grad;
        double* tmp2;

        w1_grad = (double*) malloc(28*28 * 128 * sizeof(double));
        tmp2 = (double*) malloc(dataSize * 28*28 * sizeof(double));

        cpu_Matmul(tmp, weights[1], tmp2, dataSize, 128, 128);
        cpu_Matmul(input, tmp2, w1_grad, 28*28, dataSize, 128);
        cpu_Matmul_element(w1_grad, masked1, w1_grad, 28*28, 128);

        cpu_Matsum(weights[0], w1_grad, weights[0], 28*28, 128);
        cpu_Dim1Sum(tmp2, weights[0] + 28*28 * 128, dataSize, 128);

        // free memories
        free(tmp);
        free(tmp2);
        free(masked1);
        free(masked2);
        free(w3_grad);
        free(w2_grad);
        free(w1_grad);

        free(output1);
        free(output2);
        free(output3);
        free(loss);
        

    } else if (mode == 1) { // Proceed GPU mode
        // Forward
        double* d_input;
        CHECK(cudaMalloc(&d_input, dataSize * 28 * 28 * sizeof(double)));
        CHECK(cudaMemcpy(d_input, input, 28 * 28 * dataSize * sizeof(double), cudaMemcpyHostToDevice));

        uint8_t* d_labels;
        CHECK(cudaMalloc(&d_labels, dataSize * sizeof(uint8_t)));
        CHECK(cudaMemcpy(d_labels, labels, dataSize * sizeof(uint8_t), cudaMemcpyHostToDevice));

        double *d_output1, *d_masked1;
        CHECK(cudaMalloc(&d_output1, dataSize * 128 * sizeof(double)));
        CHECK(cudaMalloc(&d_masked1, dataSize * 128 * sizeof(double)));

        double *d_weight0;
        double *d_weight1;
        double *d_weight2;
        CHECK(cudaMalloc(&d_weight0, (28*28+1) * 128 * sizeof(double)));
        CHECK(cudaMalloc(&d_weight1, 129 * 128 * sizeof(double)));
        CHECK(cudaMalloc(&d_weight2, 129 * 10 * sizeof(double)));
        CHECK(cudaMemcpy(d_weight0, weights[0], (28*28+1) * 128 * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_weight1, weights[1], 129 * 128 * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_weight2, weights[2], 129 * 10 * sizeof(double), cudaMemcpyHostToDevice));

        gpu_Dense(d_input, d_weight0, d_output1, dataSize, 28 * 28, 128);

        gpu_ReLU<<<(dataSize * 128 + 1023) / 1024, 1024>>>(d_output1, d_output1, d_masked1, dataSize * 128);
        CHECK(cudaDeviceSynchronize());

        double *d_output2, *d_masked2;
        CHECK(cudaMalloc(&d_output2, dataSize * 128 * sizeof(double)));
        CHECK(cudaMalloc(&d_masked2, dataSize * 128 * sizeof(double)));

        gpu_Dense(d_output1, d_weight1, d_output2, dataSize, 128, 128);
        
        gpu_ReLU<<<(dataSize * 128 + 1023) / 1024, 1024>>>(d_output2, d_output2, d_masked2, dataSize * 128);
        CHECK(cudaDeviceSynchronize());

        double* d_output3;
        CHECK(cudaMalloc(&d_output3, dataSize * 10 * sizeof(double)));

        gpu_Dense(d_output2, d_weight2, d_output3, dataSize, 128, 10);
        
        gpu_Softmax<<<dataSize, 10>>>(d_output3, d_output3, dataSize, 10);
        CHECK(cudaDeviceSynchronize());
        call<<<1, 1>>>(d_output3, 1, 10);
        CHECK(cudaDeviceSynchronize());
        
        // Loss
        double* d_loss;
        CHECK(cudaMalloc(&d_loss, dataSize * 10 * sizeof(double)));
        CHECK(cudaDeviceSynchronize());
        gpu_NLL_Loss<<<dataSize * 10, 1>>>(d_output3, d_labels, d_loss, 10, dataSize);
        call<<<1, 1>>>(d_loss, 1, 10);
        CHECK(cudaDeviceSynchronize());
        printf("\n-----\n");

        CHECK(cudaDeviceSynchronize());

        // Backward
        gpu_Mat_scale<<<(dataSize * 10 + 1023) / 1024, 1024>>>(d_loss, -mu, d_loss, dataSize, 10);
        CHECK(cudaDeviceSynchronize());

        double* d_w3_grad;
        CHECK(cudaMalloc(&d_w3_grad, 128 * 10 * sizeof(double)));
        gpu_Matmul<<<dim3((10 + 31) / 32, (128 + 31) / 32), dim3(32, 32)>>>(d_output2, d_loss, d_w3_grad, 128, dataSize, 10);
        CHECK(cudaDeviceSynchronize());
        gpu_Matsum<<<(128 * 10 + 1023) / 1024, 1024>>>(d_weight2, d_w3_grad, d_weight2, 128, 10);
        CHECK(cudaDeviceSynchronize());

        double *d_tmp;
        CHECK(cudaMalloc(&d_tmp, dataSize * 128 * sizeof(double)));
        gpu_Matmul<<<dim3((128 + 31) / 32, (128 + 31) / 32), dim3(32, 32)>>>(d_loss, d_weight2, d_tmp, dataSize, 10, 128);
        CHECK(cudaDeviceSynchronize());

        double* d_w2_grad;
        CHECK(cudaMalloc(&d_w2_grad, 128 * 128 * sizeof(double)));
        gpu_Matmul<<<dim3((128 + 31) / 32, (128 + 31) / 32), dim3(32, 32)>>>(d_output1, d_tmp, d_w2_grad, 128, dataSize, 128);
        CHECK(cudaDeviceSynchronize());
        gpu_Matmul_element<<<(128 * 128 + 1023) / 1024, 1024>>>(d_w2_grad, d_masked2, d_w2_grad, 128, 128);
        CHECK(cudaDeviceSynchronize());
        gpu_Matsum<<<(128 * 128 + 1023) / 1024, 1024>>>(d_weight1, d_w2_grad, d_weight1, 128, 128);
        CHECK(cudaDeviceSynchronize());

        double* d_tmp2;
        CHECK(cudaMalloc(&d_tmp2, dataSize * 28 * 28 * sizeof(double)));
        gpu_Matmul<<<dim3((128 + 31) / 32, (128 + 31) / 32), dim3(32, 32)>>>(d_tmp, d_weight1, d_tmp2, dataSize, 128, 28 * 28);
        CHECK(cudaDeviceSynchronize());

        double* d_w1_grad;
        CHECK(cudaMalloc(&d_w1_grad, 28 * 28 * 128 * sizeof(double)));
        gpu_Matmul<<<dim3((28 * 28 + 31) / 32, (128 + 31) / 32), dim3(32, 32)>>>(d_input, d_tmp2, d_w1_grad, 28 * 28, dataSize, 128);
        CHECK(cudaDeviceSynchronize());
        gpu_Matmul_element<<<(28 * 28 * 128 + 1023) / 1024, 1024>>>(d_w1_grad, d_masked1, d_w1_grad, 28 * 28, 128);
        CHECK(cudaDeviceSynchronize());
        gpu_Matsum<<<(28 * 28 * 128 + 1023) / 1024, 1024>>>(d_weight0, d_w1_grad, d_weight0, 28 * 28, 128);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(weights[0], d_weight0, (28*28+1) * 128 * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(weights[1], d_weight1, 129 * 128 * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(weights[2], d_weight2, 129 * 10 * sizeof(double), cudaMemcpyDeviceToHost));

        // Free GPU memory
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output1));
        CHECK(cudaFree(d_output2));
        CHECK(cudaFree(d_output3));
        CHECK(cudaFree(d_weight0));
        CHECK(cudaFree(d_weight1));
        CHECK(cudaFree(d_weight2));
        CHECK(cudaFree(d_loss));
        CHECK(cudaFree(d_masked1));
        CHECK(cudaFree(d_masked2));
        CHECK(cudaFree(d_w3_grad));
        CHECK(cudaFree(d_w2_grad));
        CHECK(cudaFree(d_w1_grad));
        CHECK(cudaFree(d_tmp));
        CHECK(cudaFree(d_tmp2));
    }
    
    free(input);
}