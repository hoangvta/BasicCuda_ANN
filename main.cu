#include "file_manipulate.hu"
#include "networks.hu"

int main(int argc, char** argv) {

    uint8_t* images, *labels;
    uint32_t* image_len, *label_len;
    dim3* image_dim, *label_dim;
 
    readDataset("train_images", images, image_len, image_dim);
    readDataset("train_labels", labels, label_len, label_dim);

    double** weights;
    weights = (double**) malloc(3 * sizeof(double**));
    weights[0] = (double*) malloc((784 + 1) * 128 * sizeof(double));
    weights[1] = (double*) malloc((128 + 1) * 128 * sizeof(double));
    weights[2] = (double*) malloc((128 + 1) * 10 * sizeof(double));

    // weight init
    for (int i = 0; i < 785; i++) {
        for (int j = 0; j < 128; j++) {
            weights[0][i * 128 + j] = 0.01;
        }
    }

    for (int i = 0; i < 129; i++) {
        for (int j = 0; j < 128; j++) {
            weights[1][i * 128 + j] = 0.01;
        }
    }

    for (int i = 0; i < 129; i++) {
        for (int j = 0; j < 5; j++) {
            weights[2][i * 10 + j] = 0.01;
        }
    }

    for (int i = 0; i < 100; i++) {
        fit(1, images, weights, labels, *image_len);
        printf("\n");
    }

    free(images);
    free(labels);
    free(image_len);
    free(label_len);

    free(weights[0]);
    free(weights[1]);
    free(weights[2]);
    free(weights);

    return 0;
}