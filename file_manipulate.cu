#include "file_manipulate.hu"

uint32_t readBackward(uint32_t val) {
    return ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) | ((val & 0xFF0000) >> 8) | ((val >> 24) & 0xFF);
}

void readDataset(char* loc, uint8_t* &data, uint32_t* &dataSize, dim3* &dataDim) {
    FILE* file = fopen(loc, "rb");

    if (file == NULL) {
        exit(EXIT_FAILURE);
    }

    uint32_t magicNumber;
    fread(&magicNumber, sizeof(uint32_t), 1, file);
    magicNumber = readBackward(magicNumber);

    if (magicNumber == 2051) {
        printf("Reading image file from: %s\n", loc);
    } else if (magicNumber == 2049) {
        printf("Reading label file from: %s\n", loc);
    } else {
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // uint32_t numItems;
    dataSize = (uint32_t *) malloc(sizeof(uint32_t));
    fread(dataSize, sizeof(uint32_t), 1, file);
    *dataSize = readBackward(*dataSize);
    printf("Number of items: %u\n", *dataSize);

    // *dataSize = numItems;

    if (magicNumber == 2051) {
        uint32_t numRows, numCols;
        fread(&numRows, sizeof(uint32_t), 1, file);
        fread(&numCols, sizeof(uint32_t), 1, file);
        numRows = readBackward(numRows);
        numCols = readBackward(numCols);

        printf("Image dimensions: %u x %u\n", numRows, numCols);

        // Allocate memory for reading an image
        data = (uint8_t *)malloc(numRows * numCols * (*dataSize));
        if (data == NULL) {
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // Read and display the first image as an example
        fread(data, numRows * numCols * (*dataSize), 1, file);
        // *dataDim = {numRows, numCols, 1};

    } else if (magicNumber == 2049) { // Label file
        // Allocate memory for reading all labels
        data = (uint8_t *)malloc((*dataSize));
        if (data == NULL) {
            perror("Error allocating memory");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // Read all labels
        fread(data, (*dataSize), 1, file);

        // Display the first 10 labels as an example
        printf("First 10 labels:\n");
        for (uint32_t i = 0; i < 10 && i < (*dataSize); i++) {
            printf("%u ", data[i]);
        }
        printf("\n");
    }

    fclose(file);
}

void readWeights(char* loc, double** &weights) {

}

void saveWeights(char* loc, double** &weights) {
    
}