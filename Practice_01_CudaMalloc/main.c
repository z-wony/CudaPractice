#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include "utils.h"

#include <cuda_runtime.h>

#define SAMPLE_IMG "../resources/hwaseong_fortress_640x480.jpeg"

void cudaMallocTest()
{
    printf("====== Step 1. Image read ======\n");
    unsigned char *imgBuffer = NULL;
    int imgWidth = 0;
    int imgHeight = 0;
    int ret = 0;

    ret = read_JPEG_file(SAMPLE_IMG, &imgBuffer, &imgWidth, &imgHeight);
    if (ret) {
        fprintf(stderr, "Image read error\n");
        return;
    }
    printf("- (Debug print) Jpeg image width: %d, height: %d\n", imgWidth, imgHeight);
    printf("====================================================\n\n");

    printf("====== Step 2. cudaMallocPitch ======\n");
    void *devPtr = NULL;
    size_t pitch = 0;
    size_t widthByte = imgWidth * 3; // 3 = Pixel size of RGB color space
    size_t height = imgHeight;
    cudaError_t cuRet;
    cuRet = cudaMallocPitch(&devPtr, &pitch, widthByte, height);
    if (cuRet) {
        fprintf(stderr, "Cuda malloc failed\n");
        free(imgBuffer);
        return;
    }

    printf("- (Debug print) pitch(Byte): %d, width(Byte): %d, height: %d\n", (int)pitch, (int)widthByte, (int)height);
    printf("====================================================\n\n");

    cudaFree(devPtr);
    free(imgBuffer);
}

int main()
{
    cudaMallocTest();

    return 0;
}
