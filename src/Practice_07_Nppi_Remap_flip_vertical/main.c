#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include "utils.h"

#include <nppi.h>
#include <cuda_runtime.h>

#define SAMPLE_IMG "../../resources/hwaseong_fortress_640x480.jpeg"

typedef struct _gpuMemory {
    void *memory;
    size_t pitch;
    size_t widthByte;
    size_t height;
} gpuMemory;

void imageReadToGPU(gpuMemory *gpuMem)
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

    size_t widthByte = imgWidth * 3; // 3 = Pixel size of RGB color space
    size_t height = imgHeight;

    // This compatable with memory from cudaMallocPitch
    printf("====== Step 2. nppiMalloc_8u_C3 ======\n");
    Npp8u *mem = NULL;
    int stepBytes = 0;
    mem = nppiMalloc_8u_C3(imgWidth, imgHeight, &stepBytes);

    printf("- (Debug print) pitch(Byte): %d, width(Byte): %d, height: %d\n", (int)stepBytes, (int)widthByte, (int)height);
    printf("====================================================\n\n");


    printf("====== Step 3. cudaMemcpy2D (Host To Device) ======\n");
    cudaMemcpy2D(mem,           // dst Buffer (Device)
        stepBytes,              // Pitch size of dst (devPtr)
        imgBuffer,              // src Buffer (Host)
        (size_t)(imgWidth * 3), // Pitch size of src (imgBuffer)
        (size_t)(imgWidth * 3), // Width size of src (imgBuffer)
        (size_t)imgHeight,
        cudaMemcpyHostToDevice); // Direction of copy (IMPORTANT)
    printf("====================================================\n\n");

    gpuMem->memory = mem;
    gpuMem->pitch = (size_t)stepBytes;
    gpuMem->widthByte = widthByte;
    gpuMem->height = height;

    free(imgBuffer);
}

void allocDestMemory(gpuMemory *gpuMem)
{
    printf("====== Step 4. nppiMalloc_8u_C3 ======\n");
    Npp8u *mem = NULL;
    int stepBytes = 0;
    mem = nppiMalloc_8u_C3(gpuMem->widthByte / 3, gpuMem->height, &stepBytes);
    printf("====================================================\n\n");

    gpuMem->memory = mem;
    gpuMem->pitch = (size_t)stepBytes;
}

void _createXMap(int width, int height, Npp32f **xMap, int *steps)
{
    printf("====== Step 5-1. Alloc CPU memory ======\n");
    float *cpuMem = (float *)malloc(sizeof(float) * width * height);
    printf("====================================================\n\n");
    
    printf("====== Step 5-2. Set X Map data ======\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float xCoordMap = (float)x; // Nothing to do
            cpuMem[y * width + x] = xCoordMap;
        }
    }
    printf("====================================================\n\n");

    printf("====== Step 5-3. Alloc GPU memory ======\n");
    Npp32f *map = nppiMalloc_32f_C1(width, height, steps);
    printf("====================================================\n\n");

    printf("====== Step 5-4. cudaMemcpy2D (Host To Device) ======\n");
    cudaMemcpy2D(map,           // dst Buffer (Device)
        (size_t)*steps,         // Pitch size of dst (devPtr)
        (void *)cpuMem,         // src Buffer (Host)
        (size_t)(width * sizeof(float)), // Pitch size of src (imgBuffer)
        (size_t)(width * sizeof(float)), // Width size of src (imgBuffer)
        (size_t)height,
        cudaMemcpyHostToDevice); // Direction of copy (IMPORTANT)
    printf("====================================================\n\n");

    *xMap = map;
    free(cpuMem);
}

void _createYMap(int width, int height, Npp32f **yMap, int *steps)
{
    printf("====== Step 6-1. Alloc CPU memory ======\n");
    float *cpuMem = (float *)malloc(sizeof(float) * width * height);
    printf("====================================================\n\n");
    
    printf("====== Step 6-2. Set Y Map data ======\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float yCoordMap = (float)(height - y - 1); // Invert data using inversion of Y coordinate
            cpuMem[y * width + x] = yCoordMap;
        }
    }
    printf("====================================================\n\n");

    printf("====== Step 6-3. Alloc GPU memory ======\n");
    Npp32f *map = nppiMalloc_32f_C1(width, height, steps);
    printf("====================================================\n\n");

    printf("====== Step 6-4. cudaMemcpy2D (Host To Device) ======\n");
    cudaMemcpy2D(map,           // dst Buffer (Device)
        (size_t)*steps,         // Pitch size of dst (devPtr)
        (void *)cpuMem,         // src Buffer (Host)
        (size_t)(width * sizeof(float)), // Pitch size of src (imgBuffer)
        (size_t)(width * sizeof(float)), // Width size of src (imgBuffer)
        (size_t)height,
        cudaMemcpyHostToDevice); // Direction of copy (IMPORTANT)
    printf("====================================================\n\n");

    *yMap = map;
    free(cpuMem);
}


void remapImageFlipHorizontal(gpuMemory *srcMem, gpuMemory *dstMem)
{
    int dstWidth = dstMem->widthByte / 3;
    int dstHeight = dstMem->height;

    printf("====== Step 5. Cteare X Map ======\n");
    Npp32f *xMap = NULL;
    int xMapSteps = 0;
    _createXMap(dstWidth, dstHeight, &xMap, &xMapSteps);
    printf("====================================================\n\n");

    printf("====== Step 6. Create Y Map ======\n");
    Npp32f *yMap = NULL;
    int yMapSteps = 0;
    _createYMap(dstWidth, dstHeight, &yMap, &yMapSteps);
    printf("====================================================\n\n");

    NppiSize srcSize = { (int)srcMem->widthByte / 3, (int)srcMem->height };
    NppiRect srcRoi = { 0, 0, srcSize.width, srcSize.height };
    NppiSize dstSize = { (int)dstMem->widthByte / 3, (int)dstMem->height };

    printf("====== Step 7. Try Remap (nppiRemap_8u_C3R) ======\n");
    NppStatus ret = nppiRemap_8u_C3R(srcMem->memory,
                     srcSize,
                     (int)srcMem->pitch,
                     srcRoi,
                     xMap,
                     xMapSteps,
                     yMap,
                     yMapSteps,
                     dstMem->memory,
                     (int)dstMem->pitch,
                     dstSize,
                     NPPI_INTER_LINEAR);
    if (ret)
        printf("Remap error (%d)\n", (int)ret);
    printf("====================================================\n\n");
    nppiFree(xMap);
    nppiFree(yMap);
}

void imageWriteToGPU(gpuMemory *gpuMem, const char *filepath)
{
    int width = (int)gpuMem->widthByte / 3;
    int height = (int)gpuMem->height;
    unsigned char *hostMem = (unsigned char *)malloc(sizeof(unsigned char) * width * height * 3);

    printf("====== Step 6. cudaMemcpy2D (Device To Host) ======\n");
    cudaMemcpy2D(hostMem,           // dst Buffer (Host)
        width * 3,                  // Pitch size of dst (hostMem)
        gpuMem->memory,             // src Buffer (Host)
        gpuMem->pitch,              // Pitch size of src
        gpuMem->widthByte,          // Width size of src
        gpuMem->height,
        cudaMemcpyDeviceToHost); // Direction of copy (IMPORTANT)
    printf("====================================================\n\n");

    printf("====== Step 7. write memory to jpg image ======\n");
    write_JPEG_file(filepath, 100, hostMem, width, height);
    printf("====================================================\n\n");

    free(hostMem);
}

int main()
{
    gpuMemory mem1 = { 0, };
    gpuMemory mem2 = { 0, };

    imageReadToGPU(&mem1);

    mem2.widthByte = mem1.widthByte;
    mem2.height = mem1.height;
    allocDestMemory(&mem2);

    remapImageFlipHorizontal(&mem1, &mem2);

    imageWriteToGPU(&mem2, "output.jpg");

    nppiFree(mem1.memory);
    nppiFree(mem2.memory);

    printf("- Origin image\n");
    printf("xdg-open %s\n", SAMPLE_IMG);
    printf("- Output image\n");
    printf("xdg-open %s\n", "output.jpg");

    return 0;
}
