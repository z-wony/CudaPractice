#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include "utils.h"

#include <nppi.h>

#define SAMPLE_IMG "../../resources/hwaseong_fortress_640x480.jpeg"

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

    printf("====== Step 2. nppiMalloc_8u_C3 ======\n");
    Npp8u *mem = NULL;
    int stepBytes = 0;
    mem = nppiMalloc_8u_C3(imgWidth, imgHeight, &stepBytes);

    printf("- (Debug print) steps(Byte): %d, width: %d, height: %d\n", stepBytes, imgWidth, imgHeight);
    printf("====================================================\n\n");

    nppiFree(mem);
    free(imgBuffer);
}

int main()
{
    cudaMallocTest();

    return 0;
}
