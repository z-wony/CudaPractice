#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include "utils.h"

#include <cuda_runtime.h>

void fillColorToROI(unsigned char *img, int stepBytes, int x, int y, int w, int h, int r, int g, int b)
{
    for (int yy = 0; yy < h; yy++) {
        for (int xx = 0; xx < w; xx++) {
            img[((yy + y) * stepBytes) + ((xx + x) * 3) + 0] = (unsigned char)r;
            img[((yy + y) * stepBytes) + ((xx + x) * 3) + 1] = (unsigned char)g;
            img[((yy + y) * stepBytes) + ((xx + x) * 3) + 2] = (unsigned char)b;
        }
    }
}

void createColorRectImage(const char *filepath)
{
    printf("====== Step 1. Image writing ======\n");
    int imgWidth = 640;
    int imgHeight = 480;
    unsigned char *imgBuffer = (unsigned char *)malloc(sizeof(unsigned char) * imgWidth * imgHeight * 3);

    /////////////////
    // RED  //     //
    /////////////////
    //      //     //
    /////////////////
    fillColorToROI(imgBuffer, imgWidth * 3, 0, 0, imgWidth / 2, imgHeight / 2, 255, 0, 0);
    /////////////////
    //      // BLUE//
    /////////////////
    //      //     //
    /////////////////
    fillColorToROI(imgBuffer, imgWidth * 3, imgWidth / 2, 0, imgWidth / 2, imgHeight / 2, 0, 0, 255);
    /////////////////
    //      //     //
    /////////////////
    //GREEN //     //
    /////////////////
    fillColorToROI(imgBuffer, imgWidth * 3, 0, imgHeight / 2, imgWidth / 2, imgHeight / 2, 0, 255, 0);
    /////////////////
    //      //     //
    /////////////////
    //      //BLACK//
    /////////////////
    fillColorToROI(imgBuffer, imgWidth * 3, imgWidth / 2, imgHeight / 2, imgWidth / 2, imgHeight / 2, 0, 0, 0);

    write_JPEG_file(filepath, 100, imgBuffer, imgWidth, imgHeight);
    printf("====================================================\n\n");
    free(imgBuffer);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Please input with 'filepath'\n");
        return -1;
    }

    createColorRectImage(argv[1]);

    return 0;
}
