#include <stdio.h>
#include <cudnn.h>

int main(int argc, char *argv[]) {
    printf("cuDNN Version:\n");
    printf("\tMajor: %d\n", CUDNN_MAJOR);
    printf("\tMinor: %d\n", CUDNN_MINOR);
    printf("\tPatch: %d\n", CUDNN_PATCHLEVEL);
    return 0;
}
