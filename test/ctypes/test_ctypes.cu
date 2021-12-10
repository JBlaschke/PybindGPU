#include <stdio.h>
#include <iostream>
#include <cuda_hip_wrapper.h>


// These functions are intended to link to ctype objects, so we need to specify
// c-style linkage here
extern "C" {

void print_address(void * data_ptr) {
    printf("Address of data_ptr is %p\n", data_ptr);
}


void test_ptr_int(int data_ptr[], size_t n) {
    std::cout << "data_ptr = {";
    for (int i = 0; i < n; i++) {
        int data = data_ptr[i];
        std::cout << data << ",";
    }
    std::cout << "}" << std::endl;
}


__global__
void test_ptr_int_kernel(size_t data_ptr[], size_t n) {
    printf("dev_data_ptr = {");
    for (int i = 0; i < n; i++) {
        int data = data_ptr[i];
        printf("%d,", data);
    }
    printf("}\n");
}

void test_ptr_int_cuda(size_t data_ptr[], size_t n) {
    test_ptr_int_kernel<<<1, 1>>>(data_ptr, n);
}

}