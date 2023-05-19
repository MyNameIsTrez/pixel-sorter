// Thread block size
#define BLOCK_SIZE %(block_size)d

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WIDTH_A %(w_a)d // Matrix A width
#define WIDTH_B %(w_b)d // Matrix B width

#define A_GET(x, y) A_LOCAL[y * BLOCK_SIZE + x]
#define B_GET(x, y) B_LOCAL[y * BLOCK_SIZE + x]

__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
void
matmul(
    __global float* C,
    __global float* A,
    __global float* B
) {
    __local float A_LOCAL[BLOCK_SIZE * BLOCK_SIZE];
    __local float B_LOCAL[BLOCK_SIZE * BLOCK_SIZE];

    // Block index
    int block_x = get_group_id(0);
    int block_y = get_group_id(1);

    // Thread index
    int thread_x = get_local_id(0);
    int thread_y = get_local_id(1);

    // Index of the first sub-matrix of A processed by the block
    int a_begin = WIDTH_A * BLOCK_SIZE * block_y;

    // Index of the last sub-matrix of A processed by the block
    int a_end   = a_begin + WIDTH_A - 1;

    // Step size used to iterate through the sub-matrices of A
    int a_step  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int b_begin = BLOCK_SIZE * block_x;

    // Step size used to iterate through the sub-matrices of B
    int b_step  = BLOCK_SIZE * WIDTH_B;

    // c_sub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float c_sub = 0.0f;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = a_begin, b = b_begin;
             a <= a_end;
             a += a_step, b += b_step) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        A_GET(thread_x, thread_y) = A[a + WIDTH_A * thread_y + thread_x];
        B_GET(thread_x, thread_y) = B[b + WIDTH_B * thread_y + thread_x];

        // Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            c_sub += A_GET(k, thread_y) * B_GET(thread_x, k);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = c_sub;
}
