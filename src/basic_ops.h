#ifndef basic_ops_H_
#define basic_ops_H_
#include "hls_stream.h"
#include <hls_math.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <ap_cint.h>

using namespace std;

typedef int8 DTYPE;
const   int tot_B_num = 4; // number of colums/rows of the activations $ weights
//the matrix for activations/weights will be (tot_B_num by ot_B_num)
const   int mult_num_block = 2; // number of blocks fetched per multiplication

typedef struct{
	DTYPE inputs[mult_num_block];} blc_vec; //block vector with tot_B_num elements

typedef struct{
	DTYPE a[mult_num_block][mult_num_block];} b_mat;
//b_mat = the block of matrix multiply that will be computed 2 by 2 in our example

typedef struct{
	DTYPE in[mult_num_block][total_B_num];} int_mem;

//matrix multiplication
void block_maltmul(hls::stream<blc_vec>& weights, int_act& activation, b_mat & mult_out);

//addition
void addition(b_mat& partial1, b_mat& partial2, b_mat& sum);

//element wise multiplication
void e_mult( b_mat& partial1, b_mat& partial2, b_mat& product);

//tanh function
void tanh_(b_mat& partial, b_mat& out_);

//sigmoid function, I believe the HLS math library does not have a sigmoid function
void sigmoid(b_mat& partial, b_mat& out_);
void transfer_data ( b_mat& c_t, b_mat& c_t1);


#endif

