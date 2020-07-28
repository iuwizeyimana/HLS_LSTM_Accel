//this files includes all the operations that are used in an LSTM

#include "basic_ops.h"
void block_maltmul(hls::stream<blc_vec>& weights, int_mem& activation, b_mat& mult_out)
{

	//the "Parallel Programing for FPGAs: The HLS Book" chapt 6 by by Ryan Kastner was used as reference
	#pragma HLS DATAFLOW

	partialmult : for (int k = 0; k<tot_B_num; k++)
	{
		blc_vec tempW = weights.read();
		for (int i = 0; i < mult_num_block; i++)
		{
			#pragma HLS PIPELINE 
			for (int j = 0; j< mult_num_block; j++)
			{
				#pragma HLS PIPELINE 
				mult_out.a[i][j] = mult_out.a[i][j] + activation.in[i][k] * tempW.inputs[j];
			}
		}
	}


}


void addition(b_mat& partial1, b_mat& partial2, b_mat& sum)
{
	//DTYPE S[mult_num_block][mult_num_block] = {0};

	for(int i = 0; i< mult_num_block; i++)
	{
		#pragma HLS PIPELINE
		for(int j = 0; j< mult_num_block; j++)
		{
		// a loop fission might also be of help here
			#pragma HLS unroll factor = 2
			#pragma HLS PIPELINE II = 1
			sum.a[i][j] = partial1.a[i][j] + partial2.a[i][j];
		}
	}


}
void e_mult(b_mat& partial1, b_mat& partial2, b_mat& product)
{
//element wise multiplication
	for(int i = 0; i< mult_num_block; i++)
	{
		#pragma HLS PIPELINE
		for(int j = 0; j< mult_num_block; j++)
		{
		// a loop fission might also be of help here
			#pragma HLS unroll factor = 2
			#pragma HLS PIPELINE II = 1
			product.a[i][j] = partial1.a[i][j] *  partial2.a[i][j];
		}
	}


}
void tanh_(b_mat& partial, b_mat& out_)
{

	for(int i = 0; i< mult_num_block; i++)
	{
		#pragma HLS PIPELINE
		for(int j = 0; j< mult_num_block; j++)
		{
		// a loop fission might also be of help here
			#pragma HLS unroll factor = 2
			#pragma HLS PIPELINE
            //this tanh function is in the HLS math library
			out_.a[i][j] = tanh.a(partial[i][j]);
		}
	}


}
void sigmoid(b_mat& partial, b_mat& out_)
{
	DTYPE temp = 0;

	for(int i = 0; i< mult_num_block; i++)
	{
		#pragma HLS PIPELINE
		for(int j = 0; j< mult_num_block; j++)
		{
		// a loop fission might also be of help here
			#pragma HLS PIPELINE
			temp = (partial.a[i][j]);
  //since the HLS math library does not include sigmoid the exp  function from the library along with the sigmoid formula were used for the calculation
			out_.a[i][j] = 1/(1+exp(temp));
		}
	}


}

void transfer_data ( b_mat& c_t, b_mat& c_t1)
{
    for(int i = 0; i< mult_num_block; i++)
    {
        #pragma HLS PIPELINE
        for(int j = 0; j< mult_num_block; j++)
        {
            #pragma HLS PIPELINE
            c_t1.a[i][j] = c_t.a[i][j];
        }
    }
}

