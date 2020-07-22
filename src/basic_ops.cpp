//this files includes all the operations that are used in an LSTM

#include "basic_ops.h"
void block_maltmul(hls::stream<blc_vec>& weights, hls::stream<blc_vec>& activation, b_mat& mult_out, DTYPE iteration)
{

	//the "Parallel Programing for FPGAs: The HLS Book" chapt 6 by by Ryan Kastner was used as reference
	#pragma HLS DATAFLOW
	int counter = iteration % (tot_B_num/mult_num_block);

	static DTYPE int_act[mult_num_block][tot_B_num]; // internal activation in my e.g a 2 by 4
	if(counter == 0) { //only load the input activations when necessary
	loadA:         for (int i = 0; i<tot_B_num; i++)
			{
				#pragma HLS unroll factor = 2
				blc_vec tempA = activation.read();
				for (int j = 0; j < mult_num_block; j++)
				{
					#pragma HLS PIPELINE II = 1
					int_act[j][i] = tempA.a[j];
				}
			}
	}

 	DTYPE int_part[mult_num_block][mult_num_block] = {0};
	partialmult : for (int k = 0; k<tot_B_num; k++)
	{
		blc_vec tempW = weights.read();
		for (int i = 0; i < mult_num_block; i++)
		{
			#pragma HLS PIPELINE II = 1
			for (int j = 0; j< mult_num_block; j++)
			{
				#pragma HLS PIPELINE II = 1
				int_part[i][j] = int_part[i][j] + int_act[i][k] * tempW.a[j];
			}
		}
	}

	writeoutput: for(int i = 0; i < mult_num_block; i++)
	{
		for (int j = 0; j<mult_num_block; j++)
		{
			#pragma HLS unroll factor = 2
			mult_out.out[i][j] = int_part[i][j];
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
			sum[i][j] = partial1[i][j] + partial2[i][j];
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
			product[i][j] = partial1[i][j] *  partial2[i][j];
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
			out_[i][j] = tanh(partial[i][j]);
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
			temp = (partial[i][j]);
  //since the HLS math library does not include sigmoid the exp  function from the library along with the sigmoid formula were used for the calculation
			out_[i][j] = 1/(1+exp(temp));
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
            c_t1[i][j] = c_t[i][j];
        }
    }
}

