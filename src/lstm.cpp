#include "block_lstm.h"

void LSTM(
	  hls::stream<b_vec>& x_t1,   // first input block
	  hls::stream<b_vec>& x_t2,   // second input block
	  hls::stream<b_vec>& h_t1_1, // first h(t-1) block
	  hls::stream<b_vec>& h_t1_2, // second  h(t-1) block
	  hls::stream<b_vec>& w_xo_1, // first weight blocks
	  hls::stream<b_vec>& w_ho_1,
	  hls::stream<b_vec>& w_xi_1,
	  hls::stream<b_vec>& w_hi_1,
	  hls::stream<b_vec>& w_xc_1,
	  hls::stream<b_vec>& w_hc_1,
	  hls::stream<b_vec>& h_f_1,
	  hls::stream<b_vec>& x_f_1,

	  hls::stream<b_mat>& h_t_1,
	  hls::stream<b_vec>& w_xo_2, // seconf weight blocks
	  hls::stream<b_vec>& w_ho_2,
	  hls::stream<b_vec>& w_xi_2,
	  hls::stream<b_vec>& w_hi_2,
	  hls::stream<b_vec>& w_xc_2,
	  hls::stream<b_vec>& w_hc_2,
	  hls::stream<b_vec>& h_f_2,
	  hls::stream<b_vec>& x_f_2,
	  hls::stream<b_mat>& h_t_2
	)
{
	#pragma HLS DATAFLOW	
	// we will first load the input activation in 2  different memory blocks
	// and hidden states in 2 different memory blocks as well
	int_mem  int_act_1;  //internal activation for first input block
	int_mem int_act_2;  //internal activation for second input block

	int_mem  int_hid_1; //internal hidden state  for first h(t-1) block
	int_mem int_hid_2;  //internal hidden state  for second h(t-1) block

	loadA_1:	for(int i = 0; i<tot_B_num; i++)
			{
				#pragma HLS unroll factor = 2
				blc_vec tempA1 = x_t1.read();
				for(int j = 0; j < mult_num_block; j++)
				{
					#pragma HLS PIPELINE II = 1
					int_act_1.in[j][i] = tempA1.inputs[j];
				}
			}

	loadA_2:	for(int i = 0; i<tot_B_num; i++)
			{
				#pragma HLS unroll factor = 2
				blc_vec tempA2 = x_t2.read();
				for(int j = 0; j < mult_num_block; j++)
				{
					#pragma HLS PIPELINE II = 1
					int_act_2.in[j][i] = tempA2.inputs[j];
				}
			}

	loadH_1:	for(int i = 0; i<tot_B_num; i++)
			{
				#pragma HLS unroll factor = 2
				blc_vec tempH1 = h_t1_1.read();
				for(int j = 0; j < mult_num_block; j++)
				{
					#pragma HLS PIPELINE II = 1
					int_hid_1.in[j][i] = tempH1.inputs[j];
				}
			}

	loadH_2:	for(int i = 0; i<tot_B_num; i++)
			{
				#pragma HLS unroll factor = 2
				blc_vec tempH2 = h_t1_2.read();
				for(int j = 0; j < mult_num_block; j++)
				{
					#pragma HLS PIPELINE II = 1
					int_hid_2.in[j][i] = tempH2.inputs[j];
				}
			}

	// we will then instantiate 4 block_lstms
	
	
	block_lstm_compute:
			block_lstm(int_act_1, int_hid_1, w_xo_1, w_ho_1, w_xi_1, w_hi_1, w_xc_1, w_hc_1, h_f_1, x_f_1, h_t_1);
			block_lstm(int_act_1, int_hid_1, w_xo_2, w_ho_2, w_xi_2, w_hi_2, w_xc_2, w_hc_2, h_f_2, x_f_2, h_t_2);
			block_lstm(int_act_2, int_hid_2, w_xo_1, w_ho_1, w_xi_1, w_hi_1, w_xc_1, w_hc_1, h_f_1, x_f_1, h_t_1);
			block_lstm(int_act_2, int_hid_2, w_xo_2, w_ho_2, w_xi_2, w_hi_2, w_xc_2, w_hc_2, h_f_2, x_f_2, h_t_2);
}
