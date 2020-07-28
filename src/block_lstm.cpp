// this file containts the built LSTM architecture
// the "Recurrent Neural Networks Hardware Implementation on FPGA" paper by Andre XIan Ming was used as reference
#include "basic_ops.h"
#include "block_lstm.h"

// we will have weights for the input, output, forget and candidate memory and all of them will comprise of two part, the input part and hidden part

void block_lstm(
		int_mem& x_t,
		int_mem& h_t1,
		hls::stream<b_vec>& w_xo, 
		hls::stream<b_vec>& w_ho, 
		hls::stream<b_vec>& w_xi, 
		hls::stream<b_vec>& w_hi, 
		hls::stream<b_vec>& w_xc, 
		hls::stream<b_vec>& w_hc, 
		hls::stream<b_vec>& h_f, 
		hls::stream<b_vec>& x_f, 
		hls::stream<b_mat>& h_t  
		)
{
    #pragma HLS DATAFLOW
    //we need some internal matrices to hold intermediate values
    b_mat* xo_mat_out, ho_mat_out, out_sum, o_t,
          xi_mat_out, hi_mat_out, in_sum, i_t,
          xc_mat_out, hc_mat_out, mem_sum, c_hat_t,
          xf_mat_out, hf_mat_out, for_sum, f_t;

    b_mat* i_d_ch, f_d_c1, o_d_c;
    b_mat* c_t, c_t1, c_tanh_out;
    b_mat* h_t_;

    c_t1 = {0}; //initialize it to avoid collecting garbage data

out_gate:
    block_maltmul(w_xo, x_t, xo_mat_out);
    block_maltmul(w_ho, h_t1, ho_mat_out );
    addition(xo_mat_out, ho_mat_out, out_sum);
    sigmoid(out_sum, o_t);

in_gate:
    block_maltmul(w_xi, x_t, xi_mat_out);
    block_maltmul(w_hi, h_t1, hi_mat_out);
    addition(xi_mat_out, hi_mat_out, in_sum);
    sigmoid(in_sum, i_t);

candidate_mem_gate:
    block_maltmul(w_xi, x_t, xi_mat_out);
    block_maltmul(w_hi, h_t1, hi_mat_out);
    addition(xi_mat_out, hi_mat_out, in_sum);
    sigmoid(in_sum, i_t);

forget_gate:
    block_maltmul(w_xf, x_t, xf_mat_out);

    block_maltmul(w_hf, h_t1, hf_mat_out);
    addition(xf_mat_out, hf_mat_out, for_sum);
    sigmoid(for_sum, f_t);

cell_gate:
    emult(i_t, c_hat_t, i_d_ch);
    emult(f_t, c_t1, f_d_c1);
    addition(i_d_ch, f_d_c1, c_t);
    transfer_data (c_t, c_t1)
    tanh_(c_t, c_tanh_out);
    emult(o_t, c_tanh_out, h_t_);

//now write the new hidden state into the stream

    h_t.write(h_t_);

}

