#ifndef block_lstm_H_
#define block_lstm_H_
#include "basic_ops.h"

//block_lstm 

void block_lstm(
		int_mem&  x_t,
		int_mem& h_t1,
		hls::stream<b_vec>& w_xo,
		hls::stream<b_vec>& w_ho,
		hls::stream<b_vec>& w_xi,
		hls::stream<b_vec>& w_hi,
		hls::stream<b_vec>& w_xc
		hls::stream<b_vec>& w_hc,
		hls::stream<b_vec>& h_f,
		hls::stream<b_vec>& x_f,
		hls::stream<b_mat>% h_t
		);

#endif
