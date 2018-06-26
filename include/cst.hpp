#pragma once

#include <iostream>
#include <chrono>

#include "logging.hpp"
#include "constants.hpp"
#include "data_loader.hpp"

#include "sdsl/suffix_trees.hpp"

using cst_type = sdsl::cst_sct3<sdsl::csa_wt_int<>>;

cst_type build_cst(corpus_t& corpus,args_t&)
{
	cst_type cst;
	sdsl::construct_im(cst,corpus.text,0);
	return cst;
}