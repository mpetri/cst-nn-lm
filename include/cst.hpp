#pragma once

#include <chrono>
#include <iostream>

#include "constants.hpp"
#include "data_loader.hpp"
#include "logging.hpp"

#include <boost/filesystem.hpp>

#include "sdsl/suffix_trees.hpp"
#include "sdsl/wavelet_trees.hpp"

using cst_type = sdsl::cst_sct3<sdsl::csa_wt_int<sdsl::wt_huff_int<>, 2, 2> >;

namespace constants {
std::string CST_FILE = "cst.sdsl";
}

cst_type build_or_load_cst(corpus_t& corpus, args_t& args)
{
    auto cst_file = args["path"].as<std::string>() + "/" + constants::CST_FILE;
    cst_type cst;
    if (boost::filesystem::exists(cst_file)) {
        sdsl::load_from_file(cst, cst_file);
    } else {
        sdsl::construct_im(cst, corpus.text, 0);
        sdsl::store_to_file(cst, cst_file);
    }
    return cst;
}
