#pragma once

#include <chrono>
#include <iostream>

#include "constants.hpp"
#include "data_loader.hpp"
#include "logging.hpp"
#include "util.hpp"

#include <boost/filesystem.hpp>

#include "sdsl/suffix_trees.hpp"
#include "sdsl/wavelet_trees.hpp"

using cst_type = sdsl::cst_sct3<sdsl::csa_wt_int<sdsl::wt_huff_int<>, 2, 2> >;
using cst_node_type = typename cst_type::node_type;

namespace constants {
std::string CST_FILE = "cst.sdsl";
}

cst_type build_or_load_cst(const corpus_t& corpus, args_t& args)
{
    cstnn_timer timer("build_or_load_cst");
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

struct prefix_t {
    cst_node_type node;
    std::vector<uint32_t> prefix;
    std::vector<float> dist;
    bool operator<(const prefix_t& other) const {
        return prefix.size() < other.prefix.size();
    }

    size_t size_in_bytes() const {
        return prefix.size()*sizeof(uint32_t) + dist.size()*sizeof(float)+sizeof(cst_node_type);
    }
};

struct sentence_t {
    cst_node_type node;
    std::vector<uint32_t> prefix;
    std::vector<uint32_t> suffix;
    bool operator<(const sentence_t& other) const {
        if(prefix.size() == other.prefix.size()) {
            return suffix.size() < other.suffix.size();
        }
        return prefix.size() < other.prefix.size();
    }
};

std::vector<uint32_t>
edge_label(const cst_type& cst,cst_node_type node)
{
    std::vector<uint32_t> label;
    auto node_depth = cst.depth(node);
    for(size_t i=0;i<node_depth;i++) {
        label.push_back(cst.edge(node,i+1));
    }
    return label;
}

void
add_prefix(std::vector<prefix_t>& prefixes,const cst_type& cst,cst_node_type node,const corpus_t& corpus)
{
    if(!cst.is_leaf(node)) {
        prefix_t p;
        p.node = node;
        p.prefix = edge_label(cst,node);
        p.dist.resize(corpus.vocab.size());
        auto node_depth = cst.depth(node);
        for (const auto& child : cst.children(node)) {
            auto tok = cst.edge(child, node_depth + 1);
            double size = cst.size(child);
            p.dist[tok] = size;
        }
        prefixes.push_back(p);
    }
}

std::vector<prefix_t>
find_all_prefixes(const cst_type& cst,const corpus_t& corpus)
{
    cstnn_timer timer("find_all_prefixes");
    std::vector<prefix_t> prefixes;
    auto lb = cst.csa.C[corpus.vocab.start_sent_tok];
    auto rb = cst.csa.C[corpus.vocab.start_sent_tok + 1] - 1;
    auto start_node = cst.node(lb,rb); // cst node of <s>

    auto cst_itr = cst.begin(start_node);
    auto cst_end = cst.end(start_node);

    while(cst_itr != cst_end) {
        auto cur_node = *cst_itr;
        if(cst_itr.visit() == 2) {
            add_prefix(prefixes,cst,cur_node,corpus);
        }
        ++cst_itr;
    }
    CNLOG << "found prefixes = " << prefixes.size();
    size_t space_bytes = 0;
    for(auto& p : prefixes) {
        space_bytes += p.size_in_bytes();
    }
    CNLOG << "prefixes storage space = " << space_bytes / (1024*1024) << "MiB";
    return prefixes;
}

void
add_sentence(std::vector<sentence_t>& sentences,prefix_t& p,size_t tok,const cst_type& cst,const corpus_t& corpus)
{
    if(tok != corpus.vocab.stop_sent_tok) {
        size_t char_pos;
        auto leaf_node = cst.child(p.node,tok,char_pos);
        sentence_t s;
        s.prefix = p.prefix;
        s.node = leaf_node;
        size_t depth = s.prefix.size() + 2;
        while(tok != corpus.vocab.stop_sent_tok) {
            s.suffix.push_back(tok);
            tok = cst.edge(leaf_node,depth);
            depth++;
        }
        s.suffix.push_back(tok);
        sentences.push_back(s);
    }
}

std::vector<sentence_t>
find_all_sentences(std::vector<prefix_t>& all_prefixes,const cst_type& cst,const corpus_t& corpus)
{
    cstnn_timer timer("find_all_sentences");
    std::vector<sentence_t> sentences;

    for(auto& prefix : all_prefixes) {
        for(size_t i = 0;i<prefix.dist.size();i++) {
            if(prefix.dist[i] == 1) {
                // leaf child
                add_sentence(sentences,prefix,i,cst,corpus);
            }
        }
    }

    return sentences;
}