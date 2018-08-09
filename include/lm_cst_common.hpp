#pragma once


#include <future>
#include <iomanip>

#include <boost/progress.hpp>

#include "dynet/dict.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/nodes.h"
#include "dynet/rnn.h"
#include "dynet/timing.h"
#include "dynet/training.h"

#include "constants.hpp"
#include "cst.hpp"
#include "data_loader.hpp"
#include "logging.hpp"

struct prefix_batch_t {
    bool keep_dist = false;
    size_t prefix_len;
    size_t num_predictions;
    size_t size;
    std::vector<std::vector<uint32_t>> prefix;
    std::vector<cst_node_type> cst_nodes;
    std::vector<float> dist;

    bool operator<(const prefix_batch_t& other) const {
        return prefix_len < other.prefix_len;
    }

    void print(const corpus_t& c) {
        std::cout << "PREFIX_BATCH ===================================" << std::endl;
        for(size_t j=0;j<prefix[0].size();j++) {
            for(size_t i=0;i<prefix.size();i++) {
                std::cout << c.vocab.inverse_lookup(prefix[i][j]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "================================================" << std::endl;
    }
};

struct one_hot_batch_t {
    size_t size;
    std::vector<std::vector<uint32_t>> prefix;
    std::vector<std::vector<uint32_t>> suffix;
};

using prefix_batches_t = std::vector<prefix_batch_t>;
using one_hot_batches_t = std::vector<one_hot_batch_t>;

prefix_batches_t
create_prefix_batches(std::vector<prefix_t>& all_prefixes,const corpus_t& corpus,size_t batch_size)
{
    cstnn_timer timer("create_prefix_batches");
    prefix_batches_t prefix_batches;
    std::sort(all_prefixes.begin(),all_prefixes.end());
    auto prefix_itr = all_prefixes.begin();
    auto prefix_end= all_prefixes.end();
    size_t left = all_prefixes.size();
    while(prefix_itr != prefix_end) {
        auto batch_start = prefix_itr;
        auto batch_end = batch_start;
        if(left < batch_size) {
            batch_end = batch_start + left;
        } else {
            batch_end = batch_start + batch_size;
        }

        auto tmp = batch_end - 1;
        while( tmp->prefix.size() != batch_start->prefix.size()) {
            --tmp;
        }
        batch_end = tmp + 1;

        prefix_batch_t pb;
        pb.prefix_len = batch_start->prefix.size();
        pb.size = std::distance(batch_start,batch_end);
        pb.prefix.resize(pb.prefix_len);
        for(size_t i=0;i<pb.prefix_len;i++) {
            for(size_t j=0;j<pb.size;j++) {
                auto& cp = (batch_start + j)->prefix;
                pb.prefix[i].push_back(cp[i]);
            }
        }
        for(size_t j=0;j<pb.size;j++) {
            auto node = (batch_start + j)->node;
            pb.cst_nodes.push_back(node);
        }
        prefix_batches.emplace_back(std::move(pb));
        prefix_itr += pb.size;
        left -= pb.size;
    }
    CNLOG << "prefix batches = " << prefix_batches.size();
    std::vector<uint32_t> batch_size_dist(batch_size+1);
    for(auto& b : prefix_batches) batch_size_dist[b.size]++;
    for(size_t i=0;i<batch_size_dist.size();i++) {
        if(batch_size_dist[i] != 0) {
            double percent = 100.0 * double(batch_size_dist[i]) / double(prefix_batches.size());
            CNLOG << "\tbatch size " << i << " - # " << batch_size_dist[i] << " (" << percent << "%)";
        }
    }
    std::vector<uint32_t> batch_prefix_dist(20000);
    for(auto& b : prefix_batches) batch_prefix_dist[b.prefix.size()]++;
    for(size_t i=0;i<batch_prefix_dist.size();i++) {
        if(batch_prefix_dist[i] != 0) {
            double percent = 100.0 * double(batch_prefix_dist[i]) / double(prefix_batches.size());
            CNLOG << "\tprefix size " << i << " - # " << batch_prefix_dist[i] << " (" << percent << "%)";
        }
    }

    return prefix_batches;
}

one_hot_batches_t
create_sentence_batches(std::vector<sentence_t>& all_sentences,const corpus_t& corpus,size_t batch_size)
{
    cstnn_timer timer("create_sentence_batches");
    one_hot_batches_t sent_batches;
    std::sort(all_sentences.begin(),all_sentences.end());

    size_t longest_sent = 0;
    size_t id = 0;
    for(size_t i=0;i<all_sentences.size();i++) {
        size_t len = all_sentences[i].prefix.size() + all_sentences[i].suffix.size();
        if(len > longest_sent) {
            id = i;
            longest_sent = len;
        }
    }

    CNLOG << "LS LEN = " << longest_sent;
    CNLOG << "LS PREFIX = " << corpus.vocab.print_sentence(all_sentences[id].prefix);
    CNLOG << "LS SUFFIX = " << corpus.vocab.print_sentence(all_sentences[id].suffix);

    auto s_itr = all_sentences.begin();
    auto s_end = all_sentences.end();
    size_t left = all_sentences.size();
    while(s_itr != s_end) {
        auto batch_start = s_itr;
        auto batch_end = s_end;
        if(left < batch_size) {
            batch_end = batch_start + left;
        } else {
            batch_end = batch_start + batch_size;
        }

        auto tmp = batch_end - 1;
        while( tmp->prefix.size() != batch_start->prefix.size()) {
            --tmp;
        }
        while( (tmp->suffix.size() - batch_start->suffix.size()) > 5) {
            --tmp;
        }
        batch_end = tmp + 1;
        auto batch_last = tmp;
        auto last_suffix_size = batch_last->suffix.size();
        tmp = batch_start;
        while( tmp != batch_end ) {
            auto before = tmp->suffix.size();
            while( tmp->suffix.size() != last_suffix_size) {
                tmp->suffix.push_back( corpus.vocab.eof_tok );
            }
            auto after = tmp->suffix.size();
            ++tmp;
        }
        one_hot_batch_t sb;

        sb.size = std::distance(batch_start,batch_end);

        sb.prefix.resize(batch_start->prefix.size());
        sb.suffix.resize(batch_start->suffix.size());
        for(size_t i=0;i<sb.prefix.size();i++) {
            for(size_t j=0;j<sb.size;j++) {
                auto& cp = (batch_start + j)->prefix;
                sb.prefix[i].push_back(cp[i]);
            }
        }
        for(size_t i=0;i<sb.suffix.size();i++) {
            for(size_t j=0;j<sb.size;j++) {
                auto& cs = (batch_start + j)->suffix;
                sb.suffix[i].push_back(cs[i]);
            }
        }
        sent_batches.emplace_back(std::move(sb));
        s_itr += sb.size;
        left -= sb.size;
    }
    CNLOG << "sentence batches = " << sent_batches.size();
    std::vector<uint32_t> batch_size_dist(batch_size+1);
    for(auto& b : sent_batches) batch_size_dist[b.size]++;
    for(size_t i=0;i<batch_size_dist.size();i++) {
        if(batch_size_dist[i] != 0) {
            double percent = 100.0 * double(batch_size_dist[i]) / double(sent_batches.size());
            CNLOG << "\tbatch size " << i << " - # " << batch_size_dist[i] << " (" << percent << "%)";
        }
    }
    std::vector<uint32_t> batch_suffix_dist(20000);
    for(auto& b : sent_batches) batch_suffix_dist[b.suffix.size()]++;
    for(size_t i=0;i<batch_suffix_dist.size();i++) {
        if(batch_suffix_dist[i] != 0) {
            double percent = 100.0 * double(batch_suffix_dist[i]) / double(sent_batches.size());
            CNLOG << "\tsuffix size " << i << " - # " << batch_suffix_dist[i] << " (" << percent << "%)";
        }
    }
    std::vector<uint32_t> batch_prefix_dist(20000);
    for(auto& b : sent_batches) batch_prefix_dist[b.prefix.size()]++;
    for(size_t i=0;i<batch_prefix_dist.size();i++) {
        if(batch_prefix_dist[i] != 0) {
            double percent = 100.0 * double(batch_prefix_dist[i]) / double(sent_batches.size());
            CNLOG << "\tprefix size " << i << " - # " << batch_prefix_dist[i] << " (" << percent << "%)";
        }
    }

    size_t blen = 0;
    for(size_t i=0;i<sent_batches.size();i++) {
        size_t len = sent_batches[i].prefix.size() + sent_batches[i].suffix.size();
        if(blen < len) {
            blen = len;
        }
    }
    CNLOG << "LONGEST BATCH = " << blen;

    return sent_batches;
}

std::tuple<prefix_batches_t,one_hot_batches_t>
create_train_batches(const cst_type& cst,const corpus_t& corpus, args_t& args,size_t batch_size)
{
    // mark all sentence starts in a bitvector
    sdsl::bit_vector sent_pos(cst.size());
    {
        cstnn_timer timer("mark all sentence starts in a bitvector");
        for(size_t i=0;i<corpus.num_sentences;i++) {
            auto sent_start_pos = corpus.sent_starts[i];
            auto pos_in_sa_order = cst.csa.isa[sent_start_pos];
            sent_pos[pos_in_sa_order] = 1;
        }
    }
    sdsl::rank_support_v5<> rank(&sent_pos);

    auto all_prefixes = find_all_prefixes(cst,corpus,rank);
    auto all_sentences = find_all_sentences(all_prefixes,cst,corpus,sent_pos);

    auto prefix_batches = create_prefix_batches(all_prefixes,corpus,batch_size);
    auto sent_batches = create_sentence_batches(all_sentences,corpus,batch_size);

    return std::make_tuple(prefix_batches,sent_batches);
}

std::tuple<dynet::Expression, size_t>
build_train_graph_prefix(language_model& lm,dynet::ComputationGraph& cg,prefix_batch_t& batch,double drop_out)
{
    lm.rnn.new_graph(cg);
    lm.rnn.start_new_sequence();
    if(drop_out != 0.0) {
        lm.rnn.set_dropout(0,drop_out);
    }
    lm.i_R = dynet::parameter(cg, lm.p_R);
    lm.i_bias = dynet::parameter(cg, lm.p_bias);
    for (size_t i = 0; i < batch.prefix.size()-1; ++i) {
        lm.rnn.add_input(dynet::lookup(cg, lm.p_c, batch.prefix[i]));
    }
    auto last_toks = batch.prefix.back();
    dynet::Expression i_y_t = lm.rnn.add_input(dynet::lookup(cg, lm.p_c, last_toks));
    dynet::Expression i_r_t = lm.i_bias + lm.i_R * i_y_t;

    dynet::Expression i_pred = -dynet::log_softmax(i_r_t);
    dynet::Expression i_pred_linear = dynet::reshape(i_pred, { (unsigned int)batch.dist.size() });
    dynet::Expression i_true = dynet::input(cg, { (unsigned int)batch.dist.size() }, batch.dist );
    dynet::Expression i_error = dynet::transpose(i_true) * i_pred_linear;
    return std::make_tuple(i_error,batch.num_predictions);
}

std::tuple<dynet::Expression,size_t>
build_train_graph_sents(language_model& lm,dynet::ComputationGraph& cg,one_hot_batch_t& batch,double drop_out)
{
    size_t num_predictions = 0;
    lm.rnn.new_graph(cg);
    lm.rnn.start_new_sequence();
    if(drop_out != 0.0) {
        lm.rnn.set_dropout(0,drop_out);
    }
    lm.i_R = dynet::parameter(cg, lm.p_R);
    lm.i_bias = dynet::parameter(cg, lm.p_bias);
    std::vector<Expression> errs;
    dynet::Expression i_y_t;
    for (size_t i = 0; i < batch.prefix.size(); ++i) {
        auto& cur_tok = batch.prefix[i];
        auto i_x_t = dynet::lookup(cg, lm.p_c, cur_tok);
        i_y_t = lm.rnn.add_input(i_x_t);
    }



    for (size_t i = 0; i < batch.suffix.size()-1; ++i) {
        auto i_r_t = lm.i_bias + lm.i_R * i_y_t;
        auto i_err = dynet::pickneglogsoftmax(i_r_t, batch.suffix[i]);
        num_predictions += batch.size;
        errs.push_back(i_err);
        auto i_x_t = dynet::lookup(cg, lm.p_c, batch.suffix[i]);
        i_y_t = lm.rnn.add_input(i_x_t);
    }
    auto i_r_t = lm.i_bias + lm.i_R * i_y_t;
    auto i_err = dynet::pickneglogsoftmax(i_r_t, batch.suffix.back());
    errs.push_back(i_err);
    num_predictions += batch.size;
    return std::make_pair(dynet::sum_batches(dynet::sum(errs)),num_predictions);
}

void compute_dist(prefix_batch_t& pb,const cst_type& cst,const corpus_t& corpus) {
    if(pb.dist.size() == 0) {
        pb.dist.resize(pb.size*corpus.vocab.size());
        pb.num_predictions = 0;
        for(size_t i=0;i<pb.cst_nodes.size();i++) {
            auto offset = i * corpus.vocab.size();
            auto node = pb.cst_nodes[i];
            auto node_depth = cst.depth(node);
            for (const auto& child : cst.children(node)) {
                auto tok = cst.edge(child, node_depth + 1);
                double size = cst.size(child);
                pb.num_predictions += size;
                pb.dist[offset+tok] = size;
            }
        }
    }
}
