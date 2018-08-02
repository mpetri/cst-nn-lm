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
    size_t size;
    std::vector<std::vector<uint32_t>> prefix;
    std::vector<cst_node_type> cst_nodes;
    std::vector<float> dist;
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
    return prefix_batches;
}

one_hot_batches_t
create_sentence_batches(std::vector<sentence_t>& all_sentences,const corpus_t& corpus,size_t batch_size)
{
    cstnn_timer timer("create_sentence_batches");
    one_hot_batches_t sent_batches;
    std::sort(all_sentences.begin(),all_sentences.end());
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
        while( tmp->suffix.size() != batch_start->suffix.size()) {
            --tmp;
        }
        batch_end = tmp + 1;

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

    return sent_batches;
}

std::tuple<prefix_batches_t,one_hot_batches_t>
create_train_batches(const cst_type& cst,const corpus_t& corpus, args_t& args,size_t batch_size)
{
    auto all_prefixes = find_all_prefixes(cst,corpus);
    auto all_sentences = find_all_sentences(all_prefixes,cst,corpus);

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
        lm.rnn.set_dropout(drop_out);
    }
    lm.i_R = dynet::parameter(cg, lm.p_R);
    lm.i_bias = dynet::parameter(cg, lm.p_bias);
    for (size_t i = 0; i < batch.prefix.size()-1; ++i) {
        dynet::Expression i_x_t = dynet::lookup(cg, lm.p_c, batch.prefix[i]);
        lm.rnn.add_input(i_x_t);
    }
    auto last_toks = batch.prefix.back();
    dynet::Expression i_x_t = dynet::lookup(cg, lm.p_c, last_toks);
    dynet::Expression i_y_t = lm.rnn.add_input(i_x_t);
    dynet::Expression i_r_t = lm.i_bias + lm.i_R * i_y_t;

    dynet::Expression i_pred = -dynet::log_softmax(i_r_t);
    dynet::Expression i_pred_linear = dynet::reshape(i_pred, { (unsigned int)batch.dist.size() });
    dynet::Expression i_true = dynet::input(cg, { (unsigned int)batch.dist.size() }, batch.dist );
    dynet::Expression i_error = dynet::transpose(i_true) * i_pred_linear;
    return std::make_tuple(i_error,batch.size);
}

std::tuple<dynet::Expression,size_t>
build_train_graph_sents(language_model& lm,dynet::ComputationGraph& cg,one_hot_batch_t& batch,double drop_out)
{
    size_t num_predictions = 0;
    lm.rnn.new_graph(cg);
    lm.rnn.start_new_sequence();
    if(drop_out != 0.0) {
        lm.rnn.set_dropout(drop_out);
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
        size_t num_children = 0;
        for(size_t i=0;i<pb.cst_nodes.size();i++) {
            auto offset = i * corpus.vocab.size();
            auto node = pb.cst_nodes[i];
            auto node_depth = cst.depth(node);
            for (const auto& child : cst.children(node)) {
                auto tok = cst.edge(child, node_depth + 1);
                double size = cst.size(child);
                pb.dist[offset+tok] = size;
                num_children++;
            }
        }

        if(num_children > 100) {
            pb.keep_dist = true;
        }
    }
}

void train_cst_sent(language_model& lm,const corpus_t& corpus, args_t& args)
{
    CNLOG << "build or load CST";
    auto cst = build_or_load_cst(corpus, args);

    auto num_epochs = args["epochs"].as<size_t>();
    auto batch_size = args["batch_size"].as<size_t>();
    auto drop_out = args["drop_out"].as<double>();
    int64_t report_interval = args["report_interval"].as<size_t>();

    CNLOG << "start training cst sentence lm";
    CNLOG << "\tepochs = " << num_epochs;
    CNLOG << "\tbatch_size = " << batch_size;
    CNLOG << "\tdrop_out = " << drop_out;
    auto dev_corpus_file = args["path"].as<std::string>() + "/" + constants::DEV_FILE;

    // (1) create the batches
    CNLOG << "create the batches. batch_size = " << batch_size;
    std::vector<prefix_batch_t> prefix_batches;
    std::vector<one_hot_batch_t> one_hot_batches;
    auto prep_start = std::chrono::high_resolution_clock::now();
    std::tie(prefix_batches,one_hot_batches) = create_train_batches(cst,corpus,args,batch_size);
    auto prep_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prep_diff = prep_end - prep_start;
    CNLOG << "created batches in " << " - " << prep_diff.count() << "s";

    dynet::AdamTrainer trainer(lm.model, 0.001, 0.9, 0.999, 1e-8);
    std::mt19937 rng(constants::RAND_SEED);
    std::vector<uint32_t> batch_ids(prefix_batches.size()+one_hot_batches.size());
    for(size_t i=0;i<batch_ids.size();i++) batch_ids[i] = i;

    for (size_t epoch = 1; epoch <= num_epochs; epoch++) {
        CNLOG << "start epoch " << epoch << "/" << num_epochs;
        std::shuffle(batch_ids.begin(),batch_ids.end(), rng);

        size_t last_report = 0;
        for(size_t i=0;i<batch_ids.size();i++) {
            auto train_start = std::chrono::high_resolution_clock::now();
            auto cur_batch_id = batch_ids[i];

            dynet::Expression loss;
            size_t num_predictions;
            dynet::ComputationGraph cg;
            float loss_float;
            if(cur_batch_id >= prefix_batches.size()) {
                auto& cur_batch = one_hot_batches[cur_batch_id-prefix_batches.size()];
                compute_dist(cur_batch,cst,corpus);

                trainer.clip_threshold = trainer.clip_threshold * cur_batch.size;
                std::tie(loss,num_predictions) = build_train_graph_sents(lm,cg,cur_batch,drop_out);
                loss_float = dynet::as_scalar(cg.forward(loss));
                cg.backward(loss);
                trainer.update();

                if(!cur_batch.keep_dist) {
                    cur_batch.dist.clear();
                }
            } else {
                auto& cur_batch = prefix_batches[cur_batch_id];
                trainer.clip_threshold = trainer.clip_threshold * cur_batch.size;
                std::tie(loss,num_predictions) = build_train_graph_prefix(lm,cg,cur_batch,drop_out);

                loss_float = dynet::as_scalar(cg.forward(loss));
                cg.backward(loss);
                trainer.update();
            }

            auto instance_loss = loss_float / num_predictions;
            auto train_stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> train_diff = train_stop - train_start;
            auto time_per_instance = train_diff.count() / num_predictions * 1000.0;

            if ( int64_t(i-last_report) > report_interval || i+1 == batch_ids.size()) {
                double percent = double(i) / double(batch_ids.size()) * 100;
                last_report = i;
                CNLOG << std::fixed << std::setprecision(1) << std::floor(percent) << "% "
                      << (i+1) << "/" << batch_ids.size()
                      << " batch_size = " << num_predictions
                      << " TIME = "<< time_per_instance << "ms/instance"
                      << " ppl = " << exp(instance_loss);
            }
        }

        CNLOG << "finish epoch " << epoch << ". compute dev pplx ";
        auto pplx = evaluate_pplx(lm, corpus, dev_corpus_file);
        CNLOG << "epoch " << epoch << " dev pplx = " << pplx;
    }

}
