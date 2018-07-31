#pragma once

#include <future>
#include <iomanip>
#include <queue>

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

using cst_node_type = typename cst_type::node_type;

struct train_instance_t {
    std::vector<uint32_t> prefix;
    std::vector<uint32_t> suffix;
    size_t num_occ;
    cst_node_type parent;
    cst_node_type cur_node;
};

struct instances_t {

};

train_instance_t create_leaf_instance(const cst_type& cst, const corpus_t& corpus,cst_node_type cur)
{
    train_instance_t ti;
    ti.num_occ = 1;
    ti.parent_node = cst.parent(cur);
    ti.cur_node = cur;

    return ti;
}

template<class t_itr>
std::vector<train_instance_t>
process_token_subtrees(const cst_type& cst, const corpus_t& corpus,t_itr start,t_itr end)
{
    std::vector<train_instance_t> instances;
    auto itr = start;
    while(itr != end) {
        auto cst_node = *itr;

        auto cst_itr = cst.begin(cst_node);
        auto cst_end = cst.end(cst_node);
        while(cst_itr != cst_end) {
            auto cur_node = *cst_itr;
            if(cst.is_leaf(cur_node)) {
                instances.push_back(create_leaf_instance(cst,corpus,cur_node));
            } else {
                instances.push_back(create_internal_instance(cst,corpus,cur_node));
            }
            ++cst_itr;
        }

        ++itr;
    }
    return instances;
}

instances_t create_all_instances(const cst_type& cst, const corpus_t& corpus, args_t& args)
{
    auto threads = args["threads"].as<size_t>();
    std::mt19937 rng(constants::RAND_SEED);

    instances_t instances;
    auto lb = cst.csa.C[corpus.vocab.start_sent_tok];
    auto rb = cst.csa.C[corpus.vocab.start_sent_tok + 1] - 1;
    auto start_node = cst.node(lb,rb); // cst node of <s>
    
    // all the subtrees of <s> that we need to explore
    std::vector<cst_node_type> subtrees;
    for (const auto& child : cst.children(cst_node)) {
        subtrees.push_back(child);
    }
    std::shuffle(subtrees.begin(),subtrees.end(), rng);
    
    std::vector<std::future<std::vector<train_instance_t> > > results;
    auto itr = subtrees.begin();
    auto trees_per_thread = subtrees.size() / threads;
    for (size_t thread = 0; thread < threads; thread++) {
        auto start = itr;
        auto end = itr + trees_per_thread;
        if(thread == threads-1) {
            end = subtrees.end();
        }
        itr += trees_per_thread;
        results.push_back(std::async(std::launch::async,process_token_subtrees, cst, corpus, start, end));
    }

    // merge and insert
    instances.add(create_internal_instance(cst,corpus,start_node));
    for (auto& e : results) {
        auto res = e.get();
        for(auto& instance : res) {
            instances.add(instance);
        }
    }
    return instances;
}

language_model create_lm(const cst_type& cst, const corpus_t& corpus, args_t& args)
{
    auto dev_corpus_file = args["path"].as<std::string>() + "/" + constants::DEV_FILE;
    std::mt19937 rng(constants::RAND_SEED);

    auto num_epochs = args["epochs"].as<size_t>();
    auto batch_size = args["batch_size"].as<size_t>();
    auto epoch_size = args["epoch_size"].as<size_t>();

    language_model lm(corpus.vocab, args);

    auto instances = create_all_instances(corpus,cst);

    for (size_t epoch = 1; epoch <= num_epochs; epoch++) {
        size_t last_report = 0;
        for(size_t i=0;i<epoch_size;i++) {
            auto sampled_instance = instances.sample();

            auto train_start = std::chrono::high_resolution_clock::now();
            auto loss = lm.build_train_graph(cg,sampled_instance);
            auto loss_expr = std::get<0>(loss);
            double num_pred = std::get<1>(loss);
            auto loss_float = dynet::as_scalar(cg.forward(loss_expr));
            auto instance_loss = loss_float / num_pred;
            auto train_end = std::chrono::high_resolution_clock::now();

            if ( (i - last_report) > 8192 || i == instances.size()) {
                std::chrono::duration<double> train_diff = train_end - train_start;
                auto time_per_instance = train_diff.count() / actual_batch_size * 1000.0;
                double percent = double(cur) / double(instances.size()) * 100;
                last_report = i;
                CNLOG << std::fixed << std::setprecision(1) << std::floor(percent) << "% "
                      << cur << "/" << instances.size()
                      << " batch_size = " << num_pred
                      << " FW/BW/UPDATE  "
                      << time_per_instance << "ms/instance - loss = " << instance_loss;
            }
        }
        CNLOG << "finish epoch " << epoch << ". compute dev pplx ";
        auto pplx = evaluate_pplx(lm, corpus, dev_corpus_file);
        CNLOG << "epoch dev pplx = " << pplx;
    }

    return lm;
}
