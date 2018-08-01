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
    std::vector<float> dist;
    size_t num_occ;
    cst_node_type parent;
    cst_node_type cur_node;
};

struct instance_batch_t {
    size_t batch_size;
    std::vector<std::vector<uint32_t>> toks;
    std::vector<float> dist;
};

struct instance_storage {
    size_t total_num_instances = 0;
    std::vector<size_t> instance_len_dist;
    std::vector<std::vector<train_instance_t>> instances;
    std::vector<size_t> cur_len_offsets;
    std::discrete_distribution<> len_dist;
    std::mt19937 rng;

    instance_storage() {
        instance_len_dist.resize(constants::MAX_SENTENCE_LEN+1);
        cur_len_offsets.resize(constants::MAX_SENTENCE_LEN+1);
        instances.resize(constants::MAX_SENTENCE_LEN+1);
        rng.seed(constants::RAND_SEED);
    }

    void add(train_instance_t& ti) {
        auto inst_len = ti.prefix.size();
        instance_len_dist[inst_len] += ti.num_occ;
        instances[inst_len].push_back(ti);
        total_num_instances++;
    }

    size_t size() const {
        return total_num_instances;
    }

    void freeze() {
        len_dist = std::discrete_distribution<>(instance_len_dist.begin(),instance_len_dist.end());
        std::vector<double> p = len_dist.probabilities();
        std::cout << "len_dist probs = "
        for(size_t i = 0;i<p.size();i++) {
            if(p[i] != 0) {
                std::cout << "<" << i << ":" << p[i] << '>';
            }
        }
        std::cout << '\n';
        shuffle();
    }

    void shuffle() {
        for(size_t i=0;i<instances.size();i++) {
            std::shuffle(instances[i].begin(),instances[i].end(), rng);
            cur_len_offsets[i] = 0;
        }
    }

    instance_batch_t sample_batch(size_t batch_size) {
        instance_batch_t ib;

        // sample a len for this batch
        auto batch_sent_len = len_dist(rng);
        while( cur_len_offsets[batch_sent_len] != instances[batch_sent_len].size() ) {
            batch_len = len_dist(rng);
        }

        // sample batch samples
        size_t left = instances[batch_sent_len].size() - cur_len_offsets[batch_sent_len];
        size_t actual_batch_size = batch_size;
        if(left < batch_size) {
            actual_batch_size = left;
        }

        bool all_one_hot = true;
        ib.toks.resize(batch_sent_len);
        for(size_t j=0;j<batch_sent_len;j++) {
            for(size_t i=0;i<actual_batch_size;i++) {
                auto& cur_instance = instances[batch_sent_len][ cur_len_offsets[batch_sent_len] + i];
                ib.toks[j].push_back(cur_instance.prefix[j]);
                if(cur_instance.num_occ != 1) {
                    all_one_hot = false;
                }
            }
        }

        // one hot batch or not?
        if(all_one_hot) {
            
        } else {

        }

        cur_len_offsets[batch_sent_len] += actual_batch_size;

        return ib;
    }
};

train_instance_t create_leaf_instance(const cst_type& cst, const corpus_t& corpus,cst_node_type cur)
{
    train_instance_t ti;
    ti.num_occ = 1;
    ti.parent_node = cst.parent(cur);
    ti.cur_node = cur;

    // determine prefix edge
    auto parent_depth = cst.depth(ti.parent_node);    
    size_t cur_depth = 0;
    while(cur_depth != parent_depth) {
        auto tok = cst.edge(ti.cur_node, cur_depth + 1);
        ti.prefix.push_back(tok);
        ++cur_depth;
    }
    
    // determine the one hot stuff
    auto tok = cst.edge(ti.cur_node, cur_depth + 1);
    while(tok != corpus.vocab.stop_sent_tok) {
        ti.suffix.push_back(tok);
        tok = cst.edge(ti.cur_node, cur_depth + 1);
    }
    ti.suffix.push_back(tok);

    return ti;
}

train_instance_t create_internal_instance(const cst_type& cst, const corpus_t& corpus,cst_node_type cur)
{
    train_instance_t ti;
    ti.num_occ = cst.size(cur);
    ti.parent_node = cst.parent(cur);
    ti.cur_node = cur;

    // determine prefix edge
    auto parent_depth = cst.depth(ti.parent_node);    
    size_t cur_depth = 0;
    while(cur_depth != parent_depth) {
        auto tok = cst.edge(ti.cur_node, cur_depth + 1);
        ti.prefix.push_back(tok);
        ++cur_depth;
    }

    // determine the dist of what follows
    ti.dist = std::vector<float>(corpus.vocab.size(),0);
    for (const auto& child : cst.children(ti.cur_node)) {
        auto tok = cst.edge(child, cur_depth + 1);
        double size = cst.size(child);
        ti.dist[tok] = size;
    }

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

instance_storage create_all_instances(const cst_type& cst, const corpus_t& corpus, args_t& args)
{
    auto threads = args["threads"].as<size_t>();
    std::mt19937 rng(constants::RAND_SEED);

    instance_storage instances;
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

    // create the distributions which we sample
    instances.freeze();
    return instances;
}

dynet::Expression
build_train_graph_cst_sample(language_model& lm,dynet::ComputationGraph& cg,batch_instance_t& instance,double drop_out)
{

}

void train_cst_sample_lm(language_model& lm,const corpus_t& corpus, args_t& args)
{
    auto dev_corpus_file = args["path"].as<std::string>() + "/" + constants::DEV_FILE;
    std::mt19937 rng(constants::RAND_SEED);

    CNLOG << "build or load CST";
    auto cst = build_or_load_cst(corpus, args);

    auto num_epochs = args["epochs"].as<size_t>();
    auto batch_size = args["batch_size"].as<size_t>();
    auto epoch_size = args["epoch_size"].as<size_t>();
    auto drop_out = args["drop_out"].as<double>();

    CNLOG << "start training dynet lm";
    CNLOG << "\tepochs = " << num_epochs;
    CNLOG << "\tepoch_size = " << num_epochs;
    CNLOG << "\tbatch_size = " << batch_size;
    CNLOG << "\tdrop_out = " << drop_out;

    auto instances = create_all_instances(corpus,cst);

    for (size_t epoch = 1; epoch <= num_epochs; epoch++) {
        size_t last_report = 0;
        for(size_t i=0;i<epoch_size;i++) {
            auto sampled_batch = instances.sample_batch(batch_size);

            auto train_start = std::chrono::high_resolution_clock::now();
            auto loss = build_train_graph(lm,cg,sampled_batch);
            auto loss_expr = std::get<0>(loss);
            double num_pred = std::get<1>(loss);
            auto loss_float = dynet::as_scalar(cg.forward(loss_expr));
            auto instance_loss = loss_float / num_pred;
            auto train_end = std::chrono::high_resolution_clock::now();

            if ( (i - last_report) > 8192 || i == instances.size()) {
                std::chrono::duration<double> train_diff = train_end - train_start;
                auto time_per_instance = train_diff.count() / actual_batch_size * 1000.0;
                double percent = double(cur) / double(epoch_size) * 100;
                last_report = i;
                CNLOG << std::fixed << std::setprecision(1) << std::floor(percent) << "% "
                      << cur << "/" << epoch_size
                      << " batch_size = " << num_pred
                      << " FW/BW/UPDATE  "
                      << time_per_instance << "ms/instance - loss = " << instance_loss;
            }
        }
        CNLOG << "finish epoch " << epoch << ". compute dev pplx ";
        auto pplx = evaluate_pplx(lm, corpus, dev_corpus_file);
        CNLOG << "epoch dev pplx = " << pplx;

        instances.shuffle();
    }

    return lm;
}
