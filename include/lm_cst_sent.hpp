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

using cst_node_type = typename cst_type::node_type;

struct instance_t {
    bool all_one_hot;
    std::vector<uint32_t> sentence;
    size_t padding = 0;
    size_t real_len = 0;
    size_t rand = 0;
    bool operator<(const instance_t& other) const
    {
        if (real_len == other.real_len) {
            return rand < other.rand;
        }
        return real_len < other.real_len;
    }
    template <class t_itr>
    instance_t(t_itr& itr, size_t len)
    {
        for (size_t i = 0; i < len; i++) {
            sentence.push_back(*itr++);
        }
        real_len = len;
    }
};

template <class t_itr>
std::tuple<dynet::Expression, size_t> build_train_graph_cst_sent(
    language_model& lm,
    dynet::ComputationGraph& cg, t_itr& start, t_itr& end,
    std::vector<std::vector<float>>& dists,const corpus_t& corpus,
    double drop_out)
{
    size_t batch_size = std::distance(start, end);
    size_t sentence_len = start->sentence.size();

    lm.rnn.new_graph(cg);
    lm.rnn.start_new_sequence();
    if(drop_out != 0.0) {
        lm.rnn.set_dropout(drop_out);
    }

    i_R = dynet::parameter(cg, lm.p_R);
    i_bias = dynet::parameter(cg, lm.p_bias);
    // Initialize variables for batch errors
    std::vector<dynet::Expression> errs;

    // Set all inputs to the SOS symbol
    auto sos_tok = start->sentence.front();
    std::vector<uint32_t> current_tok(batch_size, sos_tok);
    std::vector<uint32_t> next_tok(batch_size);

    // Run rnn on batch
    size_t actual_predictions = 0;
    for (size_t j = 0; j < batch_size; j++) {
        auto instance = start + j;
        actual_predictions += (instance->real_len - 1);
    }

    for (size_t i = 0; i < sentence_len - 1; ++i) {
        for (size_t j = 0; j < batch_size; j++) {
            auto instance = start + j;
            next_tok[j] = instance->sentence[i+1];
        }

        // Embed the current tokens
        auto i_x_t = dynet::lookup(cg, lm.p_c, current_tok);
        // Run one step of the rnn : y_t = RNN(x_t)
        auto i_y_t = lm.rnn.add_input(i_x_t);
        // Project to the token space using an affine transform
        auto i_r_t = i_bias + i_R * i_y_t;

        // Compute error for each member of the batch
        dynet::Expression i_pred = -dynet::log_softmax(i_r_t);
        dynet::Expression i_pred_linear = dynet::reshape(i_pred, { (unsigned int) dists[i].size() });
        dynet::Expression i_true = dynet::input(cg, { (unsigned int)dists[i].size() }, dists[i]);
        dynet::Expression i_error = dynet::transpose(i_true) * i_pred_linear;
        errs.push_back(i_error);

        // Change input
        current_tok = next_tok;
    }
    // Add all errors
    return std::make_tuple(dynet::sum(errs), actual_predictions);
}

template<class t_itr>
std::vector<std::vector<float>> compute_batch_losses(const cst_type& cst,const corpus_t& corpus,t_itr itr,t_itr end) {
    size_t batch_size = std::distance(itr, end);
    size_t sentence_len = itr->sentence.size();

    static std::unordered_map<uint64_t,std::vector<float>> loss_cache;
    
    std::vector<std::vector<float>> losses(sentence_len);
    for(size_t i=0;i<losses.size();i++) losses[i].resize(corpus.vocab.size()*batch_size);

    for(size_t k=0;k<batch_size;k++) {
        auto instance = *itr;
        auto cur_node = cst.root();
        for(size_t i=0;i<instance.sentence.size()-1;i++) {
            auto& tok = instance.sentence[i];
            // toks[i][k] = tok;
            auto instance_loss_itr = losses[i].begin() + (corpus.vocab.size() * k);
            size_t char_pos;
            cur_node = cst.child(cur_node,tok,char_pos);
            if(cst.is_leaf(cur_node)) {
                // everything else is one hot
                *(instance_loss_itr + instance.sentence[i+1]) = 1;
                for(size_t j=i+1;j<instance.sentence.size()-1;j++) {
                    // toks[j][k] = instance.sentence[j];
                    instance_loss_itr = losses[j].begin() + (corpus.vocab.size() * k);
                    *(instance_loss_itr + instance.sentence[j+1]) = 1;
                }
                break;
            } else {
                double node_size = cst.size(cur_node);
                if(node_size > 25) {
                    auto node_id = cst.id(cur_node);
                    auto itr = loss_cache.find(node_id);
                    if(itr != loss_cache.end()) {
                        auto& stored_loss = itr->second;
                        for(size_t i=0;i<stored_loss.size();i++) *(instance_loss_itr + i) = stored_loss[i];
                    } else {
                        std::vector<float> stored_loss(corpus.vocab.size(),0.0f);
                        auto node_depth = cst.depth(cur_node);
                        for (const auto& child : cst.children(cur_node)) {
                            auto tok = cst.edge(child, node_depth + 1);
                            double size = cst.size(child);
                            *(instance_loss_itr + tok) = size / node_size;
                            stored_loss[tok] = size / node_size;
                        }
                        loss_cache[node_id] = stored_loss;
                    }
                } else {
                    auto node_depth = cst.depth(cur_node);
                    std::vector<uint32_t> prefix(instance.sentence.begin(),instance.sentence.begin()+i+1);
                    double node_size = cst.size(cur_node);
                    for (const auto& child : cst.children(cur_node)) {
                        auto tok = cst.edge(child, node_depth + 1);
                        double size = cst.size(child);
                        *(instance_loss_itr + tok) = size / node_size;
                    }
                }
            }
        }
        ++itr;
    }

    return losses;
}


void train_cst_sent(language_model& lm,const corpus_t& corpus, args_t& args)
{
    CNLOG << "build or load CST";
    auto cst = build_or_load_cst(corpus, args);

    auto num_epochs = args["epochs"].as<size_t>();
    auto batch_size = args["batch_size"].as<size_t>();
    auto epoch_size = args["epoch_size"].as<size_t>();
    auto drop_out = args["drop_out"].as<double>();

    CNLOG << "start training cst sentence lm";
    CNLOG << "\tepochs = " << num_epochs;
    CNLOG << "\tepoch_size = " << num_epochs;
    CNLOG << "\tbatch_size = " << batch_size;
    CNLOG << "\tdrop_out = " << drop_out;
    auto dev_corpus_file = args["path"].as<std::string>() + "/" + constants::DEV_FILE;

    // (1) create the batches
    CNLOG << "create the batches. batch_size = " << batch_size;
    std::vector<instance_t> instances;
    auto prep_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < corpus.num_sentences; i++) {
        auto start_sent = corpus.text.begin() + corpus.sent_starts[i];
        size_t sent_len = corpus.sent_lens[i];
        instances.emplace_back(start_sent, sent_len);
    }
    auto prep_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prep_diff = prep_end - prep_start;
    CNLOG << "created batches in " << " - " << prep_diff.count() << "s";

    CNLOG << "NUMBER OF INSTANCES = " << instances.size();
    dynet::AdamTrainer trainer(lm.model, 0.001, 0.9, 0.999, 1e-8);
    trainer.clip_threshold = trainer.clip_threshold * batch_size;
    std::mt19937 gen(constants::RAND_SEED);
    std::uniform_int_distribution<> dis(0,100000000);
    for (size_t epoch = 1; epoch <= num_epochs; epoch++) {
        CNLOG << "start epoch " << epoch << "/" << num_epochs;

        CNLOG << "shuffle instances";
        // (0) remove existing padding if necessary
	    size_t max_len = 0;
        for (auto& instance : instances) {
            instance.sentence.resize(instance.real_len);
            instance.padding = 0;
            instance.rand = dis(gen);
        }
        // (1) perform a random shuffle that respects sentence len
        std::sort(instances.begin(), instances.end());

        // (2) add padding to instances in the same batch if necessary
        CNLOG << "add padding to instances in batch";
        {
            auto padd_sym = corpus.vocab.stop_sent_tok;
            auto itr = instances.begin();
            auto end = instances.end();
            while (itr != end) {
                auto batch_itr = itr;
                auto batch_end = batch_itr + std::min(batch_size,size_t(std::distance(itr,end))) - 1;
                while (batch_itr->sentence.size() != batch_end->sentence.size()) {
                    size_t to_add = batch_end->sentence.size() - batch_itr->sentence.size();
                    for (size_t i = 0; i < to_add; i++) {
                        batch_itr->sentence.push_back(padd_sym);
                        batch_itr->padding++;
                    }
                    ++batch_itr;
                }
                itr = batch_end + 1;
            }
        }

        CNLOG << "start training...";
        auto start = instances.begin();
        auto last_report = instances.begin();
        auto itr = instances.begin();
        auto end = instances.end();
        while (itr != end) {
            auto batch_end = itr + std::min(batch_size,size_t(std::distance(itr,end)));
            auto actual_batch_size = std::distance(itr,batch_end);

            auto batch_loss_start = std::chrono::high_resolution_clock::now();
            auto batch_losses = compute_batch_losses(cst,corpus,itr,batch_end);
            auto batch_loss_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double>  batch_loss_diff = batch_loss_end - batch_loss_start;
            auto time_per_loss_instance = batch_loss_diff.count() / actual_batch_size * 1000.0;

            dynet::ComputationGraph cg;
            auto train_start = std::chrono::high_resolution_clock::now();
            auto loss_tuple = build_train_graph_cst_sent(lm,cg, itr, batch_end,batch_losses,corpus,drop_out);
            auto loss_expr = std::get<0>(loss_tuple);
            auto num_predictions = std::get<1>(loss_tuple);
            auto loss_float = dynet::as_scalar(cg.forward(loss_expr));
            auto instance_loss = loss_float / num_predictions;
            cg.backward(loss_expr);
            trainer.update();
            itr = batch_end;
            auto train_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> train_diff = train_end - train_start;
            auto time_per_instance = train_diff.count() / actual_batch_size * 1000.0;

            if (std::distance(last_report, itr) > 32768 || batch_end == end) {
                double percent = double(std::distance(start, itr)) / double(instances.size()) * 100;
                last_report = itr;
                CNLOG << std::fixed << std::setprecision(1) << std::floor(percent) << "% "
                      << std::distance(start, itr) << "/" << instances.size()
                      << " batch_size = " << actual_batch_size
                      << " FW/BW/UPDATE  - loss_comp "
                      << time_per_loss_instance << "ms/instance | train "
                      << time_per_instance << "ms/instance - loss = " << instance_loss;
            }
        }
        CNLOG << "finish epoch " << epoch << ". compute dev pplx ";

        auto pplx = evaluate_pplx(lm, corpus.vocab, dev_corpus_file);
        CNLOG << "epoch " << epoch MM << " dev pplx = " << pplx;
    }

    return lm;
}
