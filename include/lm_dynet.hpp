#pragma once

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

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>

#include "constants.hpp"
#include "data_loader.hpp"
#include "logging.hpp"
#include "lm_common.hpp"

using namespace std;
using namespace dynet;

struct instance_t {
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
std::tuple<dynet::Expression, size_t> 
build_train_graph_dynet(language_model& lm,dynet::ComputationGraph& cg, t_itr& start, t_itr& end,double drop_out)
{
    size_t batch_size = std::distance(start, end);
    size_t sentence_len = start->sentence.size();

    lm.rnn.new_graph(cg);
    lm.rnn.start_new_sequence();
    if(drop_out != 0.0) {
        lm.rnn.set_dropout(drop_out);
    }

    lm.i_R = dynet::parameter(cg, lm.p_R);
    lm.i_bias = dynet::parameter(cg, lm.p_bias);

    std::vector<Expression> errs;
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
        auto i_err = dynet::pickneglogsoftmax(i_r_t, next_tok);
        errs.push_back(i_err);
        // Change input
        current_tok = next_tok;
    }
    // Add all errors
    return std::make_tuple(sum_batches(sum(errs)), actual_predictions);
}

void train_dynet_lm(language_model& lm,const corpus_t& corpus, args_t& args)
{
    auto num_epochs = args["epochs"].as<size_t>();
    auto batch_size = args["batch_size"].as<size_t>();
    auto report_size = args["report_size"].as<size_t>();
    auto drop_out = args["drop_out"].as<double>();

    CNLOG << "start training dynet lm";
    CNLOG << "\tepochs = " << num_epochs;
    CNLOG << "\tbatch_size = " << batch_size;
    CNLOG << "\tdrop_out = " << drop_out;

    auto dev_corpus_file = args["path"].as<std::string>() + "/" + constants::DEV_FILE;

    // (1) create the batches
    CNLOG << "create the batches. batch_size = " << batch_size;
    std::vector<instance_t> sentences;
    auto prep_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < corpus.num_sentences; i++) {
        auto start_sent = corpus.text.begin() + corpus.sent_starts[i];
        size_t sent_len = corpus.sent_lens[i];
        sentences.emplace_back(start_sent, sent_len);
    }
    auto prep_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prep_diff = prep_end - prep_start;
    CNLOG << "created batches in " << " - " << prep_diff.count() << "s";

    CNLOG << "number of sentences = " << sentences.size();
    dynet::AdamTrainer trainer(lm.model, 0.001, 0.9, 0.999, 1e-8);
    trainer.clip_threshold = trainer.clip_threshold * batch_size;
    std::mt19937 gen(constants::RAND_SEED);
    std::uniform_int_distribution<> dis(0,100000000);
    for (size_t epoch = 1; epoch <= num_epochs; epoch++) {
        CNLOG << "start epoch " << epoch << "/" << num_epochs;

        CNLOG << "shuffle sentences";
        // (0) remove existing padding if necessary
	    size_t max_len = 0;
        for (auto& instance : sentences) {
            instance.sentence.resize(instance.real_len);
	        instance.padding = 0;
	        instance.rand = dis(gen);
        }
        // (1) perform a random shuffle that respects sentence len
        std::sort(sentences.begin(), sentences.end());

        // (2) add padding to instances in the same batch if necessary
        CNLOG << "add padding to instances in batch";
        {
            auto padd_sym = corpus.vocab.stop_sent_tok;
            auto itr = sentences.begin();
            auto end = sentences.end();
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
        auto start = sentences.begin();
        auto last_report = sentences.begin();
        auto itr = sentences.begin();
        auto end = sentences.end();
        while (itr != end) {
            auto batch_end = itr + std::min(batch_size,size_t(std::distance(itr,end)));
            auto actual_batch_size = std::distance(itr,batch_end);

            dynet::ComputationGraph cg;
            auto train_start = std::chrono::high_resolution_clock::now();
            auto loss_tuple = build_train_graph_dynet(lm,cg, itr, batch_end,drop_out);
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

            if (std::distance(last_report, itr) > report_size || batch_end == end) {
                double percent = double(std::distance(start, itr)) / double(sentences.size()) * 100;
                last_report = itr;
                CNLOG << std::fixed << std::setprecision(1) << std::floor(percent) << "% "
                      << std::distance(start, itr) << "/" << sentences.size()
                      << " batch_size = " << actual_batch_size
                      << " TIME = "<< time_per_instance << "ms/instance"
                      << " num_predictions = " << num_predictions
                      << " ppl = " << exp(instance_loss);
            }
        }
        CNLOG << "finish epoch " << epoch << ". compute dev pplx ";

        auto pplx = evaluate_pplx(lm, corpus, dev_corpus_file);
        CNLOG << "epoch " << epoch << " dev pplx = " << pplx;
    }
}
