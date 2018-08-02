#pragma once

#include "dynet/dict.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/nodes.h"
#include "dynet/rnn.h"
#include "dynet/timing.h"
#include "dynet/training.h"
#include <dynet/io.h>

struct language_model {
    dynet::ParameterCollection model;
    uint32_t LAYERS;
    uint32_t INPUT_DIM;
    uint32_t HIDDEN_DIM;
    uint32_t VOCAB_SIZE;
    dynet::LookupParameter p_c;
    dynet::Parameter p_R;
    dynet::Parameter p_bias;
    dynet::Expression i_c;
    dynet::Expression i_R;
    dynet::Expression i_bias;
    dynet::LSTMBuilder rnn;

    language_model(const vocab_t& vocab, args_t& args)
    {
        LAYERS = args["layers"].as<uint32_t>();
        INPUT_DIM = args["input_dim"].as<uint32_t>();
        HIDDEN_DIM = args["hidden_dim"].as<uint32_t>();
        VOCAB_SIZE = vocab.size();
        CNLOG << "LM parameters ";
        CNLOG << "\tlayers = " << LAYERS;
        CNLOG << "\tinput_dim = " << INPUT_DIM;
        CNLOG << "\thidden_dim = " << HIDDEN_DIM;
        CNLOG << "\tvocab size = " << VOCAB_SIZE;

        // Add embedding parameters to the model
        p_c = model.add_lookup_parameters(VOCAB_SIZE, { INPUT_DIM });
        p_R = model.add_parameters({ VOCAB_SIZE, HIDDEN_DIM });
        p_bias = model.add_parameters({ VOCAB_SIZE });
        rnn = dynet::LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model);
    }

    void store(std::string file_name) {
        dynet::TextFileSaver s(file_name);
        s.save(model);
    }

    void load(std::string file_name) {
        dynet::TextFileLoader l(file_name);
        l.populate(m);
    }
};


template <class t_itr>
dynet::Expression build_valid_graph(language_model& lm,dynet::ComputationGraph& cg, t_itr itr, size_t len)
{
    lm.rnn.new_graph(cg);
    lm.rnn.start_new_sequence();
    lm.rnn.disable_dropout();

    lm.i_R = dynet::parameter(cg, lm.p_R);
    lm.i_bias = dynet::parameter(cg, lm.p_bias);

    std::vector<dynet::Expression> errors(len - 1);
    for (size_t i = 0; i < len - 1; i++) {
        auto cur_sym = *itr++;
        auto next_sym = *itr;
        dynet::Expression i_x_t = dynet::lookup(cg, lm.p_c, cur_sym);
        dynet::Expression i_y_t = lm.rnn.add_input(i_x_t);
        dynet::Expression i_r_t = lm.i_bias + lm.i_R * i_y_t;
        errors[i] = dynet::pickneglogsoftmax(i_r_t, next_sym);
    }
    return dynet::sum(errors);
}

double
evaluate_pplx(language_model& lm, const corpus_t& corpus, std::string file)
{
    double loss = 0.0;
    double predictions = 0;
    auto test_corpus = data_loader::parse_file(corpus.vocab, file);
    boost::progress_display show_progress(test_corpus.num_sentences);
    for (size_t i = 0; i < test_corpus.num_sentences; i++) {
        auto start_sent = test_corpus.text.begin() + test_corpus.sent_starts[i];
        auto sent_len = test_corpus.sent_lens[i];
        dynet::ComputationGraph cg;
        auto loss_expr = build_valid_graph(lm, cg, start_sent, sent_len);
        loss += dynet::as_scalar(cg.forward(loss_expr));
        predictions += sent_len - 1;
        ++show_progress;
    }
    return exp(loss / predictions);
}

