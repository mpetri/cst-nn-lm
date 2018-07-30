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
    bool one_hot = false;
    bool processed = false;
    cst_node_type cst_node;
    size_t num_occ;
    size_t num_children;
    std::vector<uint32_t> prefix;
    std::vector<uint32_t> suffix;
    std::vector<float> dist;
    bool operator<(const train_instance_t& other) const
    {
        if (prefix.size() == other.prefix.size()) {
            return num_occ > other.num_occ;
        }
        return prefix.size() < other.prefix.size();
    }
};

using instances_t = std::vector<train_instance_t>;

void add_new_instance(const corpus_t& corpus,const cst_type& cst,instances_t& instances,
                                     train_instance_t& parent,uint32_t tok,cst_node_type cst_node)
{
    if(tok == corpus.vocab.stop_sent_tok) return;
    train_instance_t new_instance;
    new_instance.cst_node = cst_node;
    new_instance.prefix = parent.prefix;
    new_instance.prefix.push_back(tok);
    if(cst.is_leaf(cst_node)) {
        // new node is leaf node, create special node
        new_instance.one_hot = true;
        new_instance.num_children = 1;
        new_instance.num_occ = 1;

        auto cur_depth = new_instance.prefix.size();
        auto next_tok = cst.edge(cst_node, cur_depth + 1);
        while(next_tok != corpus.vocab.stop_sent_tok) {
            new_instance.suffix.push_back(next_tok);
            cur_depth++;
            next_tok = cst.edge(cst_node, cur_depth + 1);
        }
        new_instance.suffix.push_back(next_tok);
    } else {
        // new node is NOT a leafe node
        new_instance.one_hot = false;
        new_instance.processed = false;
        new_instance.num_children = cst.degree(cst_node);
        new_instance.num_occ = cst.size(cst_node);
    }
    instances.push_back(new_instance);
}

std::vector<float> create_true_dist(const corpus_t& corpus,const cst_type& cst,instances_t& instances,train_instance_t& instance)
{
    std::vector<float> dist(corpus.vocab.size(),0);
    if(instance.dist.size() != 0) { // prestored?
        return instance.dist;
    }
    auto node_depth = instance.prefix.size();
    for (const auto& child : cst.children(instance.cst_node)) {
        auto tok = cst.edge(child, node_depth + 2);
        double size = cst.size(child);
        dist[tok] = size;
        if(!instance.processed) {
            add_new_instance(corpus,cst,instances,instance,tok,child);
        }
    }
    instance.processed = true;
    if(instance.num_children >= 25) { // should we store this?
        instance.dist = dist;
    }
    return dist;
}

struct language_model3 {
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

    language_model3(const vocab_t& vocab, args_t& args)
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

    template <class t_itr>
    dynet::Expression build_train_graph(dynet::ComputationGraph& cg,instances_t& instances,size_t& cur_pos,size_t max_batch_size,
        const corpus_t& corpus,const cst_type& cst)
    {
        auto batch_size = max_batch_size;
        if(cur_pos + max_batch_size < instances.size()) {
            batch_size = instances.size() - cur_pos;
        }
        std::vector<dynet::Expression> errors;
        for(size_t i=0;i<batch_size;i++) {
            auto& instance = instances[cur_pos+i];

            // Initialize the RNN for a new computation graph
            rnn.new_graph(cg);
            // Prepare for new sequence (essentially set hidden states to 0)
            rnn.start_new_sequence();
            // Instantiate embedding parameters in the computation graph
            // output -> word rep parameters (matrix + bias)
            i_R = dynet::parameter(cg, p_R);
            i_bias = dynet::parameter(cg, p_bias);

            size_t prefix_len = instance.prefix.size();
            for (size_t i = 0; i < prefix_len - 1; ++i) {
                auto cur_tok = instance.prefix[i];
                auto i_x_t = dynet::lookup(cg, p_c, cur_tok);
                rnn.add_input(i_x_t);
            }
            auto last_prefix_tok = instance.prefix.back();
            auto i_x_t = dynet::lookup(cg, p_c, last_prefix_tok);
            auto i_y_t = rnn.add_input(i_x_t);

            if(instance.one_hot) {
                size_t suffix_len = instance.suffix.size();
                for (size_t i = 0; i < suffix_len - 1; ++i) {
                    auto cur_tok = instance.suffix[i];
                    // Project to the token space using an affine transform
                    auto i_r_t = i_bias + i_R * i_y_t;
                    // Compute error for each member of the batch
                    auto i_err = dynet::pickneglogsoftmax(i_r_t, cur_tok);
                    errors.push_back(i_err);
                    // Embed the current tokens
                    i_x_t = dynet::lookup(cg, p_c, cur_tok);
                    // Run one step of the rnn : y_t = RNN(x_t)
                    i_y_t = rnn.add_input(i_x_t);
                }
                auto i_r_t = i_bias + i_R * i_y_t;
                auto last_tok = instance.suffix.back();
                auto i_err = dynet::pickneglogsoftmax(i_r_t, last_tok);
                errors.push_back(i_err);
            } else {
                auto dist = create_true_dist(corpus,cst,instances,instance);
                unsigned int dist_len = dist.size();
                dynet::Expression i_r_t = i_bias + i_R * i_y_t;
                dynet::Expression i_pred = -dynet::log_softmax(i_r_t);
                dynet::Expression i_pred_linear = dynet::reshape(i_pred, {dist_len});
                dynet::Expression i_true = dynet::input(cg, {dist_len}, dist);
                dynet::Expression i_error = dynet::transpose(i_true) * i_pred_linear;
                errors.push_back(i_error);
            }
        }
        cur_pos += batch_size;
        return dynet::sum(errors);
    }

    template <class t_itr>
    dynet::Expression build_valid_graph(dynet::ComputationGraph& cg, t_itr itr, size_t len)
    {

        // Initialize the RNN for a new computation graph
        rnn.new_graph(cg);
        // Prepare for new sequence (essentially set hidden states to 0)
        rnn.start_new_sequence();
        // Instantiate embedding parameters in the computation graph
        // output -> word rep parameters (matrix + bias)
        i_R = dynet::parameter(cg, p_R);
        i_bias = dynet::parameter(cg, p_bias);

        std::vector<dynet::Expression> errors(len - 1);
        for (size_t i = 0; i < len - 1; i++) {
            auto cur_sym = *itr++;
            auto next_sym = *itr;
            dynet::Expression i_x_t = dynet::lookup(cg, p_c, cur_sym);
            dynet::Expression i_y_t = rnn.add_input(i_x_t);
            dynet::Expression i_r_t = i_bias + i_R * i_y_t;
            errors[i] = pickneglogsoftmax(i_r_t, next_sym);
        }
        return dynet::sum(errors);
    }
};

double
evaluate_pplx(language_model3& lm, const corpus_t& corpus, std::string file)
{
    double loss = 0.0;
    double predictions = 0;

    auto test_corpus = data_loader::parse_file(corpus.vocab, file);
    boost::progress_display show_progress(corpus.num_sentences);
    for (size_t i = 0; i < test_corpus.num_sentences; i++) {
        auto start_sent = test_corpus.text.begin() + test_corpus.sent_starts[i];
        auto sent_len = test_corpus.sent_lens[i];

        dynet::ComputationGraph cg;
        auto loss_expr = lm.build_valid_graph(cg, start_sent, sent_len);
        loss += dynet::as_scalar(cg.forward(loss_expr));
        predictions += sent_len - 1;
        ++show_progress;
    }
    return exp(loss / predictions);
}

void create_start_instance(const cst_type& cst,const corpus_t& corpus,instances_t& instances)
{
    auto lb = cst.csa.C[corpus.vocab.start_sent_tok];
    auto rb = cst.csa.C[corpus.vocab.start_sent_tok + 1] - 1;
    auto start_node = cst.node(lb,rb); // cst node of <s>
    train_instance_t start_instance;
    start_instance.num_children = cst.degree(start_node);
    start_instance.num_occ = rb-lb+1;
    start_instance.prefix.push_back(corpus.vocab.start_sent_tok);
    instances.push_back(start_instance);
}

language_model3 create_lm(const cst_type& cst, const corpus_t& corpus, args_t& args)
{
    auto dev_corpus_file = args["path"].as<std::string>() + "/" + constants::DEV_FILE;
    std::mt19937 rng(constants::RAND_SEED);

    auto num_epochs = args["epochs"].as<size_t>();
    auto batch_size = args["batch_size"].as<size_t>();
    language_model3 lm(corpus.vocab, args);

    instances_t instances;
    size_t cur = 0;
    create_start_instance(corpus,instances,cst);
    dynet::AdamTrainer trainer(lm.model, 0.001, 0.9, 0.999, 1e-8);
    for (size_t epoch = 1; epoch <= num_epochs; epoch++) {
        CNLOG << "start epoch " << epoch << "/" << num_epochs;

        while(cur != instances.size()) {
            auto done_before = cur;
            dynet::ComputationGraph cg;
            auto loss_expr = lm.build_train_graph(cg,instances,cur,batch_size,corpus,cst);
            auto loss_float = dynet::as_scalar(cg.forward(loss_expr));
            auto done_after = cur;
            auto actual_batch_size = done_after - done_before;
            auto instance_loss = loss_float / actual_batch_size;
            cg.backward(loss_expr);
            trainer.update();
        }

        CNLOG << "finish epoch " << epoch << ". compute dev pplx ";

        auto pplx = evaluate_pplx(lm, corpus, dev_corpus_file);
        CNLOG << "epoch dev pplx = " << pplx;

        // restart with new ordering for next epoch
        cur = 0;
        std::shuffle(instances.begin(),instances.end(), rng);
    }

    return lm;
}
