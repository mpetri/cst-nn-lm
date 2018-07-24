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

struct language_model2 {
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

    language_model2(const vocab_t& vocab, args_t& args)
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
    std::tuple<dynet::Expression, size_t> build_train_graph_batch(dynet::ComputationGraph& cg, t_itr& start, t_itr& end,
        std::vector<std::vector<float>>& dists)
    {
        size_t batch_size = std::distance(start, end);
        size_t sentence_len = start->sentence.size();

        // Initialize the RNN for a new computation graph
        rnn.new_graph(cg);
        // Prepare for new sequence (essentially set hidden states to 0)
        rnn.start_new_sequence();
        // Instantiate embedding parameters in the computation graph
        // output -> word rep parameters (matrix + bias)
        i_R = parameter(cg, p_R);
        i_bias = parameter(cg, p_bias);
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
                next_tok[j] = instance->sentence[i];
            }

            // Embed the current tokens
            auto i_x_t = dynet::lookup(cg, p_c, current_tok);
            // Run one step of the rnn : y_t = RNN(x_t)
            auto i_y_t = rnn.add_input(i_x_t);
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
        return std::make_tuple(sum_batches(sum(errs)), actual_predictions);
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
evaluate_pplx(language_model2& lm, const vocab_t& vocab, std::string file)
{
    double loss = 0.0;
    double predictions = 0;

    auto corpus = data_loader::parse_file(vocab, file);
    boost::progress_display show_progress(corpus.num_sentences);
    for (size_t i = 0; i < corpus.num_sentences; i++) {
        auto start_sent = corpus.text.begin() + corpus.sent_starts[i];
        auto sent_len = corpus.sent_lens[i];

        dynet::ComputationGraph cg;
        auto loss_expr = lm.build_valid_graph(cg, start_sent, sent_len);
        loss += dynet::as_scalar(cg.forward(loss_expr));
        predictions += sent_len - 1;
        ++show_progress;
    }
    return exp(loss / predictions);
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
            auto instance_loss_itr = losses[i].begin() + (corpus.vocab.size() * k);
            size_t char_pos;
            cur_node = cst.child(cur_node,tok,char_pos);
            if(cst.is_leaf(cur_node)) {
                // everything else is one hot
                *(instance_loss_itr + instance.sentence[i+1]) = 1;
                for(size_t j=i+1;j<instance.sentence.size()-1;j++) {
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
                        std::vector<float> stored_loss(corpus.vocab.size());
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


language_model2 create_lm(const cst_type& cst, const corpus_t& corpus, args_t& args)
{
    auto num_epochs = args["epochs"].as<size_t>();
    auto batch_size = args["batch_size"].as<size_t>();
    language_model2 lm(corpus.vocab, args);

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
    CNLOG << "CREATE INSTANCES "
          << " - " << prep_diff.count() << "s";

    CNLOG << "NUMBER OF INSTANCES = " << instances.size();
    dynet::AdamTrainer trainer(lm.model, 0.001, 0.9, 0.999, 1e-8);
    trainer.clip_threshold = trainer.clip_threshold * batch_size;
    std::mt19937 gen(12345);
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
            auto loss_tuple = lm.build_train_graph_batch(cg, itr, batch_end,batch_losses);
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

            if (std::distance(last_report, itr) > 8192 || batch_end == end) {
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
        CNLOG << "epoch dev pplx = " << pplx;
    }

    return lm;
}
