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

struct train_instance_t {
    size_t num_occ;
    cst_node_type cst_node;
    size_t num_children;
    std::vector<uint32_t> prefix;
    bool operator<(const train_instance_t& other) const
    {
        if (prefix.size() == other.prefix.size()) {
            return num_occ > other.num_occ;
        }
        return prefix.size() < other.prefix.size();
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
    std::tuple<dynet::Expression, size_t> build_train_graph_batch_ngram(dynet::ComputationGraph& cg, t_itr& start, t_itr& end,
        std::vector<float>& dists, size_t dist_len)
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

        float* dist_ptr = dists.data();

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
            dynet::Expression i_pred_linear = dynet::reshape(i_pred, { (unsigned int)(dist_len*batch_size) });
            dynet::Expression i_true = dynet::input(cg, { (unsigned int)(dist_len*batch_size) }, dist_ptr);
            dynet::Expression i_error = dynet::transpose(i_true) * i_pred_linear;
            dist_ptr += (dist_len*batch_size);
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

std::string
print_prefix(std::vector<uint32_t>& prefix, const vocab_t& vocab)
{
    std::string s = "[";
    for (size_t i = 0; i < prefix.size() - 1; i++) {
        s += vocab.inverse_lookup(prefix[i]) + " ";
    }
    return s + vocab.inverse_lookup(prefix.back()) + "]";
}

std::vector<train_instance_t>
create_instances(const cst_type& cst, const vocab_t& vocab, cst_node_type cst_node, std::vector<uint32_t> prefix, size_t threshold)
{
    std::vector<train_instance_t> instances;
    if (prefix.back() < vocab.start_sent_tok)
        return instances;

    double node_size = cst.size(cst_node);
    if (node_size >= threshold) {
        train_instance_t new_instance;
        new_instance.num_occ = node_size;
        new_instance.prefix = prefix;
        new_instance.cst_node = cst_node;
        new_instance.num_children = 0;
        auto node_depth = cst.depth(cst_node);
        for (const auto& child : cst.children(cst_node)) {
            auto tok = cst.edge(child, node_depth + 1);
            double size = cst.size(child);
            if (tok != vocab.start_sent_tok && tok != vocab.stop_sent_tok && size >= threshold) {
                auto child_prefix = prefix;
                child_prefix.push_back(tok);
                auto child_instances = create_instances(cst, vocab, child, child_prefix, threshold);
                instances.insert(instances.end(), child_instances.begin(), child_instances.end());
            }
            new_instance.num_children++;
        }
        instances.push_back(new_instance);
    }
    return instances;
}

template <class t_dist_itr>
void create_dist(const cst_type& cst, const train_instance_t& instance, t_dist_itr dist_itr, size_t vocab_size)
{
    double node_size = instance.num_occ;
    auto node_depth = cst.depth(instance.cst_node);
    for (size_t i = 0; i < vocab_size; i++) {
        *(dist_itr + i) = 0;
    }
    for (const auto& child : cst.children(instance.cst_node)) {
        auto tok = cst.edge(child, node_depth + 1);
        double size = cst.size(child);
        *(dist_itr + tok) = size / node_size;
    }
}

std::vector<train_instance_t>
process_token_subtree(const cst_type& cst, const vocab_t& vocab, size_t start, size_t step, size_t threshold)
{
    std::vector<train_instance_t> instances;
    for (size_t j = start; j < vocab.size(); j += step) {
        size_t lb = cst.csa.C[j];
        size_t rb = cst.csa.C[j + 1] - 1;
        if (rb - lb + 1 >= threshold) {
            auto cst_node = cst.node(lb, rb);
            std::vector<uint32_t> prefix(1, j);
            auto subtree_instances = create_instances(cst, vocab, cst_node, prefix, threshold);
            instances.insert(instances.end(), subtree_instances.begin(), subtree_instances.end());
        }
    }
    return instances;
}

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

template <class t_itr>
void print_batch(t_itr& start, t_itr& end, std::vector<float>& dists, size_t dist_len)
{
    std::cout << "====================================================" << std::endl;
    size_t batch_size = std::distance(start, end);
    size_t vocab_size = dist_len / batch_size;
    for (size_t i = 0; i < batch_size; i++) {
        auto instance = start + i;
        auto dist = dists.begin() + (i * vocab_size);
        std::cout << std::setw(3) << i << " - " << instance->num_occ << " - " << instance->num_children << " [";
        for (size_t j = 0; j < instance->prefix.size() - 1; j++) {
            std::cout << instance->prefix[j] << ",";
        }
        std::cout << instance->prefix.back() << "] - <";
        for (size_t j = 0; j < vocab_size; j++) {
            if (dist[j] != 0) {
                std::cout << j << ":" << dist[j] << ",";
            }
        }
        std::cout << ">" << std::endl;
    }
    std::cout << "====================================================" << std::endl;
}

template<class t_itr>
std::vector<float> compute_batch_losses(const cst_type& cst,const corpus_t& corpus,t_itr itr,t_itr end) {
    std::vector<float> losses;
    losses.reserve(std::distance(itr,end)*corpus.vocab.size());
    std::vector<float> instance_loss(corpus.vocab.size());
    while(itr != end) {
        auto instance = *itr;
        auto cur_node = cst.root();
        for(size_t i=0;i<instance.size()-1;i++) {
            auto& tok = instance[i];
            auto instance_loss_itr = instance_loss.begin();
            for (size_t i = 0; i < vocab_size; i++) {
                *(instance_loss_itr + i) = 0;
            }
            size_t char_pos;
            cur_node = cst.child(cur_node,tok,char_pos);
            if(cst.is_leaf(cur_node)) {
                // everything else is one hot
                instance_loss[ instance[i+1] ] = 1;
                std::copy(instance_loss.begin(),instance_loss.end(),std::back_inserter(losses));
                for(size_t j=i+1;j<instance.size()-1;j++) {
                    std::vector<float> one_hot_loss(corpus.vocab.size());
                    one_hot_loss[ instance[j+1] ] = 1;
                    std::copy(one_hot_loss.begin(),one_hot_loss.end(),std::back_inserter(losses));
                }
                break;
            } else {
                auto node_depth = cst.depth(cur_node);
                double node_size = cst.size(cur_node);
                for (const auto& child : cst.children(cur_node)) {
                    auto tok = cst.edge(child, node_depth + 1);
                    double size = cst.size(child);
                    *(instance_loss_itr + tok) = size / node_size;
                }
                std::copy(instance_loss.begin(),instance_loss.end(),std::back_inserter(losses));
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
    RNNBatchLanguageModel lm(corpus.vocab, args);

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

            auto batch_losses = compute_batch_losses(cst,itr,batch_end);

            dynet::ComputationGraph cg;
            auto train_start = std::chrono::high_resolution_clock::now();
            auto loss_tuple = lm.build_train_graph_batch(cg, itr, batch_end,batch_losses,corpus.vocab.size());
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
                      << " FW/BW/UPDATE  - "
                      << time_per_instance << "ms/instance - loss = " << instance_loss;
            }
        }
        CNLOG << "finish epoch " << epoch << ". compute dev pplx ";

        auto pplx = evaluate_pplx(lm, corpus.vocab, dev_corpus_file);
        CNLOG << "epoch dev pplx = " << pplx;
    }

    return lm;
}
