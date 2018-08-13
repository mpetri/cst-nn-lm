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

#include <boost/progress.hpp>

struct language_model_ngram {
    dynet::ParameterCollection model;
    uint32_t INPUT_DIM;
    uint32_t HIDDEN_DIM;
    uint32_t NGRAM_SIZE;
    uint32_t VOCAB_SIZE;
    dynet::LookupParameter p_c;
    dynet::Parameter p_R;
    dynet::Parameter p_bias;
    dynet::Parameter p_O;
    dynet::Parameter p_bias_O;

    dynet::Expression i_c;
    dynet::Expression i_R;
    dynet::Expression i_bias;
    dynet::Expression i_O;
    dynet::Expression i_bias_O;

    language_model_ngram(const vocab_t& vocab, args_t& args)
    {
        INPUT_DIM = args["input_dim"].as<uint32_t>();
        HIDDEN_DIM = args["hidden_dim"].as<uint32_t>();
        NGRAM_SIZE = args["ngram_size"].as<uint32_t>();
        VOCAB_SIZE = vocab.size();
        CNLOG << "LM-ngram parameters ";
        CNLOG << "\tngram_size = " << NGRAM_SIZE;
        CNLOG << "\tinput_dim = " << INPUT_DIM;
        CNLOG << "\thidden_dim = " << HIDDEN_DIM;
        CNLOG << "\tvocab_size = " << VOCAB_SIZE;

        // Add embedding parameters to the model
        p_c = model.add_lookup_parameters(VOCAB_SIZE, { INPUT_DIM });
        p_R = model.add_parameters({ HIDDEN_DIM,INPUT_DIM*NGRAM_SIZE });
        p_O = model.add_parameters({ VOCAB_SIZE, HIDDEN_DIM });
        p_bias = model.add_parameters({ HIDDEN_DIM });
        p_bias_O = model.add_parameters({ VOCAB_SIZE });
    }

    void store(std::string file_name) {
        dynet::TextFileSaver s(file_name);
        s.save(model);
    }

    void load(std::string file_name) {
        if (boost::filesystem::exists(file_name)) {
            dynet::TextFileLoader l(file_name);
            l.populate(model);
        } else {
            CNLOG << "ERROR: model file " << file_name << " does not exist.";
        }
    }

};

template <class t_itr>
std::tuple<dynet::Expression, size_t>
build_train_graph_ngram(language_model_ngram& lm,dynet::ComputationGraph& cg,const corpus_t& corpus,t_itr start, t_itr end, double drop_out = 0.0)
{
    size_t batch_size = std::distance(start, end);
    size_t sentence_len = start->sentence.size();

    lm.i_R = dynet::parameter(cg, lm.p_R);
    lm.i_bias = dynet::parameter(cg, lm.p_bias);

    lm.i_O = dynet::parameter(cg, lm.p_O);
    lm.i_bias_O = dynet::parameter(cg, lm.p_bias_O);

    std::vector<dynet::Expression> errs;
    // Set all inputs to the SOS symbol
    auto sos_tok = start->sentence.front();
    std::vector<uint32_t> current_tok(batch_size, sos_tok);
    std::vector<uint32_t> next_tok(batch_size);

    // create a vector of dynet expressions and fill with
    // padding for the first few computations
    std::vector<uint32_t> padding_tok(batch_size, corpus.vocab.eof_tok);
    std::vector< dynet::Expression > context;
    for(size_t i=0;i<lm.NGRAM_SIZE-1;i++)
        context.push_back(dynet::lookup(cg, lm.p_c, padding_tok));

    for (size_t i = 0; i < sentence_len - 1; ++i) {
        for (size_t j = 0; j < batch_size; j++) {
            auto instance = start + j;
            next_tok[j] = instance->sentence[i+1];
        }

        // Embed the current tokens
        context.push_back(dynet::lookup(cg, lm.p_c, current_tok));

        // Concact with the previous ngram-size-1 toks
        auto i_x_t = dynet::concatenate(context);

        if(drop_out != 0.0) {
            i_x_t = dynet::dropout(i_x_t,drop_out);
        }

        // Project to the token space using an affine transform
        auto i_r_t = dynet::rectify(lm.i_bias + i_R * i_x_t);

        if(drop_out != 0.0) {
            i_r_t = dynet::dropout(i_r_t,drop_out);
        }

        // back to vocab space
        auto i_y_t = lm.i_bias_O + lm.i_O * i_r_t;

        // Compute error for each member of the batch
        auto i_err = dynet::pickneglogsoftmax(i_y_t, next_tok);
        errs.push_back(i_err);

        // Change input tok and the context for the next prediction
        current_tok = next_tok;
        context.erase(context.begin());
    }
    // Add all errors
    return std::make_tuple(dynet::sum_batches(dynet::sum(errs)), errs.size()*batch_size);
}



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


double
evaluate_pplx(language_model_ngram& lm, const corpus_t& corpus, std::string file)
{
    double loss = 0.0;
    double predictions = 0;
    auto test_corpus = data_loader::parse_file(corpus.vocab, file);
    boost::progress_display show_progress(test_corpus.num_sentences);
    std::vector<instance_t> sents;
    for (size_t i = 0; i < test_corpus.num_sentences; i++) {
        auto start_sent = test_corpus.text.begin() + test_corpus.sent_starts[i];
        auto sent_len = test_corpus.sent_lens[i];
        sents.emplace_back(start_sent,sent_len);
    }
    for (size_t i = 0; i < test_corpus.num_sentences; i++) {
        dynet::ComputationGraph cg;
        auto loss_tuple = build_train_graph_ngram(lm, cg,corpus, sents.begin() + i, sents.begin() + i + 1);
        auto loss_expr = std::get<0>(loss_tuple);
        auto num_predictions = std::get<1>(loss_tuple);
        auto loss_float = dynet::as_scalar(cg.forward(loss_expr));
        loss += loss_float;
        predictions += num_predictions;
        ++show_progress;
    }
    return exp(loss / predictions);
}

template<class t_trainer>
void train_ngram_onehot(language_model_ngram& lm,const corpus_t& corpus, args_t& args,t_trainer& trainer)
{
    auto num_epochs = args["epochs"].as<size_t>();
    auto batch_size = args["batch_size"].as<size_t>();
    int64_t report_interval = args["report_interval"].as<size_t>();
    double drop_out = args["drop_out"].as<double>();

    CNLOG << "start training ngram lm";
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
    trainer.clip_threshold = trainer.clip_threshold * batch_size;
    std::mt19937 rng(constants::RAND_SEED);
    double best_pplx = std::numeric_limits<double>::max();

    // (2) add padding to instances in the same batch if necessary
    CNLOG << "add padding to instances in batch";
    std::vector< std::pair<uint32_t,uint32_t> > batch_start;
    {
        std::sort(sentences.begin(),sentences.end());
        auto padd_sym = corpus.vocab.eof_tok;
        auto start = sentences.begin();
        auto itr = sentences.begin();
        auto end = sentences.end();
        while (itr != end) {
            auto batch_itr = itr;
            auto batch_end = batch_itr + std::min(batch_size,size_t(std::distance(itr,end))) - 1;
            batch_start.emplace_back( (uint32_t) std::distance(start,itr) , (uint32_t) std::distance(itr,batch_end+1) );
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

    for (size_t epoch = 1; epoch <= num_epochs; epoch++) {
        CNLOG << "start epoch " << epoch << "/" << num_epochs;

        CNLOG << "shuffle batches";
        std::shuffle(batch_start.begin(),batch_start.end(), rng);

        CNLOG << "start training...";
        auto last_report = 0;
        std::vector<float> window_loss(20);
        std::vector<float> window_predictions(20);
        size_t next_dev = 100;
        for (size_t i = 0; i < batch_start.size(); i++) {
            auto batch_itr = sentences.begin() + batch_start[i].first;
            auto batch_end = batch_itr + batch_start[i].second;
            auto actual_batch_size = std::distance(batch_itr,batch_end);

            {
                dynet::ComputationGraph cg;
                auto train_start = std::chrono::high_resolution_clock::now();
                auto loss_tuple = build_train_graph_ngram(lm,cg,corpus, batch_itr, batch_end, drop_out);
                auto loss_expr = std::get<0>(loss_tuple);
                auto num_predictions = std::get<1>(loss_tuple);
                auto loss_float = dynet::as_scalar(cg.forward(loss_expr));
                window_loss[i%window_loss.size()] = loss_float;
                window_predictions[i%window_loss.size()] = num_predictions;
                auto instance_loss = loss_float / num_predictions;
                cg.backward(loss_expr);

                trainer.update();
                auto train_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> train_diff = train_end - train_start;
                auto time_per_instance = train_diff.count() / actual_batch_size * 1000.0;

                if ( int64_t(i-last_report) >= report_interval || i == batch_start.size() - 1) {
                    double percent = double(i+1) / double(batch_start.size()) * 100;
                    float wloss = std::accumulate(window_loss.begin(),window_loss.end(), 0.0);
                    float wpred = std::accumulate(window_predictions.begin(),window_predictions.end(), 0.0);
                    last_report = i;
                    CNLOG << std::fixed << std::setprecision(1) << std::floor(percent) << "% "
                        << i+1 << "/" << batch_start.size()
                        << " batch_size = " << actual_batch_size
                        << " TIME = "<< time_per_instance << "ms/instance"
                        << " slen = " << batch_itr->sentence.size()
                        << " num_predictions = " << num_predictions
                        << " ppl = " << exp(instance_loss)
                        << " avg-ppl = " << exp(wloss / wpred);
                }
            }

            if( (i+1) == next_dev) {
                CNLOG << "evalute dev pplx";
                auto pplx = evaluate_pplx(lm, corpus, dev_corpus_file);
                CNLOG << "epoch " << epoch << ". processed " << i+1 << " batches. evaluate dev pplx = " << pplx;
                next_dev = next_dev * 2;
            }
        }
        CNLOG << "finish epoch " << epoch << ". compute dev pplx ";
        auto pplx = evaluate_pplx(lm, corpus, dev_corpus_file);
        CNLOG << "epoch " << epoch << " dev pplx = " << pplx;


        if( pplx < best_pplx && args.count("store") ) {
            CNLOG << "better dev pplx. store model";
            best_pplx = pplx;
            auto lm_file_path = args["store"].as<std::string>();
            CNLOG << "store language model to " << lm_file_path;
            lm.store(lm_file_path);
        }

    }
}
