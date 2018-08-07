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


struct language_model_ngram {
    dynet::ParameterCollection model;
    uint32_t INPUT_DIM;
    uint32_t HIDDEN_DIM;
    uint32_t NGRAM_SIZE;
    uint32_t VOCAB_SIZE;
    dynet::LookupParameter p_c;
    dynet::Parameter p_R;
    dynet::Parameter p_bias;
    dynet::Expression i_c;
    dynet::Expression i_R;
    dynet::Expression i_bias;

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
        p_R = model.add_parameters({ INPUT_DIM*NGRAM_SIZE, HIDDEN_DIM });
        p_bias = model.add_parameters({ HIDDEN_DIM });
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
build_train_graph_ngram(language_model_ngram& lm,dynet::ComputationGraph& cg,corpus_t& corpus,t_itr& start, t_itr& end,double drop_out)
{
    size_t batch_size = std::distance(start, end);
    size_t sentence_len = start->sentence.size();

    lm.i_R = dynet::parameter(cg, lm.p_R);
    lm.i_bias = dynet::parameter(cg, lm.p_bias);

    std::vector<dynet::Expression> errs;
    // Set all inputs to the SOS symbol
    auto sos_tok = start->sentence.front();
    std::vector<uint32_t> current_tok(batch_size, sos_tok);
    std::vector<uint32_t> next_tok(batch_size);

    // create a vector of dynet expressions and fill with
    // padding for the first few computations
    std::vector<uint32_t> padding_tok(batch_size, corpus.vocab.eof_tok);
    std::vector< dynet::Expression > context(sentence_len + lm.NGRAM_SIZE);
    for(size_t i=0;i<lm.NGRAM_SIZE;i++)
        context[i] = dynet::lookup(cg, lm.p_c, padding_tok);
    auto ctx_start = context.begin();
    auto ctx_end = context.begin() + lm.NGRAM_SIZE - 1;

    for (size_t i = 0; i < sentence_len - 1; ++i) {
        for (size_t j = 0; j < batch_size; j++) {
            auto instance = start + j;
            next_tok[j] = instance->sentence[i+1];
        }

        // Embed the current tokens
        *ctx_end = dynet::lookup(cg, lm.p_c, current_tok);

        // Concact with the previous ngram-size-1 toks
        auto i_x_t = dynet::concatenate(ctx_start,ctx_end);

        // Project to the token space using an affine transform
        auto i_r_t = dynet::rectify(lm.i_bias + lm.i_R * i_x_t);

        // Compute error for each member of the batch
        auto i_err = dynet::pickneglogsoftmax(i_r_t, next_tok);
        errs.push_back(i_err);

        // Change input tok and the context for the next prediction
        current_tok = next_tok;
        ++ctx_start;
        ++ctx_end;
    }
    // Add all errors
    return std::make_tuple(dynet::sum_batches(dynet::sum(errs)), errs.size()*batch_size);
}


double
evaluate_pplx(language_model_ngram& lm, const corpus_t& corpus, std::string file)
{
    double loss = 0.0;
    double predictions = 0;
    auto test_corpus = data_loader::parse_file(corpus.vocab, file);
    boost::progress_display show_progress(test_corpus.num_sentences);
    std::vector< std::vector<uint32_t> > sents;
    for (size_t i = 0; i < test_corpus.num_sentences; i++) {
        auto start_sent = test_corpus.text.begin() + test_corpus.sent_starts[i];
        auto sent_len = test_corpus.sent_lens[i];
        sents.emplace_back(start_sent,start_sent+sent_len);
    }
    for (size_t i = 0; i < test_corpus.num_sentences; i++) {
        dynet::ComputationGraph cg;
        auto loss_expr = build_train_graph_ngram(lm, cg, sents + i, sents + i + 1);
        loss += dynet::as_scalar(cg.forward(loss_expr));
        predictions += sent_len - 1;
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

    CNLOG << "start training dynet lm";
    CNLOG << "\tepochs = " << num_epochs;
    CNLOG << "\tbatch_size = " << batch_size;

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
    std::mt19937 gen(constants::RAND_SEED);
    std::uniform_int_distribution<> dis(0,100000000);
    double best_pplx = std::numeric_limits<double>::max();
    for (size_t epoch = 1; epoch <= num_epochs; epoch++) {
        CNLOG << "start epoch " << epoch << "/" << num_epochs;

        CNLOG << "shuffle sentences";
        // (0) remove existing padding if necessary
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
        double total_predictions = 0;
        double total_loss = 0;
        while (itr != end) {
            auto batch_end = itr + std::min(batch_size,size_t(std::distance(itr,end)));
            auto actual_batch_size = std::distance(itr,batch_end);

            dynet::ComputationGraph cg;
            auto train_start = std::chrono::high_resolution_clock::now();
            auto loss_tuple = build_train_graph_ngram(lm,cg, itr, batch_end,drop_out);
            auto loss_expr = std::get<0>(loss_tuple);
            auto num_predictions = std::get<1>(loss_tuple);
            auto loss_float = dynet::as_scalar(cg.forward(loss_expr));
            total_loss += loss_float;
            total_predictions += num_predictions;
            auto instance_loss = loss_float / num_predictions;
            cg.backward(loss_expr);
            trainer.update();
            itr = batch_end;
            auto train_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> train_diff = train_end - train_start;
            auto time_per_instance = train_diff.count() / actual_batch_size * 1000.0;

            if (std::distance(last_report, itr) >= report_interval || batch_end == end) {
                double percent = double(std::distance(start, itr)) / double(sentences.size()) * 100;
                last_report = itr;
                CNLOG << std::fixed << std::setprecision(1) << std::floor(percent) << "% "
                      << std::distance(start, itr) << "/" << sentences.size()
                      << " batch_size = " << actual_batch_size
                      << " TIME = "<< time_per_instance << "ms/instance"
                      << " num_predictions = " << num_predictions
                      << " ppl = " << exp(instance_loss)
                      << " avg-ppl = " << exp(total_loss / total_predictions);
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
