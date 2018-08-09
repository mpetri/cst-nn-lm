#pragma once

#include "constants.hpp"
#include "cst.hpp"
#include "data_loader.hpp"
#include "logging.hpp"
#include "lm_cst_common.hpp"
#include "lm_common.hpp"

template<class t_trainer>
void train_cst_rand(language_model& lm,const corpus_t& corpus, args_t& args,t_trainer& trainer)
{
    CNLOG << "build or load CST";
    auto cst = build_or_load_cst(corpus, args);

    auto num_epochs = args["epochs"].as<size_t>();
    auto batch_size = args["batch_size"].as<size_t>();
    auto drop_out = args["drop_out"].as<double>();
    int64_t report_interval = args["report_interval"].as<size_t>();

    CNLOG << "start training cst sentence lm with rand batch order";
    CNLOG << "\tepochs = " << num_epochs;
    CNLOG << "\tbatch_size = " << batch_size;
    CNLOG << "\tdrop_out = " << drop_out;
    auto dev_corpus_file = args["path"].as<std::string>() + "/" + constants::DEV_FILE;

    // (1) create the batches
    CNLOG << "create the batches. batch_size = " << batch_size;
    std::vector<prefix_batch_t> prefix_batches;
    std::vector<one_hot_batch_t> one_hot_batches;
    auto prep_start = std::chrono::high_resolution_clock::now();
    std::tie(prefix_batches,one_hot_batches) = create_train_batches(cst,corpus,args,batch_size);
    auto prep_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prep_diff = prep_end - prep_start;
    CNLOG << "created batches in " << " - " << prep_diff.count() << "s";

    std::mt19937 rng(constants::RAND_SEED);
    std::vector<uint32_t> batch_ids(prefix_batches.size()+one_hot_batches.size());
    for(size_t i=0;i<batch_ids.size();i++) batch_ids[i] = i;

    double best_pplx = std::numeric_limits<double>::max();
    for (size_t epoch = 1; epoch <= num_epochs; epoch++) {
        CNLOG << "start epoch " << epoch << "/" << num_epochs;
        std::shuffle(batch_ids.begin(),batch_ids.end(), rng);

        std::vector<float> window_loss(20);
        std::vector<float> window_predictions(20);
        size_t last_report = 0;
        size_t next_dev = 100;
        for(size_t i=0;i<batch_ids.size();i++) {
            auto train_start = std::chrono::high_resolution_clock::now();
            auto cur_batch_id = batch_ids[i];

            dynet::Expression loss;
            size_t num_predictions;
            dynet::ComputationGraph cg;
            float loss_float;
            std::string batch_type = "S";
            if(cur_batch_id >= prefix_batches.size()) {
                auto& cur_batch = one_hot_batches[cur_batch_id-prefix_batches.size()];
                trainer.clip_threshold = trainer.clip_threshold * cur_batch.size;
                std::tie(loss,num_predictions) = build_train_graph_sents(lm,cg,cur_batch,drop_out);
                loss_float = dynet::as_scalar(cg.forward(loss));
                cg.backward(loss);
                trainer.update();
            } else {
                batch_type = "P";
                auto& cur_batch = prefix_batches[cur_batch_id];
                compute_dist(cur_batch,cst,corpus);

                trainer.clip_threshold = trainer.clip_threshold * cur_batch.size;
                std::tie(loss,num_predictions) = build_train_graph_prefix(lm,cg,cur_batch,drop_out);
                loss_float = dynet::as_scalar(cg.forward(loss));
                cg.backward(loss);
                trainer.update();
                cur_batch.dist.clear();
                cur_batch.dist.shrink_to_fit();
            }

            auto instance_loss = loss_float / num_predictions;
            auto train_stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> train_diff = train_stop - train_start;
            auto time_per_instance = train_diff.count() / num_predictions * 1000.0;
            window_loss[i%window_loss.size()] = loss_float;
            window_predictions[i%window_loss.size()] = num_predictions;

            if ( int64_t(i-last_report) >= report_interval || i+1 == batch_ids.size()) {
                double percent = double(i) / double(batch_ids.size()) * 100;
                float wloss = std::accumulate(window_loss.begin(),window_loss.end(), 0.0);
                float wpred = std::accumulate(window_predictions.begin(),window_predictions.end(), 0.0);
                last_report = i;
                CNLOG << std::fixed << std::setprecision(1) << std::floor(percent) << "% "
                      << (i+1) << "/" << batch_ids.size()
                      << " batch_type = " << batch_type
                      << " num_predictions = " << num_predictions
                      << " TIME = "<< time_per_instance << "ms/instance"
                      << " ABSTIME = "<< train_diff.count()* 1000.0 << "ms"
                      << " ppl = " << exp(instance_loss)
                      << " avg-ppl = " << exp(wloss / wpred);
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

