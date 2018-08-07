#include <chrono>
#include <iostream>

#include <boost/program_options.hpp>

#include "constants.hpp"
#include "data_loader.hpp"

#include "logging.hpp"

#include "lm_ngram.hpp"

namespace po = boost::program_options;

po::variables_map parse_args(int argc, char** argv)
{
    // clang-format off
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("path", po::value<std::string>()->required(), "data path")
        ("num_sents", po::value<uint32_t>()->default_value(defaults::MAX_NUM_SENTS), "number of sentences to train on. (prefix)")
        ("load", po::value<std::string>(), "load model instead of constructing one")
        ("store", po::value<std::string>(), "store model after construction")
        ("optimizer", po::value<std::string>(), "optimizer type: SGD or Adam")
        ("type", po::value<std::string>()->required(), "lm type")
        ("lr", po::value<double>()->default_value(0.0001), "learning rate")
        ("test", "test only. no train.")
        ("vocab_size", po::value<uint32_t>()->default_value(defaults::VOCAB_SIZE), "vocab size")
        ("ngram_size", po::value<uint32_t>()->default_value(defaults::NGRAM_SIZE), "ngram size")
        ("input_dim", po::value<uint32_t>()->default_value(defaults::INPUT_DIM), "input embedding size")
        ("hidden_dim", po::value<uint32_t>()->default_value(defaults::HIDDEN_DIM), "hidden size")
        ("epochs", po::value<size_t>()->default_value(defaults::EPOCHS), "num epochs")
        ("report_interval", po::value<size_t>()->default_value(defaults::REPORT_INTERVAL), "num epochs")
        ("batch_size", po::value<size_t>()->default_value(defaults::BATCH_SIZE), "batch size");
    // clang-format on

    po::variables_map args;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), args);
        po::notify(args);
    } catch (std::exception& e) {
        std::cerr << "Error:" << e.what() << std::endl;
        std::cout << desc << std::endl;
        exit(EXIT_FAILURE);
    }
    if (args.count("help")) {
        std::cout << desc << std::endl;
        exit(EXIT_SUCCESS);
    }

    CNLOG << "extract dynet cmdline parameters";
    DynetParams params = dynet::extract_dynet_params(argc, argv);
    params.random_seed = constants::RAND_SEED;
    dynet::initialize(params);

    return args;
}

int main(int argc, char** argv)
{
    init_logging();

    CNLOG << "parse arguments";
    auto args = parse_args(argc, argv);

    CNLOG << "load and parse data";
    auto corpus = data_loader::load(args);

    auto lm_type = args["type"].as<std::string>();
    language_model_ngram lm(corpus.vocab,args);

    if( args.count("load") ) {
        auto lm_file_path = args["load"].as<std::string>();
        CNLOG << "load language model from " << lm_file_path;
        lm.load(lm_file_path);
    } 

    dynet::Trainer* trainer = nullptr;
    double learning_rate = args["lr"].as<double>();
    if(args.count("optimizer") != 0) {
        auto opt_type = args["optimizer"].as<std::string>();
        if(opt_type == "SGD") {
            trainer = new SimpleSGDTrainer(lm.model,learning_rate);
        }
        if(opt_type == "Adam") {
            trainer = new dynet::AdamTrainer(lm.model, learning_rate, 0.9, 0.999, 1e-8);
        }
    } else {
        trainer = new dynet::AdamTrainer(lm.model, learning_rate, 0.9, 0.999, 1e-8);
    }
    
    if(args.count("test") == 0) {
        CNLOG << "create language model";

        if(lm_type == "one_hot") {
            train_ngram_onehot(lm,corpus, args,*trainer);
        } else if(lm_type == "cst") {
            train_ngram_onehot(lm,corpus, args,*trainer);
        } else {
            CNLOG << "ERROR: incorrect lm type. options are: one_hot, cst";
            exit(EXIT_FAILURE);
        }

        if( args.count("store") ) {
            auto lm_file_path = args["store"].as<std::string>();
            CNLOG << "store language model to " << lm_file_path;
            lm.store(lm_file_path);
        }
    }

    CNLOG << "test language model";
    auto test_corpus_file = args["path"].as<std::string>() + "/" + constants::TEST_FILE;
    auto pplx = evaluate_pplx(lm, corpus, test_corpus_file);
    CNLOG << "test pplx = " << pplx;

    delete trainer;
    return 0;
}
