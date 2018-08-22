#include <chrono>
#include <iostream>

#include <boost/program_options.hpp>

#include "constants.hpp"
#include "data_loader.hpp"

#include "logging.hpp"

#include "lm_types.hpp"

namespace po = boost::program_options;

po::variables_map parse_args(int argc, char** argv)
{
    // clang-format off
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("path", po::value<std::string>()->required(), "data path")
        ("num_sents", po::value<uint32_t>()->default_value(defaults::MAX_NUM_SENTS), "number of sentences to train on. (prefix)")
        ("file", po::value<std::string>(), "load/store model if it exists")
        ("type", po::value<std::string>()->required(), "lm type")
        ("lr", po::value<double>()->default_value(defaults::LEARNING_RATE), "learning rate")
        ("test", "test only. no train.")
        ("vocab_size", po::value<uint32_t>()->default_value(defaults::VOCAB_SIZE), "vocab size")
        ("layers", po::value<uint32_t>()->default_value(defaults::LAYERS), "layers of the rnn")
        ("input_dim", po::value<uint32_t>()->default_value(defaults::INPUT_DIM), "input embedding size")
        ("hidden_dim", po::value<uint32_t>()->default_value(defaults::HIDDEN_DIM), "hidden size")
        ("epochs", po::value<size_t>()->default_value(defaults::EPOCHS), "num epochs")
        ("drop_out", po::value<double>()->default_value(defaults::DROP_OUT), "drop out rate")
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
    dynet::DynetParams params = dynet::extract_dynet_params(argc, argv);
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
    language_model lm(corpus.vocab,args);

    if( args.count("file") ) {
        auto lm_file_path = args["file"].as<std::string>();
        CNLOG << "load language model from " << lm_file_path;
        lm.load(lm_file_path);
    }

    double learning_rate = args["lr"].as<double>();
    dynet::AdamTrainer trainer(lm.model, learning_rate, 0.9, 0.999, 1e-8);

    if(args.count("test") == 0) {
        CNLOG << "train language model";
        if(lm_type == "one_hot") {
            train_one_hot(lm,corpus, args,trainer);
        } else {
            CNLOG << "ERROR: incorrect lm type. options are: one_hot, ";
            exit(EXIT_FAILURE);
        }
    }

    CNLOG << "test language model";
    auto test_corpus_file = args["path"].as<std::string>() + "/" + constants::TEST_FILE;
    auto pplx = evaluate_pplx(lm, corpus, test_corpus_file);
    CNLOG << "test pplx = " << pplx;

    return EXIT_SUCCESS;
}
