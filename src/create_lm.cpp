#include <chrono>
#include <iostream>

#include <boost/program_options.hpp>

#include "constants.hpp"
#include "cst.hpp"
#include "data_loader.hpp"

#include "logging.hpp"

#include "lm_dynet.hpp"
// #include "lm_cst_sent.hpp"

namespace po = boost::program_options;

po::variables_map parse_args(int argc, char** argv)
{
    // clang-format off
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("path", po::value<std::string>()->required(), "data path")
        ("type", po::value<std::string>()->required(), "lm type")
        ("vocab_size", po::value<uint32_t>()->default_value(defaults::VOCAB_SIZE), "vocab size")
        ("layers", po::value<uint32_t>()->default_value(defaults::LAYERS), "layers of the rnn")
        ("input_dim", po::value<uint32_t>()->default_value(defaults::INPUT_DIM), "input embedding size")
        ("hidden_dim", po::value<uint32_t>()->default_value(defaults::HIDDEN_DIM), "hidden size")
        ("threads", po::value<size_t>()->default_value(defaults::THREADS), "threads")
        ("epochs", po::value<size_t>()->default_value(defaults::EPOCHS), "num epochs")
        ("epoch_size", po::value<size_t>()->default_value(0), "epoch size")
        ("drop_out", po::value<double>()->default_value(defaults::DROP_OUT), "drop out rate")
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
    return args;
}

int main(int argc, char** argv)
{
    init_logging();

    CNLOG << "parse arguments";
    auto args = parse_args(argc, argv);

    CNLOG << "load and parse data";
    auto corpus = data_loader::load(args);

    CNLOG << "create language model";
    dynet::initialize(argc, argv);

    auto lm_type = args["type"].as<std::string>();
    language_model lm(corpus.vocab,args);
    if(lm_type == "standard") {
        train_dynet_lm(lm,corpus, args);
    } else if(lm_type == "cst_sent") {
        // train_cst_sent(lm,corpus, args);
    } else if(lm_type == "cst_sample") {
        // train_cst_lm(lm,corpus, args);
    } else {
        CNLOG << "ERROR: incorrect lm type. options are: dynet, cst_sent, cst_sample";
        exit(EXIT_FAILURE);
    }

    CNLOG << "test language model";
    auto test_corpus_file = args["path"].as<std::string>() + "/" + constants::TEST_FILE;
    auto pplx = evaluate_pplx(lm, corpus, test_corpus_file);
    CNLOG << "test pplx = " << pplx;

    return 0;
}
