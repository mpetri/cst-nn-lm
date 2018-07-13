#include <cassert>
#include <chrono>
#include <iostream>

#include <boost/program_options.hpp>

#include "constants.hpp"
#include "data_loader.hpp"
#include "dynet_rnnlm_batch.hpp"
#include "logging.hpp"

namespace po = boost::program_options;

po::variables_map parse_args(int argc, char** argv)
{
    // clang-format off
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("path", po::value<std::string>()->required(), "data path")
        ("vocab_size", po::value<uint32_t>()->default_value(defaults::VOCAB_SIZE), "vocab size")
        ("layers", po::value<uint32_t>()->default_value(defaults::LAYERS), "layers of the rnn")
        ("input_dim", po::value<uint32_t>()->default_value(defaults::INPUT_DIM), "input embedding size")
        ("hidden_dim", po::value<uint32_t>()->default_value(defaults::HIDDEN_DIM), "hidden size")
        ("epochs", po::value<size_t>()->default_value(defaults::EPOCHS), "num epochs")
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

    CNLOG << "create dynet rnn language model";
    dynet::initialize(argc, argv);
    auto lm = create_dynet_rnn_lm(corpus, args);

    CNLOG << "test language model";
    auto test_corpus_file = args["path"].as<std::string>() + "/" + constants::TEST_FILE;
    auto pplx = evaluate_pplx(lm, corpus.vocab, test_corpus_file);
    CNLOG << "test pplx = " << pplx;

    return 0;
}
