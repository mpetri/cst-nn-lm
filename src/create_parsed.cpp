#include <chrono>
#include <iostream>

#include <boost/program_options.hpp>

#include "constants.hpp"
#include "data_loader.hpp"

#include "logging.hpp"


namespace po = boost::program_options;

po::variables_map parse_args(int argc, char** argv)
{
    // clang-format off
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("path", po::value<std::string>()->required(), "data path")
        ("opath", po::value<std::string>()->required(), "output data path")
        ("num_sents", po::value<uint32_t>()->default_value(defaults::MAX_NUM_SENTS), "number of sentences to train on. (prefix)")
        ("vocab_size", po::value<uint32_t>()->default_value(defaults::VOCAB_SIZE), "vocab size");
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

    auto dev_corpus_file = args["path"].as<std::string>() + "/" + constants::DEV_FILE;
    auto test_corpus_file = args["path"].as<std::string>() + "/" + constants::TEST_FILE;
    
    auto dev_corpus = data_loader::parse_file(corpus.vocab, dev_corpus_file);
    auto test_corpus = data_loader::parse_file(corpus.vocab, test_corpus_file);

    CNLOG << "store parsed data";
    corpus.store_parsed(args["opath"],dev_corpus,test_corpus);

    return 0;
}
