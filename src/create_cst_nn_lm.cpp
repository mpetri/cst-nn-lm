#include <iostream>
#include <chrono>

#include <boost/program_options.hpp>

#include "logging.hpp"
#include "constants.hpp"
#include "data_loader.hpp"
#include "cst.hpp"
#include "lm.hpp"

namespace po = boost::program_options;

po::variables_map parse_args(int argc,char** argv)
{
	po::options_description desc("Allowed options");
	desc.add_options()
	    ("help", "produce help message")
	    ("path", po::value<std::string>()->required(), "data path")
	    ("vocab_size", po::value<uint32_t>()->default_value(defaults::VOCAB_SIZE), "vocab size")
		("layers", po::value<uint32_t>()->default_value(defaults::LAYERS), "layers of the rnn")
		("input_dim", po::value<uint32_t>()->default_value(defaults::INPUT_DIM), "input embedding size")
		("hidden_dim", po::value<uint32_t>()->default_value(defaults::HIDDEN_DIM), "hidden size")
	    ("epochs", po::value<uint32_t>()->default_value(defaults::EPOCHS), "num epochs")
	    ("batch_size", po::value<uint32_t>()->default_value(defaults::BATCH_SIZE), "batch size")
	;

	po::variables_map args;
	try {
		po::store(po::parse_command_line(argc, argv, desc), args);
		po::notify(args);
	} catch(std::exception& e) {
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


int main(int argc,char** argv)
{
	init_logging();

	CNLOG << "parse arguments";
	auto args = parse_args(argc,argv);

	CNLOG << "load and parse data";
	auto corpus = data_loader::load(args);

	CNLOG << "build or load CST";
	auto cst = build_or_load_cst(corpus,args);

	CNLOG << "create language model";
	dynet::initialize(argc, argv);
	auto lm = create_lm(cst,corpus.vocab,args);

	return 0;
}

