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
        ("out_path", po::value<std::string>()->required(), "output data path")
	;

	po::variables_map args;
	try {
		po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), args);
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

void
write_sentences(corpus_t& corpus,std::string directory,std::string file_name)
{
    std::string output_file = directory + "/" + file_name;
    std::ofstream ofs(output_file);
    for(size_t i=0;i<corpus.num_sentences;i++) {
		auto start_sent = corpus.text.begin() + corpus.sent_starts[i];
		auto sent_len = corpus.sent_lens[i];
        for(size_t j=0;j<sent_len-1;j++) {
            auto tok_id = *(start_sent + j);
            auto str_tok = corpus.vocab.inverse_lookup(tok_id);
            ofs << str_tok << " ";
        }
        auto tok_id = *(start_sent + sent_len - 1);
        auto str_tok = corpus.vocab.inverse_lookup(tok_id);
        ofs << str_tok << std::endl;
    }
}

int main(int argc,char** argv)
{
	init_logging();

	CNLOG << "parse arguments";
	auto args = parse_args(argc,argv);
    auto out_directory = args["out_path"].as<std::string>();
	if(boost::filesystem::create_directory(out_directory))
	{
		CNLOG<< "created output directory: "<< out_directory;
	}

	CNLOG << "load and parse train";
	auto corpus = data_loader::load(args);

    write_sentences(corpus,out_directory,constants::TRAIN_FILE);


	CNLOG << "load and parse dev";
    auto dev_corpus_file = args["path"].as<std::string>() + "/" + constants::DEV_FILE;
	auto dev = data_loader::parse_file(corpus.vocab,dev_corpus_file);
    write_sentences(dev,out_directory,constants::DEV_FILE);

	CNLOG << "load and parse test";
    auto test_corpus_file = args["path"].as<std::string>() + "/" + constants::TEST_FILE;
	auto test = data_loader::parse_file(corpus.vocab,test_corpus_file);
    write_sentences(test,out_directory,constants::TEST_FILE);

	return 0;
}

