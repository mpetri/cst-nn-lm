#pragma once

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <locale>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include <sdsl/int_vector.hpp>

namespace po = boost::program_options;
using args_t = po::variables_map;

namespace constants {
	std::string TRAIN_FILE = "train.txt";
	std::string DEV_FILE = "valid.txt";
	std::string TEST_FILE = "test.txt";
	std::string VOCAB_FILE = "vocab.txt";
}

using str2u32_type = std::unordered_map<std::string,uint32_t>;
using u32tostr_type = std::unordered_map<uint32_t,std::string>;

struct vocab_t {
	size_t max_size;
	str2u32_type tok2int;
	u32tostr_type int2tok;
	std::vector<std::pair<uint32_t,uint32_t>> tok_freqs;

	uint32_t eof_tok;
	uint32_t unk_tok;
	uint32_t start_sent_tok;
	uint32_t stop_sent_tok;

	vocab_t() {
		eof_tok = add_token("<eof>");
		stop_sent_tok = add_token("</s>");
		start_sent_tok = add_token("<s>");
		unk_tok = add_token("<unk>");
	}
	uint32_t add_token(const std::string& tok) {
		auto itr = tok2int.find(tok);
		uint32_t tok_id = tok2int.size();
		if(itr == tok2int.end()) {
			tok2int.insert({tok,tok_id});
			int2tok[tok_id] = tok;
			tok_freqs.emplace_back(tok_id,1);
		} else {
			tok_id = itr->second;
			tok_freqs[tok_id].second++;
		}
		return tok_id;
	}
	void freeze() {
		std::sort(tok_freqs.begin(),tok_freqs.end(),[](const auto& a,const auto&b) {
			return a.second > b.second;
		});
		CNLOG << "\t\tinitial unique tokens = " << tok2int.size();
		tok2int.clear();
		eof_tok = add_token("<eof>");
		stop_sent_tok = add_token("</s>");
		start_sent_tok = add_token("<s>");
		unk_tok = add_token("<unk>");
		for(size_t i=0;i<max_size;i++) {
			auto tok = tok_freqs[i].first;
			auto tok_str = int2tok[tok];
			auto new_tok_id = tok2int.size();
			tok2int.insert({tok_str,new_tok_id});
			if(tok2int.size() == max_size)
				break;
		}
		int2tok.clear();
		for(auto& t : tok2int) {
			auto& tok_str = t.first;
			auto& tok_id = t.second;
			int2tok[tok_id] = tok_str;
		}
		tok_freqs.clear();
		CNLOG << "\t\tfinal unique tokens = " << tok2int.size();
	}
	uint32_t lookup(const std::string& tok) const {
		auto itr = tok2int.find(tok);
		if(itr == tok2int.end()) {
			return unk_tok;
		}
		return itr->second;
	}
	std::string inverse_lookup(uint32_t id) const {
		auto itr = int2tok.find(id);
		if(itr == int2tok.end()) {
			return "<unk>";
		}
		return itr->second;
	}
	size_t size() const {
		return tok2int.size();
	}
};

struct corpus_t {
	std::string file;
	vocab_t vocab;
	std::vector<uint32_t> sent_starts;
	std::vector<uint32_t> sent_lens;
	sdsl::int_vector<32> text;
	size_t num_tokens = 0;
	size_t num_sentences = 0;
	size_t num_oov = 0;
};

struct data_loader {
	static std::vector<std::string>
	tokenize_line(std::string& line)
	{
		boost::algorithm::trim(line);
		std::vector<std::string> toks;
		return boost::algorithm::split(toks,line,boost::is_any_of("\t "),boost::token_compress_on);
	}

	static vocab_t create_or_load_vocab(args_t& args)
	{
		auto path = args["path"].as<std::string>();
		CNLOG << "\tcreate vocabulary";
		auto threshold = args["vocab_size"].as<uint32_t>();
		vocab_t v;
		v.max_size = threshold;
		// auto vocab_file = path + "/" + constants::VOCAB_FILE;
		// if (boost::filesystem::exists(vocab_file)) {
		// 	v.load(vocab_file);
		// 	return v;
		// }

		auto train_file = path + "/" + constants::TRAIN_FILE;
		std::ifstream input(train_file);
		input.imbue( std::locale( "C.UTF-8" ) );
	    for (std::string line; std::getline(input, line); ) {
	    	auto toks = tokenize_line(line);
	    	if(toks.size() < 2) continue;
	    	for(const auto& tok : toks) v.add_token(tok);
		}
		v.freeze();
		return v;
	}

	static void parse_text(corpus_t& corpus)
	{
		CNLOG << "\tparse input text";
		auto train_file = corpus.file;
		std::ifstream input(train_file);
		input.imbue( std::locale( "C.UTF-8" ) );
	    for (std::string line; std::getline(input, line); ) {
	    	auto toks = tokenize_line(line);
	    	if(toks.size() < 2) continue;
			corpus.sent_starts.push_back(corpus.text.size());
			size_t slen = 1;
	    	corpus.text.push_back(corpus.vocab.start_sent_tok);
	    	for(const auto& tok : toks) {
	    		corpus.text.push_back(corpus.vocab.lookup(tok));
	    		if(corpus.text.back() == corpus.vocab.unk_tok) {
	    			corpus.num_oov++;
	    		}
				slen++;
	    	}
	    	corpus.text.push_back(corpus.vocab.stop_sent_tok);
			slen++;
			corpus.sent_lens.push_back(slen);
	    	corpus.num_sentences++;
		}
		corpus.num_tokens = corpus.text.size();
	}

	static corpus_t load(args_t& args) {
		auto directory = args["path"].as<std::string>();
		auto train_file = directory + "/" + constants::TRAIN_FILE;
		corpus_t c;
		c.file = train_file;
		c.vocab = create_or_load_vocab(args);
		parse_text(c);
		CNLOG << "\t\tnum tokens = " << c.num_tokens;
		CNLOG << "\t\tnum sentences = " << c.num_sentences;
		CNLOG << "\t\tnum oov = " << c.num_oov;
		for(size_t i=0;i<100;i++) {
			auto str_tok = c.vocab.inverse_lookup(i);
			CNLOG << "\t\t" << i << " = " << str_tok;
		}
		return c;
	}

	static corpus_t parse_file(const vocab_t& vocab,std::string file)
	{
		corpus_t c;
		c.file = file;
		c.vocab = vocab;
		parse_text(c);
		CNLOG << "\t\tnum tokens = " << c.num_tokens;
		CNLOG << "\t\tnum sentences = " << c.num_sentences;
		CNLOG << "\t\tnum oov = " << c.num_oov;
		return c;
	}
};
