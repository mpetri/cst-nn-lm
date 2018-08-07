#pragma once

#include <fstream>
#include <iostream>
#include <locale>
#include <unordered_map>
#include <unordered_set>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <sdsl/int_vector.hpp>

#include "constants.hpp"
#include "logging.hpp"

namespace po = boost::program_options;
using args_t = po::variables_map;

namespace constants {
std::string PARSED_TRAIN_FILE = "train-parsed.txt";
std::string TRAIN_FILE = "train.txt";
std::string DEV_FILE = "valid.txt";
std::string TEST_FILE = "test.txt";
std::string VOCAB_FILE = "vocab.txt";
}

using str2u32_type = std::unordered_map<std::string, uint32_t>;
using u32tostr_type = std::unordered_map<uint32_t, std::string>;

struct vocab_t {
    size_t max_size;
    str2u32_type tok2int;
    u32tostr_type int2tok;
    std::vector<std::pair<uint32_t, uint32_t> > tok_freqs;

    uint32_t eof_tok;
    uint32_t unk_tok;
    uint32_t start_sent_tok;
    uint32_t stop_sent_tok;
    uint32_t padding_tok;

    vocab_t()
    {
        eof_tok = add_token("<eof>");
        stop_sent_tok = add_token("</s>");
        start_sent_tok = add_token("<s>");
        unk_tok = add_token("<unk>");
        padding_tok = eof_tok;
    }

    void load(std::string file_name) {
        CNLOG << "load vocab from file " << file_name;
        std::ifstream ifs(file_name);
        for (std::string tok; std::getline(ifs, tok); ) {
            size_t id = int2tok.size();
            tok2int[tok] = id;
            int2tok[id] = tok;
        }
    }

    void store(std::string file_name) {
        CNLOG << "store vocab to file " << file_name;
        std::ofstream ofs(file_name);
        for(size_t i=4;i<int2tok.size();i++) {
            ofs << int2tok[i] << std::endl;
        }
    }

    uint32_t add_token(const std::string& tok)
    {
        auto itr = tok2int.find(tok);
        uint32_t tok_id = tok2int.size();
        if (itr == tok2int.end()) {
            tok2int.insert({ tok, tok_id });
            int2tok[tok_id] = tok;
            tok_freqs.emplace_back(tok_id, 1);
        } else {
            tok_id = itr->second;
            tok_freqs[tok_id].second++;
        }
        return tok_id;
    }
    void freeze()
    {
        std::sort(tok_freqs.begin(), tok_freqs.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        CNLOG << "\t\tinitial unique tokens = " << tok2int.size();
        tok2int.clear();
        eof_tok = add_token("<eof>");
        stop_sent_tok = add_token("</s>");
        start_sent_tok = add_token("<s>");
        unk_tok = add_token("<unk>");
        padding_tok = eof_tok;
        for (size_t i = 0; i < max_size; i++) {
            auto tok = tok_freqs[i].first;
            auto tok_str = int2tok[tok];
            auto new_tok_id = tok2int.size();
            if(tok2int.find(tok_str) == tok2int.end()) {
                tok2int.insert({ tok_str, new_tok_id });
            }
            if (tok2int.size() == max_size)
                break;
        }
        int2tok.clear();
        for (auto& t : tok2int) {
            auto& tok_str = t.first;
            auto& tok_id = t.second;
            int2tok[tok_id] = tok_str;
        }
        tok_freqs.clear();
        CNLOG << "\t\tfinal unique tokens = " << tok2int.size();
    }
    uint32_t lookup(const std::string& tok) const
    {
        auto itr = tok2int.find(tok);
        if (itr == tok2int.end()) {
            return unk_tok;
        }
        return itr->second;
    }
    std::string inverse_lookup(uint32_t id) const
    {
        auto itr = int2tok.find(id);
        if (itr == int2tok.end()) {
            return "<unk>";
        }
        return itr->second;
    }
    size_t size() const
    {
        return tok2int.size();
    }

    std::string print_sentence(std::vector<uint32_t>& sent) const {
        std::string s = "<";
        for(size_t i=0;i<sent.size();i++) {
            s += "[" + std::to_string(sent[i]) + ",'" + inverse_lookup(sent[i]) + "']";
        }
        return s + ">";
    }

};

struct corpus_t {
    std::stirng path;
    std::string file;
    vocab_t vocab;
    sdsl::int_vector<32> sent_starts;
    sdsl::int_vector<32> sent_lens;
    sdsl::int_vector<32> text;
    uint64_t num_tokens = 0;
    uint64_t num_sentences = 0;
    uint64_t num_oov = 0;
    uint64_t num_duplicates = 0;

    void store(std::string file_name) {
        CNLOG << "store parsed corpus to file " << file_name;
        std::ofstream ofs(file_name);
        sdsl::serialize(num_tokens,ofs);
        sdsl::serialize(num_sentences,ofs);
        sdsl::serialize(num_oov,ofs);
        sdsl::serialize(num_duplicates,ofs);
        sdsl::serialize(sent_starts,ofs);
        sdsl::serialize(sent_lens,ofs);
        sdsl::serialize(text,ofs);
    }

    void load(std::string file_name) {
        CNLOG << "load parsed corpus from file " << file_name;
        std::ifstream ifs(file_name);
        sdsl::load(num_tokens,ifs);
        sdsl::load(num_sentences,ifs);
        sdsl::load(num_oov,ifs);
        sdsl::load(num_duplicates,ifs);
        sdsl::load(sent_starts,ifs);
        sdsl::load(sent_lens,ifs);
        sdsl::load(text,ifs);
    }

    void store_parsed(std::string output_dir) {
        if (!boost::filesystem::is_directory(output_dir) || !boost::filesystem::exists(output_dir)) {
            boost::filesystem::create_directory(output_dir); 
        }

        {
            std::ofstream train_out(output_dir+"/train.txt");
            for(size_t i=0;i<num_sentences;i++) {
                auto start = sent_starts[i];
                auto len = sent_lens[i];
                for(size_t j=0;j<len;j++) {
                    train_out << vocab.inverse_lookup(text[start+i]);
                }
                train_out << "\n";
            }
        }

        {
            auto valid_file = path + "/valid.txt";
            auto valid_corpus = data_loader::parse_file(corpus.vocab, valid_file);
            std::ofstream valid_out(output_dir+"/valid.txt");
            for(size_t i=0;i<valid_corpus.num_sentences;i++) {
                auto start = valid_corpus.sent_starts[i];
                auto len = valid_corpus.sent_lens[i];
                for(size_t j=0;j<len;j++) {
                    valid_out << vocab.inverse_lookup(valid_corpus.text[start+i]);
                }
                valid_out << "\n";
            }
        }

        {
            auto test_file = path + "/test.txt";
            auto test_corpus = data_loader::parse_file(corpus.vocab, test_file);
            std::ofstream test_out(output_dir+"/test.txt");
            for(size_t i=0;i<test_corpus.num_sentences;i++) {
                auto start = test_corpus.sent_starts[i];
                auto len = test_corpus.sent_lens[i];
                for(size_t j=0;j<len;j++) {
                    test_out << vocab.inverse_lookup(test_corpus.text[start+i]);
                }
                test_out << "\n";
            }
        }
    }
};

struct data_loader {
    static std::vector<std::string>
    tokenize_line(std::string& line)
    {
        boost::algorithm::trim(line);
        std::vector<std::string> toks;
        return boost::algorithm::split(toks, line, boost::is_any_of("\t "), boost::token_compress_on);
    }

    static vocab_t create_or_load_vocab(args_t& args)
    {
        auto threshold = args["vocab_size"].as<uint32_t>();
        auto path = args["path"].as<std::string>();
        auto train_file = path + "/" + constants::TRAIN_FILE;
        auto vocab_file = path + "/" + constants::VOCAB_FILE + "-" + std::to_string(threshold);

        if (boost::filesystem::exists(vocab_file)) {
            vocab_t v;
            v.load(vocab_file);
            return v;
        }

        CNLOG << "\tcreate vocabulary with threshold = " << threshold << " from " << train_file;
        vocab_t v;
        v.max_size = threshold;

        std::ifstream input(train_file);
        input.imbue(std::locale("en_US.UTF-8"));
        for (std::string line; std::getline(input, line);) {
            auto toks = tokenize_line(line);
            if(toks.size() > constants::MAX_SENTENCE_LEN) continue;
            for (const auto& tok : toks)
                v.add_token(tok);
        }
        v.freeze();
        v.store(vocab_file);
        return v;
    }

    static uint64_t hash_sentence(std::vector<uint32_t>& sent) {
        uint64_t hash = sent.size();
        for(auto& tok : sent) {
            hash ^= tok + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }

    static void parse_text(corpus_t& corpus)
    {
        CNLOG << "\tparse text " << corpus.file;
        auto train_file = corpus.file;
        std::ifstream input(train_file);
        input.imbue(std::locale("en_US.UTF-8"));

        std::unordered_set<uint64_t> sentence_filter;

        for (std::string line; std::getline(input, line);) {
            auto toks = tokenize_line(line);
            if(toks.size() > constants::MAX_SENTENCE_LEN) continue;

            std::vector<uint32_t> sentence;
            sentence.push_back(corpus.vocab.start_sent_tok);
            size_t num_oov = 0;
            for (const auto& tok : toks) {
                sentence.push_back(corpus.vocab.lookup(tok));
                if (sentence.back() == corpus.vocab.unk_tok) {
                    num_oov++;
                }
            }
            sentence.push_back(corpus.vocab.stop_sent_tok);
            auto shash = hash_sentence(sentence);
            if(sentence_filter.find(shash) == sentence_filter.end()) {
                sentence_filter.insert(shash);
                corpus.sent_starts.push_back(corpus.text.size());
                for(auto& tok : sentence) {
                    corpus.text.push_back(tok);
                }
                corpus.num_oov += num_oov;
                corpus.num_sentences++;
                corpus.sent_lens.push_back(sentence.size());
            } else {
                corpus.num_duplicates++;
            }
        }
        corpus.num_tokens = corpus.text.size();
    }

    static corpus_t load(args_t& args)
    {
        auto vocab_threshold = args["vocab_size"].as<uint32_t>();
        auto directory = args["path"].as<std::string>();
        auto train_file = directory + "/" + constants::TRAIN_FILE;
        auto parsed_file = directory + "/" + constants::PARSED_TRAIN_FILE + "-" + std::to_string(vocab_threshold);
        auto max_num_sents = args["num_sents"].as<uint32_t>();

        corpus_t c;
        c.path = directory;
        c.file = train_file;
        c.vocab = create_or_load_vocab(args);

        if (boost::filesystem::exists(parsed_file)) {
            c.load(parsed_file);
        } else {
            parse_text(c);
            c.store(parsed_file);
        }

        if(max_num_sents != 0 && max_num_sents < c.num_sentences) {
            CNLOG << "\t\tactual sentences = " << c.num_sentences;
            CNLOG << "\t\tcapping num sentences to the first " << max_num_sents;
            c.num_sentences = max_num_sents;
        }

        CNLOG << "\t\tnum tokens = " << c.num_tokens;
        CNLOG << "\t\tnum sentences = " << c.num_sentences;
        CNLOG << "\t\tnum duplicate sentences = " << c.num_duplicates;
        CNLOG << "\t\tnum oov = " << c.num_oov << " ("
              << std::fixed << std::setprecision(1)
              << double(c.num_oov * 100) / double(c.num_tokens) << "%)";
        return c;
    }

    static corpus_t parse_file(const vocab_t& vocab, std::string file)
    {
        corpus_t c;
        c.file = file;
        c.vocab = vocab;
        parse_text(c);
        CNLOG << "\t\tnum tokens = " << c.num_tokens;
        CNLOG << "\t\tnum sentences = " << c.num_sentences;
        CNLOG << "\t\tnum duplicate sentences = " << c.num_duplicates;
        CNLOG << "\t\tnum oov = " << c.num_oov << " ("
              << std::fixed << std::setprecision(1)
              << double(c.num_oov * 100) / double(c.num_tokens) << "%)";
        return c;
    }
};
