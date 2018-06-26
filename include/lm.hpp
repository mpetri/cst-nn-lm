#pragma once

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

#include "cst.hpp"
#include "constants.hpp"
#include "data_loader.hpp"
#include "logging.hpp"

using cst_node_type = typename cst_type::node_type;

struct train_instance_t {
	std::vector<uint32_t> prefix;
	std::vector<float> dist;
};

struct language_model {
	dynet::ParameterCollection model;
	uint32_t LAYERS;
	uint32_t INPUT_DIM;
	uint32_t HIDDEN_DIM;
	uint32_t VOCAB_SIZE;
	dynet::LookupParameter p_c;
	dynet::Parameter p_R;
	dynet::Parameter p_bias;
	dynet::Expression i_c;
	dynet::Expression i_R;
	dynet::Expression i_bias;
	dynet::LSTMBuilder rnn;

	language_model(const vocab_t& vocab,args_t& args) {
		LAYERS = args["layers"].as<uint32_t>();
		INPUT_DIM = args["input_dim"].as<uint32_t>();
		HIDDEN_DIM = args["hidden_dim"].as<uint32_t>();
		VOCAB_SIZE = vocab.size();
		CNLOG << "LM parameters ";
		CNLOG << "\tlayers = " << LAYERS;
		CNLOG << "\tinput_dim = " << INPUT_DIM;
		CNLOG << "\thidden_dim = " << HIDDEN_DIM;
		CNLOG << "\tvocab size = " << VOCAB_SIZE;

		// Add embedding parameters to the model
		p_c = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM});
		p_R = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM});
		p_bias = model.add_parameters({VOCAB_SIZE});
		rnn = dynet::LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model);
	}

	dynet::Expression build_train_graph(dynet::ComputationGraph& cg,train_instance_t& instance) {
		auto start = std::chrono::high_resolution_clock::now();
		// Initialize the RNN for a new computation graph
		rnn.new_graph(cg);
		// Prepare for new sequence (essentially set hidden states to 0)
		rnn.start_new_sequence();
		// Instantiate embedding parameters in the computation graph
		// output -> word rep parameters (matrix + bias)
		i_R = dynet::parameter(cg, p_R);
		i_bias = dynet::parameter(cg, p_bias);
		for (size_t i = 0; i < instance.prefix.size()-1; ++i) {
			dynet::Expression i_x_t = dynet::lookup(cg, p_c, instance.prefix[i]);
			dynet::Expression i_y_t = rnn.add_input(i_x_t);
		}
		dynet::Expression i_x_t = dynet::lookup(cg, p_c, instance.prefix.back());
		dynet::Expression i_y_t = rnn.add_input(i_x_t);
		dynet::Expression i_r_t = i_bias + i_R * i_y_t;
		dynet::Expression i_true = dynet::input(cg, {(unsigned int)instance.dist.size()}, instance.dist);
		dynet::Expression i_error = dynet::transpose(i_true) * dynet::log_softmax(i_r_t);
		auto end = std::chrono::high_resolution_clock::now();
		CNLOG << "BUILD TRAIN GRAPH " << " - " << (end-start).count() << "s";
		return i_error;
	}
};

struct pq_node_type {
	std::vector<uint32_t> prefix;
	cst_node_type cst_node;
	size_t priority;
	bool operator<(const pq_node_type& other) const {
		return priority < other.priority;
	}
};

using pq_type = std::priority_queue<pq_node_type>;

std::string
print_pq_node(pq_node_type& node,const vocab_t& vocab,const cst_type& cst)
{
	std::string node_str = "<";
	node_str += "prio=" + std::to_string(node.priority) + ",";
	node_str += "ids=[";
	for(size_t i=0;i<node.prefix.size()-1;i++)
		node_str += std::to_string(node.prefix[i]) + ",";
	node_str += std::to_string(node.prefix.back()) + "],";
	node_str += "toks=[";
	for(size_t i=0;i<node.prefix.size()-1;i++)
		node_str += vocab.inverse_lookup(node.prefix[i]) + ",";
	node_str += vocab.inverse_lookup(node.prefix.back()) + "]>";
	return node_str;
}

void
add_node(const vocab_t& vocab,const cst_type& cst,pq_type& pq,const pq_node_type& parent,const cst_node_type& cur_node,uint32_t tok)
{
	// no need to explore <eof> </s>
	if(tok < vocab.start_sent_tok) return;

	pq_node_type new_node = parent;
	new_node.prefix.push_back(tok);
	new_node.cst_node = cur_node;
	new_node.priority = cst.size(cur_node);
	if(cst.is_leaf(cur_node)) {
		// we have to finish the sentence here
		auto depth = cst.depth(parent.cst_node) + 2;
		while(true) {
			tok = cst.edge(cur_node,depth);
			if(tok == vocab.stop_sent_tok) break;
			new_node.prefix.push_back(tok);
			depth++;
		}
	}
	pq.push(new_node);
}


train_instance_t
create_instance(const cst_type& cst,pq_type& pq,const vocab_t& vocab)
{
	auto start = std::chrono::high_resolution_clock::now();
	auto top_node = pq.top(); pq.pop();
	train_instance_t new_instance;
	new_instance.dist.resize(vocab.size());
	new_instance.prefix = top_node.prefix;

	double node_size = cst.size(top_node.cst_node);

	if(node_size == 1) {
		new_instance.dist[vocab.stop_sent_tok] = 1;
	} else {
		auto parent_depth = cst.depth(top_node.cst_node);
		for(const auto& child : cst.children(top_node.cst_node)) {
			auto tok = cst.edge(child,parent_depth+1);
			double size = cst.size(child);
			new_instance.dist[tok] = size/node_size;
			if(tok != vocab.start_sent_tok)
				add_node(vocab,cst,pq,top_node,child,tok);
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end-start;
	CNLOG << "PROCESS NODE " << print_pq_node(top_node,vocab,cst) 
		  << " - " << (end-start).count() << "s";
	return new_instance;
}

language_model create_lm(const cst_type& cst,const vocab_t& vocab,args_t& args)
{
	auto num_epochs = args["epochs"].as<uint32_t>();
	auto batch_size = args["batch_size"].as<uint32_t>();
	language_model lm(vocab,args);

	dynet::AdamTrainer trainer(lm.model, 0.001, 0.9, 0.999, 1e-8);
	trainer.clip_threshold *= batch_size;

	for(size_t epoch = 1;epoch<=num_epochs;epoch++) {
		CNLOG << "start epoch " << epoch;

		// (1) create the starting nodes
		std::priority_queue<pq_node_type> pq;
		pq_node_type root;
		root.cst_node = cst.root();
		for(size_t i=vocab.start_sent_tok;i<vocab.size();i++) {
			size_t lb = cst.csa.C[i];
			size_t rb = cst.csa.C[i+1] - 1;
			auto cst_node = cst.node(lb,rb);
			add_node(vocab,cst,pq,root,cst_node,i);
		}

		std::vector<dynet::Expression> errors(batch_size);
		dynet::ComputationGraph cg;
		size_t cur = 0;
		while(!pq.empty()) {
			auto instance = create_instance(cst,pq,vocab);
			errors[cur++] = lm.build_train_graph(cg,instance);

			if(cur == batch_size) {
				auto start = std::chrono::high_resolution_clock::now();
				auto loss_expr = dynet::sum(errors);
				cg.backward(loss_expr);
       			trainer.update();
				cur = 0;
				auto end = std::chrono::high_resolution_clock::now();
				CNLOG << "BACKWARD/UPDATE " << " - " << (end-start).count() << "s";
			}
		}

	}

	return lm;
}
