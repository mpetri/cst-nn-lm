#pragma once

#include <future>

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
	size_t num_occ;
	cst_node_type cst_node;
	size_t num_children;
	std::vector<uint32_t> prefix;
	// std::vector<float> dist;
	bool operator<(const train_instance_t& other) const {
		if(prefix.size() == other.prefix.size()) {
			return num_occ > other.num_occ;
		}
		return prefix.size() < other.prefix.size();
	}
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

	template<class t_itr>
	dynet::Expression build_train_graph_batch(dynet::ComputationGraph& cg,t_itr& start,t_itr& end,
		std::vector<float>& dists,size_t dist_len
	) {
		size_t batch_size = std::distance(start,end);
		size_t prefix_len = start->prefix.size();
		
		// Initialize the RNN for a new computation graph
		rnn.new_graph(cg);
		// Prepare for new sequence (essentially set hidden states to 0)
		rnn.start_new_sequence();
		// Instantiate embedding parameters in the computation graph
		// output -> word rep parameters (matrix + bias)
		i_R = dynet::parameter(cg, p_R);
		i_bias = dynet::parameter(cg, p_bias);

		std::vector<uint32_t> cur_sym(batch_size);
		for (size_t i = 0; i < prefix_len-1; ++i) {
			for(size_t j=0;j<batch_size;j++) {
				auto instance = start + j;
				cur_sym[j] = instance->prefix[i];
			}
			dynet::Expression i_x_t = dynet::lookup(cg, p_c,cur_sym);
			rnn.add_input(i_x_t);
		}
		auto last_instance = start + batch_size - 1;
		for(size_t j=0;j<batch_size;j++) cur_sym[j] = last_instance->prefix.back();
		dynet::Expression i_x_t = dynet::lookup(cg, p_c,cur_sym);
		dynet::Expression i_y_t = rnn.add_input(i_x_t);
		dynet::Expression i_r_t = i_bias + i_R * i_y_t;

		dynet::Expression i_pred = dynet::log_softmax(i_r_t);
		dynet::Expression i_pred_linear = dynet::reshape(i_pred,{(unsigned int)dist_len});
		dynet::Expression i_true = dynet::input(cg, {(unsigned int)dist_len},dists);
		dynet::Expression i_error = dynet::transpose(i_true) * i_pred_linear;
		return i_error;
	}
};

std::string
print_prefix(std::vector<uint32_t>& prefix,const vocab_t& vocab)
{
	std::string s = "[";
	for(size_t i=0;i<prefix.size()-1;i++) {
		s += vocab.inverse_lookup(prefix[i]) + " ";
	}
	return s + vocab.inverse_lookup(prefix.back()) + "]";
}

std::vector<train_instance_t>
create_instances(const cst_type& cst,const vocab_t& vocab,cst_node_type cst_node,std::vector<uint32_t> prefix,size_t threshold)
{
	CNLOG << "START create_instances for subtree " << print_prefix(prefix,vocab);
	std::vector<train_instance_t> instances;
	if(prefix.back() < vocab.start_sent_tok) return instances;

	double node_size = cst.size(cst_node);
	CNLOG << "\tNODE SIZE = "   << (size_t)node_size;
	if(node_size >= threshold) {
		train_instance_t new_instance;
		new_instance.num_occ = node_size;
		// new_instance.dist.resize(vocab.size());
		new_instance.prefix = prefix;
		new_instance.cst_node = cst_node;
		auto node_depth = cst.depth(cst_node);
		for(const auto& child : cst.children(cst_node)) {
			auto tok = cst.edge(child,node_depth+1);
			double size = cst.size(child);
			// new_instance.dist[tok] = size/node_size;
			if(tok != vocab.start_sent_tok && tok != vocab.stop_sent_tok && size >= threshold) {
				auto child_prefix = prefix;
				child_prefix.push_back(tok);
				auto child_instances = create_instances(cst,vocab,child,child_prefix,threshold);
				instances.insert(instances.end(),child_instances.begin(),child_instances.end());
			}
			new_instance.num_children++;
		}
		instances.push_back(new_instance);
	}
	CNLOG << "STOP create_instances for subtree " << print_prefix(prefix,vocab);
	return instances;
}

template<class t_dist_itr>
void create_dist(const cst_type& cst,const train_instance_t& instance,t_dist_itr dist_itr)
{
	double node_size = instance.num_occ;
	auto node_depth = cst.depth(instance.cst_node);
	for(const auto& child : cst.children(instance.cst_node)) {
		auto tok = cst.edge(child,node_depth+1);
		double size = cst.size(child);
		*(dist_itr + tok) = size/node_size;
	}
}

std::vector<train_instance_t>
process_token_subtree(const cst_type& cst,const vocab_t& vocab,size_t start,size_t step,size_t threshold)
{
	std::vector<train_instance_t> instances;
	for(size_t j=start;j<vocab.size();j+=step) {
		size_t lb = cst.csa.C[j];
		size_t rb = cst.csa.C[j+1] - 1;
		if(rb-lb+1 >= threshold) {
			auto cst_node = cst.node(lb,rb);
			std::vector<uint32_t> prefix(1,j);
			auto subtree_instances = create_instances(cst,vocab,cst_node,prefix,threshold);
			instances.insert(instances.end(),subtree_instances.begin(),subtree_instances.end());
		}
	}
	return instances;
}

language_model create_lm(const cst_type& cst,const vocab_t& vocab,args_t& args)
{
	auto num_epochs = args["epochs"].as<size_t>();
	auto batch_size = args["batch_size"].as<size_t>();
	auto threads = args["threads"].as<size_t>();
	auto threshold = args["threshold"].as<size_t>();
	language_model lm(vocab,args);

	dynet::AdamTrainer trainer(lm.model, 0.001, 0.9, 0.999, 1e-8);
	trainer.clip_threshold *= batch_size;

	for(size_t epoch = 1;epoch<=num_epochs;epoch++) {
		CNLOG << "start epoch " << epoch;

		// (1) explore the CST a bit as a start
		CNLOG << "explore CST and create instances ";
		auto prep_start = std::chrono::high_resolution_clock::now();
		std::vector<train_instance_t> instances;
		std::vector<std::future<std::vector<train_instance_t>>> results;
		for(size_t thread=0;thread<threads;thread++) {
			size_t start = vocab.start_sent_tok + thread;
			results.push_back(std::async(std::launch::async,process_token_subtree,cst,vocab,start,threads,threshold));
		}
		for(auto& e : results) {
			auto res = e.get();
			instances.insert(instances.end(),res.begin(),res.end());
		}
		auto prep_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> prep_diff = prep_end-prep_start;
		CNLOG << "CREATE EPOCH INSTANCES " << " - " << prep_diff.count() << "s";

		std::sort(instances.begin(),instances.end());
		std::vector<float> dists(batch_size*vocab.size());
		
		auto itr = instances.begin();
		auto end = instances.end();
		while(itr != end) {
			// (1) ensure we have same length in batch
			auto batch_end = itr + batch_size;
			if(batch_end > end) {
				batch_end = end;
			}
			auto last = batch_end - 1;
			while(last->prefix.size() != itr->prefix.size()) {
				last--;
				batch_end = last + 1;
			}

			// (2) create the dists and store into one long vector 
			auto tmp = itr;
			auto dist_itr = dists.begin();
			size_t dist_len = 0;
			while(tmp != batch_end) {
				create_dist(cst,*tmp,dist_itr);
				dist_itr += vocab.size();
				dist_len += vocab.size();
				++tmp;
			}

			dynet::ComputationGraph cg;
			auto train_start = std::chrono::high_resolution_clock::now();
			auto loss_expr = lm.build_train_graph_batch(cg,itr,batch_end,dists,dist_len);
			cg.backward(loss_expr);
			trainer.update();
			auto train_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> train_diff = train_end-train_start;
			CNLOG << "BACKWARD/UPDATE " << " - " << train_diff.count() << "s";

			itr = batch_end;
		}
	}

	return lm;
}
