#pragma once

namespace defaults {
unsigned int VOCAB_SIZE = 30000;
unsigned int HIDDEN_DIM = 1024;
unsigned int LAYERS = 2;
unsigned int INPUT_DIM = 256;
unsigned int MAX_NUM_SENTS = 0;
unsigned int NGRAM_SIZE = 5;
size_t EPOCHS = 20;
size_t BATCH_SIZE = 32;
size_t THREADS = 16;
size_t THRESHOLD = 10;
size_t REPORT_INTERVAL = 8192;
double DROP_OUT = 0.5;
}

namespace constants {
size_t MAX_SENTENCE_LEN = 128;
size_t RAND_SEED = 12345;
}
