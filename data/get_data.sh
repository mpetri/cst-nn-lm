#!/bin/bash

wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.en.shuffled.gz
gzip -d news.2007.en.shuffled.gz
mkdir -p news.2007
cp news.2007.en.shuffled news.2007/raw.txt
perl ../tools/tokenizer.perl < ./news.2007/raw.txt > ./news.2007/raw.tok
head -n -10000 ./news.2007/raw.tok > ./news.2007/train.txt
tail -n 10000 ./news.2007/raw.tok > ./news.2007/valid_and_test.txt
tail -n 5000 ./news.2007/valid_and_test.txt > ./news.2007/test.txt
head -n 5000 ./news.2007/valid_and_test.txt > ./news.2007/valid.txt

wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.en.shuffled.gz
gzip -d news.2008.en.shuffled.gz
mkdir -p news.2008
cp news.2008.en.shuffled news.2008/raw.txt
perl ../tools/tokenizer.perl < ./news.2008/raw.txt > ./news.2008/raw.tok
head -n -10000 ./news.2008/raw.tok > ./news.2008/train.txt
tail -n 10000 ./news.2008/raw.tok > ./news.2008/valid_and_test.txt
tail -n 5000 ./news.2008/valid_and_test.txt > ./news.2008/test.txt
head -n 5000 ./news.2008/valid_and_test.txt > ./news.2008/valid.txt

wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.en.shuffled.gz
gzip -d news.2009.en.shuffled.gz
mkdir -p news.2009
cp news.2009.en.shuffled news.2009/raw.txt
perl ../tools/tokenizer.perl < ./news.2009/raw.txt > ./news.2009/raw.tok
head -n -10000 ./news.2009/raw.tok > ./news.2009/train.txt
tail -n 10000 ./news.2009/raw.tok > ./news.2009/valid_and_test.txt
tail -n 5000 ./news.2009/valid_and_test.txt > ./news.2009/test.txt
head -n 5000 ./news.2009/valid_and_test.txt > ./news.2009/valid.txt

wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz
gzip -d news.2010.en.shuffled.gz
mkdir -p news.2010
cp news.2010.en.shuffled news.2010/raw.txt
perl ../tools/tokenizer.perl < ./news.2010/raw.txt > ./news.2010/raw.tok
head -n -10000 ./news.2010/raw.tok > ./news.2010/train.txt
tail -n 10000 ./news.2010/raw.tok > ./news.2010/valid_and_test.txt
tail -n 5000 ./news.2010/valid_and_test.txt > ./news.2010/test.txt
head -n 5000 ./news.2010/valid_and_test.txt > ./news.2010/valid.txt

wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.en.shuffled.gz
gzip -d news.2011.en.shuffled.gz
mkdir -p news.2011
cp news.2011.en.shuffled news.2011/raw.txt
perl ../tools/tokenizer.perl < ./news.2011/raw.txt > ./news.2011/raw.tok
head -n -10000 ./news.2011/raw.tok > ./news.2011/train.txt
tail -n 10000 ./news.2011/raw.tok > ./news.2011/valid_and_test.txt
tail -n 5000 ./news.2011/valid_and_test.txt > ./news.2011/test.txt
head -n 5000 ./news.2011/valid_and_test.txt > ./news.2011/valid.txt

mkdir -p news.combined
cat ./news.2007/raw.tok ./news.2008/raw.tok ./news.2009/raw.tok ./news.2010/raw.tok ./news.2011/raw.tok > news.combined/raw.tok
sort --parallel=8 -u ./news.combined/raw.tok > ./news.combined/uniq.tok
sort --parallel=8 -R ./news.combined/uniq.tok > ./news.combined/uniq.tok.shuffled
head -n -10000 ./news.combined/uniq.tok.shuffled > ./news.combined/train.txt
tail -n 10000 ./news.combined/uniq.tok.shuffled > ./news.combined/valid_and_test.txt
tail -n 5000 ./news.combined/valid_and_test.txt > ./news.combined/test.txt
head -n 5000 ./news.combined/valid_and_test.txt > ./news.combined/valid.txt