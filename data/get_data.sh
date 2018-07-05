#!/bin/bash

# wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
# unzip wikitext-103-v1.zip
# perl ../tools/split-sentences.perl < wikitext-103/wiki.train.tokens > wikitext-103/raw.txt
# perl ../tools/tokenizer.perl < ./wikitext-103/raw.txt > ./news.2007/train.txt

wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.en.shuffled.gz
gzip -d news.2007.en.shuffled.gz
mkdir -p news.2007
mv news.2007.en.shuffled news.2007/raw.txt
perl ../tools/tokenizer.perl < ./news.2007/raw.txt > ./news.2007/train.txt

wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.en.shuffled.gz
gzip -d news.2008.en.shuffled.gz
mkdir -p news.2008
mv news.2008.en.shuffled news.2008/raw.txt
perl ../tools/tokenizer.perl < ./news.2008/raw.txt > ./news.2008/train.txt

wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.en.shuffled.gz
gzip -d news.2009.en.shuffled.gz
mkdir -p news.2009
mv news.2009.en.shuffled news.2009/raw.txt
perl ../tools/tokenizer.perl < ./news.2009/raw.txt > ./news.2009/train.txt

wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz
gzip -d news.2010.en.shuffled.gz
mkdir -p news.2010
mv news.2010.en.shuffled news.2010/raw.txt
perl ../tools/tokenizer.perl < ./news.2010/raw.txt > ./news.2010/train.txt

wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.en.shuffled.gz
gzip -d news.2011.en.shuffled.gz
mkdir -p news.2011
mv news.2011.en.shuffled news.2011/raw.txt
perl ../tools/tokenizer.perl < ./news.2011/raw.txt > ./news.2011/train.txt

mkdir -p news.combined
cat ./news.2007/train.txt ./news.2008/train.txt ./news.2009/train.txt ./news.2010/train.txt ./news.2011/train.txt > news.combined/train.txt
