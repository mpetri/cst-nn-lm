#!/bin/bash

wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
cd wikitext-103
mv wiki.test.tokens test.txt
mv wiki.valid.tokens valid.txt
mv wiki.train.tokens train.txt
