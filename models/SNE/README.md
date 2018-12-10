# README #

Code for the paper "SNE: Signed Network Embedding"

[Note by Vasco Portilheiro, December 2018: accessed from https://bitbucket.org/bookcold/sne-signed-network-embedding/src/6622a7f118b3055212888c1348cf18fb02ab33a7?at=master]

### Set up ###

* Run `python walk.py` to generate random walks
* Run `python SNE.py` to train the network embeddings
* `test.py` includes the testing code for node classification and link prediction

### Data ###
*wiki_edit.txt* is the edge list file of signed graph "WikiEditor" described in paper

*wiki_usr_labels.txt* contains the labels of "WikiEditor" nodes

### Requirements ###

* Python 3.6
* Tensorflow 1.2.1