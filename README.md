# cs224w-project
The `db` folder contains the data files for the two bitcoin networks. The `models` folder contains (unsurprisingly) the models used for the project. In particular, it contains local, modified copies of SNE and node2vec, which allow for temporal random walks. In addition, I have included in it the modified gensim `word2vec.py` file I used to generate embeddings with "relational weighting". (Note that the change on line 145 forces the code to use my modified version of the slow, Python, as opposed to Cython, version of the code.)

The final project poster and report are included here as pdfs.
