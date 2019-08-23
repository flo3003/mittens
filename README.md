<img src="img/mittens_logo.png" alt="title" width="100">

# Mittens

This package contains a faster and more efficient [NumPy](https://github.com/numpy/numpy) implementation of [Mittens](https://arxiv.org/abs/1803.09901) [1] which can work with **sparse matrices**.

## Installation

Obviously first clone this repository.

```
git clone https://github.com/flo3003/mittens.git
```

Change into the cloned mittens directory:

```
cd mittens
```

and then run:

```
python test_mittens.py
```

## Training



### Instructions to create the necessary files:

Now you will need to clone the following [Github repo](https://github.com/flo3003/glove-python) in mittens's directory

```
git clone https://github.com/flo3003/glove-python.git
```

Then run the following commands in order:

```
cd glove-python
python setup.py cythonize
pip install -e .
```

In the glove-python directory run

```
python examples/get_database_files.py -c /path/to/some/corpustextfile -d 100
```
The argument `-d` refers to the embedding dimensions. The default is 100. 

`corpustextfile` can be any plain text file (with words being separated by space) with punctuation or not. 

The following files will be created:
- `coo_matrix.csv` which contains the co-occurrence matrix of `corpustextfile` in sparse format
- `word_mapping.csv` which contains the mapping of each **word** to an **Id**
- `corpus.model` and `glove.model` are the saved corpus and glove models
- `random_initial_vectors.txt` contains the embeddings' initialization 

Then run:
```
chmod +x get_vocab.sh
./get_vocab.sh
```

These commands create the file `vocabulary.csv` which is the vocabulary of the corpus. 

Finally you need to download pretrained embeddings in GloVe format (e.g. [glove.6B](http://nlp.stanford.edu/data/glove.6B.zip)) in the `mittens` directory.

Now that we have the necessary files, we are ready to train Mittens:

Simply run 

```
python run_mittens.py -p pretrainedembeddingsfile -c glove-python/coo_matrix.csv -v vocabulary.csv -m mittens_embeddings.txt -lr 0.01 -i 250
```

You can select:
- a name for the output mittens file *(-m)* - the default is `mittens_embeddings.txt`, 
- the learning rate *(-lr)* - the default is `0.01` and 
- the number of iterations *(-i)* - the default is `250`

Once trained, `mittens_embeddings.txt` should be *compatible* with the existing embeddings in the sense that they will be oriented such that using a mix of the the two embeddings is meaningful (e.g. using original embeddings for any test-set tokens that were not in the training set).

We also store the history of the training error in `error.txt` file.

## References
[1] Nicholas Dingwall and Christopher Potts. 2018. *Mittens: An Extension of GloVe for Learning Domain-Specialized Representations*. (NAACL 2018) [[code]](https://github.com/roamanalytics/roamresearch/tree/master/Papers/Mittens)
