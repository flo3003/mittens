<img src="img/mittens_logo.png" alt="title" width="100">

# Mittens

This package contains a faster and more efficient [NumPy](https://github.com/numpy/numpy) implementation of [Mittens](https://arxiv.org/abs/1803.09901) [1] which can work with sparse matrices.

## Installation

### Dependencies

Mittens only requires `numpy`.

### User installation

You can install it by cloning the repository and running:

```
python test_mittens.py
```

## Examples

The file `vocabulary.txt` contains the list of words in the vocabulary. For example:

```
this
is
an
example
```

It is assumed that you have already computed the weighted co-occurrence matrix which is stored in `coo_matrix.txt` as follows:

```
word_a,word_b,cooccurrence
0,1,8.0
0,2,1.75
0,3,0.53
```

where 0,1,2,3 correspond to the words' indices in the `vocabulary.txt` file. 

To use Mittens, you first need pre-trained embeddings. These vectors should be stored in `pretrained_vectors.txt` as follows:

```
the -0.038194 -0.24487 0.72812 -0.39961
with -0.43608 0.39104 0.51657 -0.13861
.
.
.
```

Finally, a file with the original embeddings of the vocabulary is needed. These vectors should be stored in `original_embeddings.txt` as follows:

```
0.001 0.002 -0.001 0.000
-0.002 -0.001 -0.005 -0.003
0.000 0.003 -0.004 -0.004
-0.001 0.003 0.002 0.001
```

Each line of `original_embeddings.txt` corresponds to each word in `vocabulary.txt` in the same order.

### Mittens

Now that we have our embeddings (stored as `original_embeddings.txt`), a co-occurrence matrix (stored as `coo_matrix.txt`), the associated vocabulary (stored as `vocabulary.txt`) and the pre-trained embeddings (stored as `pretrained_vectors.txt`), we're ready to train Mittens:

Simply run 

```
python run_mittens.py -p pretrained_vectors.txt -c coo_matrix.txt -v vocabulary.txt -o original_embeddings.txt
```

or

```
python run_mittens.py -p pretrained_vectors.txt -c coo_matrix.txt -v vocabulary.txt -o original_embeddings.txt -m mittens_embeddings.txt -lr 0.01 -i 2
```

where you can select a name for the output mittens file (-m), the learning rate (-lr) and the number of iterations (-i).

Once trained, `mittens_embeddings.txt` should be *compatible* with the existing embeddings in the sense that they will be oriented such that using a mix of the the two embeddings is meaningful (e.g. using original embeddings for any test-set tokens that were not in the training set).

We also store the training error in `error.txt` file.

## References
[1] Nicholas Dingwall and Christopher Potts. 2018. *Mittens: An Extension of GloVe for Learning Domain-Specialized Representations*. (NAACL 2018) [[code]](https://github.com/roamanalytics/roamresearch/tree/master/Papers/Mittens)
