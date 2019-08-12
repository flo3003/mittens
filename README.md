<img src="img/mittens_logo.png" alt="title" width="100">

# Mittens

This package contains a faster and more efficient [NumPy](https://github.com/numpy/numpy) implementation of [Mittens](https://arxiv.org/abs/1803.09901) which can work with sparse matrices.

## Installation

### Dependencies

Mittens only requires `numpy`.

### User installation

The easiest way to install `mittens` is with `pip`:

```
pip install -U mittens
```

You can also install it by cloning the repository and adding it to your Python path. Make sure you have at least `numpy` installed.

## Examples

The file vocabulary.txt contains the list of words in the vocabulary. For example:

this
is
an
example

It is assumed that you have already computed the weighted co-occurrence matrix which is stored in a .txt file as follows:

word_a,word_b,cooccurrence
0,1,8.0
0,2,1.75
0,3,0.53

where 0,1,2,3 correspond to the words' indices in the vocabulary.txt file. 

### Mittens

To use Mittens, you first need pre-trained embeddings. In our paper, we used Pennington et al's embeddings, available from the [Stanford GloVe website](https://nlp.stanford.edu/projects/glove/).

These vectors should be stored in a dict, where the key is the token and the value is the vector. For example, the function `glove2dict` below manipulates a Stanford embedding file into the appropriate format.

```
import csv
import numpy as np

def glove2dict(glove_filename):
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed

```

Now that we have our embeddings (stored as `original_embeddings`), as well as a co-occurrence matrix and associated vocabulary, we're ready to train Mittens:

```
from mittens import Mittens

# Load `cooccurrence` and `vocab`
# Load `original_embedding`
mittens_model = Mittens(n=50, max_iter=1000)
# Note: n must match the original embedding dimension
new_embeddings = mittens_model.fit(
    cooccurrence,
    vocab=vocab,
    initial_embedding_dict= original_embedding)
```

Once trained, `new_embeddings` should be *compatible* with the existing embeddings in the sense that they will be oriented such that using a mix of the the two embeddings is meaningful (e.g. using original embeddings for any test-set tokens that were not in the training set).


## <a name="speed"></a>Speed

We compared the per-epoch speed (measured in seconds) for a variety of vocabulary sizes using randomly-generated co-occurrence matrices that were approximately 90% sparse. As we see here, for matrices that fit into memory, performance is competitive with the [official C implementation](https://github.com/stanfordnlp/GloVe) when run on a GPU.

For denser co-occurrence matrices, Mittens will have an advantage over the C implementation since it's speed does not depend on sparsity, while the official release is linear in the number of non-zero entries.

|                           | 5K (CPU) | 10K (CPU) | 20K (CPU) | 5K (GPU) | 10K (GPU) | 20K (GPU) |
|:--------------------------|---------:|----------:|----------:|---------:|----------:|----------:|
| Non-vectorized TensorFlow |     14.02|      63.80|     252.65|     13.56|      55.51|     226.41|
| Vectorized Numpy          |      1.48|       7.35|      50.03|         −|          −|          −|
| Vectorized TensorFlow     |      1.19|       5.00|      28.69|      0.27|       0.95|       3.68|
| Official GloVe            |      0.66|       1.24|       3.50|         −|          −|          −|

## References
[1] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. *GloVe: Global Vectors for Word Representation*.

[2] Nicholas Dingwall and Christopher Potts. 2018. *Mittens: An Extension of GloVe for Learning Domain-Specialized Representations*. (NAACL 2018) [[code]](https://github.com/roamanalytics/roamresearch/tree/master/Papers/Mittens)
