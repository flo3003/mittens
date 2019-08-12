import numpy as np
from scipy import sparse
import sys
import data_io
from mittens.np_mittens import Mittens, GloVe

# Load the file that contains the initial vector
pretrained_vectors="pretrained_vectors.txt"

(words,  weights) = data_io.getWordmap(pretrained_vectors)

print  weights.shape
initial_embeddings = {v:  weights[words[v]] for v in words}

# Load local IMDB vocabulary
my_vocabulary=open('vocabulary.txt', 'r')
vocab = my_vocabulary.read().split('\n')
vocab_len=len(vocab)-1

#Load co-occurence matrix
print "Reading co-occurrence matrix..."
data=np.genfromtxt('coo_matrix.txt', names=True, dtype=None, delimiter=',')
my_coo_matrix=sparse.coo_matrix((data['cooccurrence'],(data['word_a'],data['word_b'])),  shape=(vocab_len, vocab_len))

print "Converting co-occurence matrix to csr format..."
my_csr_matrix=my_coo_matrix.tocsr()

# Initialize mittens for random values
print "Initializing mittens model..."
mittens_model = Mittens(n= weights.shape[1], max_iter=2, learning_rate=0.01, init_file='original_embeddings.txt')

# Note: n must match the original embedding dimension
print "Running mittens..."
new_embeddings, hist = mittens_model.fit(my_csr_matrix, vocab=vocab[:-1], initial_embedding_dict= initial_embeddings)

# Save error history
np.savetxt("error.txt",hist)

# Save mittens embeddings
np.savetxt("mittens_embeddings.txt", new_embeddings)
