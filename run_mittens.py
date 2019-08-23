import numpy as np
from scipy import sparse
import sys
import data_io
from mittens.np_mittens import Mittens, GloVe
import argparse

# Train Mittens with the following command:
# python run_mittens.py -p pretrained_vectors.txt -c coo_matrix.txt -v vocabulary.txt -lr 0.01 -i 2

parser = argparse.ArgumentParser(description='Fit a Mittens model.')

parser.add_argument('--pretrained_vectors', '-p', action='store',
                    default=None,
                    help=('The filename that contains the pretrained vectors'))

parser.add_argument('--my_vocabulary', '-v', action='store',
                    default=None,
                    help=('The filename that contains the list of words in the vocabulary.'))

parser.add_argument('--my_coo_matrix', '-c', action='store',
                    default=None,
                    help=('The filename that contains the co-occurrence matrix in sparse format.'))

parser.add_argument('--mittens_output', '-m', action='store',
                    default="mittens_embeddings.txt",
                    help=('The filename that contains the mittens embeddings.'))

parser.add_argument('--learning_rate', '-lr', action='store',
                        default='0.01',
                        help='Learning rate.')

parser.add_argument('--iterations', '-i', action='store',
                        default=250,
                        help=('Number of iterations.'))

args = parser.parse_args()


pretrained_vectors = args.pretrained_vectors

(words,  weights) = data_io.getWordmap(pretrained_vectors)

print  weights.shape
initial_embeddings = {v:  weights[words[v]] for v in words}

my_vocabulary=open(args.my_vocabulary, 'r')
vocab = my_vocabulary.read().split('\n')
vocab_len=len(vocab)-1

print "Reading co-occurrence matrix..."
data=np.genfromtxt(args.my_coo_matrix, names=True, dtype=None, delimiter=',')
my_coo_matrix=sparse.coo_matrix((data['cooccurrence'],(data['word_a'],data['word_b'])),  shape=(vocab_len, vocab_len))

print "Converting co-occurence matrix to csr format..."
my_csr_matrix=my_coo_matrix.tocsr()

print "Initializing mittens model..."
mittens_model = Mittens(n= weights.shape[1], max_iter=int(args.iterations), learning_rate=float(args.learning_rate))

print "Running mittens..."
new_embeddings, hist = mittens_model.fit(my_csr_matrix, vocab=vocab[:-1], initial_embedding_dict= initial_embeddings)

# Save error history
np.savetxt("error.txt",hist)

# Save mittens embeddings
np.savetxt(args.mittens_output, new_embeddings)
