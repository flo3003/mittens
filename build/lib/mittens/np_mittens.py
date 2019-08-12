"""np_mittens.py

Fast implementations of Mittens and GloVe in Numpy.

See https://nlp.stanford.edu/pubs/glove.pdf for details of GloVe.

References
----------
[1] Jeffrey Pennington, Richard Socher, and Christopher D. Manning.
2014. GloVe: Global Vectors for Word Representation

[2] Nick Dingwall and Christopher Potts. 2018. Mittens: An Extension
of GloVe for Learning Domain-Specialized Representations

Authors: Nick Dingwall, Chris Potts
"""
import numpy as np
from scipy.sparse import csr_matrix

from mittens.mittens_base import randmatrix, noise
from mittens.mittens_base import MittensBase, GloVeBase
from scipy import ndimage

_FRAMEWORK = "NumPy"
_DESC = """
    The TensorFlow version is faster, especially if used on GPU. 
    To use it, install TensorFlow, restart your Python kernel and 
    import from the base class:

    >>> from mittens import {model}
    """


class Mittens(MittensBase):
    __doc__ = MittensBase.__doc__.format(
        framework=_FRAMEWORK,
        second=_DESC.format(model=MittensBase._MODEL))

    @property
    def framework(self):
        return _FRAMEWORK

    def _fit(self, coincidence, weights, log_coincidence,
             vocab=None,
             initial_embedding_dict=None,
             fixed_initialization=None):
        self._initialize_w_c_b(self.n_words, vocab, initial_embedding_dict)

        if fixed_initialization is not None:
            assert self.test_mode
            self.W = fixed_initialization['W']
            self.C = fixed_initialization['C']
            self.bw = fixed_initialization['bw']
            self.bc = fixed_initialization['bc']

        if self.test_mode:
            # These are stored for testing
            self.W_start = self.W.copy()
            self.C_start = self.C.copy()
            self.bw_start = self.bw.copy()
            self.bc_start = self.bc.copy()

        er_vec=[]
        for iteration in range(self.max_iter):
            #print("pred before:")
            pred = self._make_prediction(coincidence)
	    #print("pred:")
	    #print(pred)
	    #print(pred.shape)
	    #print("log_coincidence:")
	    #print(log_coincidence)
	    #print(log_coincidence.shape)
            #print("diffs:")
	    diffs = pred - log_coincidence.reshape(len(log_coincidence),1)
	    #print("diffs:")
	    #print(diffs)
	    #print("weights:")
	    #print(weights)
            gradients, error = self._get_gradients_and_error(diffs, coincidence, weights)
            self._check_shapes(gradients)
            self.errors.append(error)
            self._apply_updates(gradients)
            self._progressbar("error {:4.4f}".format(error), iteration)
            er_vec.append(error)
        return self.W + self.C , er_vec

    def _check_shapes(self, gradients):
        assert gradients['W'].shape == self.W.shape
        assert gradients['C'].shape == self.C.shape
        assert gradients['bw'].shape == self.bw.shape
        assert gradients['bc'].shape == self.bc.shape

    def _initialize_w_c_b(self, n_words, vocab, initial_embedding_dict):

	if self.init_file:

		self.W = np.loadtxt(self.init_file) + noise(self.n)
		#print(self.W)

		self.C = np.loadtxt(self.init_file) + noise(self.n)
		
	
	else:
        	self.W = randmatrix(n_words, self.n)  # Word weights.
		#print(self.W)
        	#print(type(self.W))
        	self.C = randmatrix(n_words, self.n)  # Context weights.
		#print(self.C)
        	#print(type(self.C))

        if initial_embedding_dict:
            assert self.n == len(next(iter(initial_embedding_dict.values())))

            self.original_embedding = np.zeros((len(vocab), self.n))
            self.has_embedding = np.zeros(len(vocab), dtype=bool)

            for i, w in enumerate(vocab):
                if w in initial_embedding_dict:
                    self.has_embedding[i] = 1
                    embedding = np.array(initial_embedding_dict[w])
                    self.original_embedding[i] = embedding
                    # Divide the original embedding into W and C,
                    # plus some noise to break the symmetry that would
                    # otherwise cause both gradient updates to be
                    # identical.
                    self.W[i] = 0.5 * embedding + noise(self.n)
                    self.C[i] = 0.5 * embedding + noise(self.n)
            # This is for testing. It differs from
            # `self.original_embedding` only in that it includes the
            # random noise we added above to break the symmetry.
            self.G_start = self.W + self.C

        self.bw = randmatrix(n_words, 1)
	#print(self.bw)
        self.bc = randmatrix(n_words, 1)
	#print(self.bc)
        self.ones = np.ones((n_words, 1))


	#full_pred = np.dot(self.W, self.C.T) + self.bw + self.bc.T

	#print("full_pred:")
	#print (full_pred)

    def _make_prediction(self,M):
        # Here we make use of numpy's broadcasting rules
        #pred = np.dot(self.W, self.C.T) + self.bw + self.bc.T
	#print (M.nonzero()[0])
	#print (M.nonzero()[1])
	#print (M.data)
	#print self.W.shape
	pred=np.sum(self.W[M.nonzero()[0]]*self.C[M.nonzero()[1]], axis=1).reshape(len(M.data),1)+self.bw[M.nonzero()[0]]+self.bc[M.nonzero()[1]]

	#print("pred:")
	#print(pred)
        return pred

    def _get_gradients_and_error(self, diffs, weighted_diffs_mat, weights):
        #print("weighted_diffs before:")
	weighted_diffs = np.multiply(weights.reshape(len(weights),1), diffs)
        #print("weighted_diffs after:")
        #print (weighted_diffs)
        # First we compute the GloVe gradients
	#print(weighted_diffs_mat.nonzero()[0])
        #print(weighted_diffs_mat.nonzero()[1])


	weighted_diffs_mat.data=np.array(weighted_diffs).ravel()

	#w = weighted_diffs_mat.transpose()

	#print(type(weighted_diffs_mat))
	
	#print ("weighted_diffs_mat.data:")
	#print (weighted_diffs_mat.data)

	#print (weighted_diffs_mat.toarray())

	#print(self.C)

	#wgrad = weighted_diffs.dot(self.C)
        #print("wgrad before:")
	wgrad = weighted_diffs_mat*self.C
        #print(type(wgrad))
        
	#print("wgrad after:")
	#print(wgrad)


	#print (weighted_diffs_mat.transpose().toarray())
	#print(self.W)

	#cgrad = weighted_diffs.T.dot(self.W)
        #print("cgrad before:")
	cgrad =  weighted_diffs_mat.transpose()*self.W
        #print(type(cgrad))
	#print("cgrad after:")
	#print(cgrad)

        #bwgrad = weighted_diffs_mat.toarray().sum(axis=1).reshape(-1, 1)
        #bcgrad = weighted_diffs_mat.toarray().sum(axis=0).reshape(-1, 1)

	#print("Ta ypoloipa orig:")
	#print(bwgrad)
	#print("bcgrad before:")

        bcgrad = ndimage.sum(weighted_diffs_mat.data, weighted_diffs_mat.nonzero()[1], np.arange(weighted_diffs_mat.nonzero()[1].min(), weighted_diffs_mat.nonzero()[1].max()+1)).reshape(-1,1)
        #print(type(bcgrad))
        #print("bcgrad after:")
        #print("bwgrad before:")
        bwgrad = ndimage.sum(weighted_diffs_mat.data, weighted_diffs_mat.nonzero()[0], np.arange(weighted_diffs_mat.nonzero()[0].min(), weighted_diffs_mat.nonzero()[0].max()+1)).reshape(-1,1)
        #print("bwgrad after:")
        #print(type(bwgrad))
        #print("Ta ypoloipa:")
        #print(bwgrad)
        #print(bcgrad)
        #error = (0.5 * np.multiply(weights, diffs ** 2)).sum()
	#bbb = (0.5 * np.multiply(weights, diffs ** 2)).sum()
	#print(bbb)
	#print(bbb.shape)
        #print(diffs.shape, weights.shape)
	error = (0.5 * np.multiply(weights.reshape(len(weights),1), diffs**2)).sum()
	#print(error)
        #print(error.shape)
        #curr_embedding = self.W + self.C
        #distance = curr_embedding - self.original_embedding
	#ggg = (np.linalg.norm(distance, ord=2, axis=1) ** 2).sum()
	#print(ggg)
	#print(ggg.shape)
	'''
        wgrad = []
        for i in range(len(data[0])): wgrad.append(ndimage.sum(data.T[i], rows, np.arange(rows.min(), rows.max()+1)))
        wgrad=np.asarray(wgrad).T
        #wgrad = weighted_diffs.dot(self.C[M.nonzero()[1]])
        weighted_diffs_T =coo_matrix((weighted_diffs.data, (cols, rows)), shape=M.shape)
        data = np.multiply(W[M.nonzero()[1]],weighted_diffs_T.data.reshape(len(M.data),1))
        cgrad = []
        for i in range(len(data[0])): wgrad.append(ndimage.sum(data.T[i], rows, np.arange(rows.min(), rows.max()+1)))
        cgrad=np.asarray(cgrad).T
        #cgrad = weighted_diffs.T.dot(self.W)

        data = weighted_diffs
        bwgrad = ndimage.sum(data, rows, np.arange(groups.min(), groups.max()+1)).reshape(-1,1)
        #bwgrad = weighted_diffs.sum(axis=1).reshape(-1, 1)
        bwgrad = ndimage.sum(data, cols, np.arange(groups.min(), groups.max()+1)).reshape(-1,1)
        #bcgrad = weighted_diffs.sum(axis=0).reshape(-1, 1)
        error = (0.5 * np.multiply(weights, diffs ** 2)).sum() #opws einai
	'''

        # Then we add the Mittens term (only if mittens > 0)
        if self.mittens > 0:
            curr_embedding = self.W + self.C
            distance = curr_embedding[self.has_embedding, :] - \
                       self.original_embedding[self.has_embedding, :]
            wgrad[self.has_embedding, :] += 2 * self.mittens * distance
            cgrad[self.has_embedding, :] += 2 * self.mittens * distance
            error += self.mittens * (
                np.linalg.norm(distance, ord=2, axis=1) ** 2).sum()
        return {'W': wgrad, 'C': cgrad, 'bw': bwgrad, 'bc': bcgrad}, error

    def _apply_updates(self, gradients):
        """Apply AdaGrad update to parameters.

        Parameters
        ----------
        gradients

        Returns
        -------

        """
        if not hasattr(self, 'optimizers'):
            self.optimizers = \
                {obj: AdaGradOptimizer(self.learning_rate)
                 for obj in ['W', 'C', 'bw', 'bc']}
        self.W -= self.optimizers['W'].get_step(gradients['W'])
        self.C -= self.optimizers['C'].get_step(gradients['C'])
        self.bw -= self.optimizers['bw'].get_step(gradients['bw'])
        self.bc -= self.optimizers['bc'].get_step(gradients['bc'])


class AdaGradOptimizer:
    """Simple AdaGrad optimizer.

    This is loosely based on the Tensorflow version. See
    https://github.com/tensorflow/tensorflow/blob/master/
    tensorflow/python/training/adagrad.py.

    Parameters
    ----------
    learning_rate : float
    initial_accumulator_value : float (default: 0.1)
        Initialize the momentum with this value.
    """

    def __init__(self, learning_rate, initial_accumulator_value=0.1):
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self._momentum = None

    def get_step(self, grad):
        """Computes the 'step' to take for the next gradient descent update.

        Returns the step rather than performing the update so that
        parameters can be updated in place rather than overwritten.

        Examples
        --------
        >>> gradient = # ...
        >>> optimizer = AdaGradOptimizer(0.01)
        >>> params -= optimizer.get_step(gradient)

        Parameters
        ----------
        grad

        Returns
        -------
        np.array
            Size matches `grad`.
        """
        if self._momentum is None:
            self._momentum = self.initial_accumulator_value * np.ones_like(grad)
        self._momentum += grad ** 2
        return self.learning_rate * grad / np.sqrt(self._momentum)


class GloVe(GloVeBase, Mittens):

    __doc__ = GloVeBase.__doc__.format(
        framework=_FRAMEWORK,
        second=_DESC.format(model=GloVeBase._MODEL))
