import time
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_array
from scipy.linalg import cholesky, solve_triangular


ACTIVATIONS = {
    # foundamental activation functions
    "linear":  lambda x: x,
    "relu":    lambda x: np.maximum(0, x),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh":    lambda x: np.tanh(x),
    # advanced activation functions
    "leaky_relu": lambda x: np.maximum(0.01 * x, x), 
    "swish": lambda x: x * (1 / (1 + np.exp(-x))),
    "mish": lambda x: x * np.tanh(np.log(1 + np.exp(x))),
    "hard_sigmoid": lambda x: np.clip(0.2 * x + 0.5, 0, 1),
    "hard_swish":   lambda x: x * np.clip(0.2 * x + 0.5, 0, 1),
}


class NodeGenerator:
    ''' Efficient node generator for Broad Learning System using NumPy '''
    
    def __init__(self, activation='relu', whiten=False):
        self.Wlist = []  # Weights for each node
        self.blist = []  # Biases  for each node
        self.whiten = whiten
        self.spW = None  # For sparse representation
        self._cached_transform_params = None  # Cached big matrices
        
        # Activation function mapping
        self.activation = ACTIVATIONS[activation]

    def orth(self, W):
        ''' Efficient orthogonalization via QR decomposition '''
        print(f'     Doing orthogonalization via QR decomposition')
        Q, _ = np.linalg.qr(W, mode='reduced')
        return Q

    def generate_nodes(self, data, feature_size, times):
        ''' Generate nodes with vectorized operation '''
        self.Wlist = []
        self.blist = []
        self._cached_transform_params = None  # Invalidate cache
        input_dim = data.shape[1]  # 

        # pre-allocate memory for weights and biases
        all_weights = np.empty((times, input_dim, feature_size))
        all_biases  = np.empty((times, feature_size))
        
        # Generate weights and biases
        print(f'     Generate weights and biases of nodes')
        for i in range(times):
            W = np.random.uniform(-1, 1, size=(input_dim, feature_size))
            if self.whiten: W = self.orth(W)  # 
            all_weights[i] = W
            all_biases[i]  = np.random.uniform(-0.5, 0.5, size=(feature_size,))
        
        # Store weights and biases
        self.Wlist = list(all_weights)
        self.blist = list(all_biases)

        return self.transform(data)

    def transform(self, X):
        ''' Efficient transformation using batched operations '''
        print('     Doing transformation and activation')
        
        if self.spW is not None:
            return self.activation(X @ self.spW)
        
        # Use cached matrices if available
        if self._cached_transform_params is None:
            W_big = np.hstack(self.Wlist)
            b_big = np.concatenate(self.blist, axis=0)
            self._cached_transform_params = (W_big, b_big)
        else:
            W_big, b_big = self._cached_transform_params
        
        return self.activation(X @ W_big + b_big)

    def update(self, otherW, otherb):
        ''' Update weights and biases '''
        self.Wlist += otherW
        self.blist += otherb
        self._cached_transform_params = None



class BLS(BaseEstimator):
    """Optimized Broad Learning System implementation using NumPy"""
    
    def __init__(self,
                 feature_times=10,
                 enhance_times=10,
                 n_classes=10,
                 mapping_function='linear',
                 enhance_function='tanh',
                 feature_size='auto',
                 reg=0.01,
                 sig=0.01,
                 use_sparse=False):
        self.feature_times = feature_times
        self.enhance_times = enhance_times
        self.feature_size = feature_size
        self.n_classes = n_classes
        self.reg = reg
        self.sig = sig
        self.use_sparse = use_sparse
        self.mapping_function = mapping_function
        self.enhance_function = enhance_function
        
        # generator for mapping and enhance node
        self.mapping_generator = NodeGenerator(mapping_function)
        self.enhance_generator = NodeGenerator(enhance_function)
        
        print("="*50)
        print(f"Initializing BLS Model")
        print(f"  Feature nodes: {feature_times}, Enhance nodes: {enhance_times}")
        print(f"  Mapping function: {mapping_function}")
        print(f"  Enhance function: {enhance_function}")
        print(f"  Regularization: {reg}, Sigma: {sig}")
        print(f"  Use sparse: {use_sparse}")
        print("="*50, '\n')

        # Model parameters
        self.W = None
        self.pseudoinverse = None
        self.is_fitted = False
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self._mapping_nodes = None  # Cache for mapping nodes


    def compute_pinv(self, A):
        """Efficient ridge pseudoinverse with dual/primal formulation"""
        n, m = A.shape
        print(f"     Computing ridge pseudoinverse for matrix of shape {A.shape}")
        print(f"     Using {'dual' if n < m else 'primal'} formulation")
        start_time = time.time()

        # Use dual formulation for wide matrices
        if n < m:
            # A A^T + λI
            M = A @ A.T + self.reg * np.eye(n)
            try:
                L = cholesky(M, lower=True)
                Linv = solve_triangular(L, np.eye(n), lower=True)
                Minv = Linv.T @ Linv
                result = A.T @ Minv
            except:  # Back to SVD if Cholesky fails
                print("     Cholesky failed, using SVD fallback")
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                s = s / (s**2 + self.sig)
                result = Vt.T @ (s[:, None] * U.T)
        else:
            # A^T A + λI
            M = A.T @ A + self.reg * np.eye(m)
            try:
                L = cholesky(M, lower=True)
                Linv = solve_triangular(L, np.eye(m), lower=True)
                Minv = Linv.T @ Linv
                result = Minv @ A.T
            except:  # Back to SVD if Cholesky fails
                print("     Cholesky failed, using SVD fallback")
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                s = s / (s**2 + self.sig)
                result = Vt.T @ (s[:, None] * U.T)
        
        print(f"     Solution computed in {time.time()-start_time:.4f}s")
        return result


    def ridge_solve(self, A, b):
        """ directly solve (A^T A + λI) x = A^T b """
        n, m = A.shape
        print(f"     Solving ridge regression for matrix of shape {A.shape}")
        print(f"     Using {'dual' if n < m else 'primal'} formulation")
        start_time = time.time()
        
        if n < m:  # dual form
            M = A @ A.T + self.reg * np.eye(n)
            try:
                L = cholesky(M, lower=True)
                y = solve_triangular(L, b, lower=True)
                y = solve_triangular(L.T, y, lower=False)
                result = A.T @ y
            except:
                print("     Cholesky failed, using SVD fallback")
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                s = s / (s**2 + self.sig)
                result = Vt.T @ (s[:, None] * (U.T @ b))
        else:      # prime form
            M = A.T @ A + self.reg * np.eye(m)
            try:
                L = cholesky(M, lower=True)
                ATA = A.T @ b
                y = solve_triangular(L, ATA, lower=True)
                x = solve_triangular(L.T, y, lower=False)
                result = x
            except:
                print("     Cholesky failed, using SVD fallback")
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                s = s / (s**2 + self.sig)
                result = Vt.T @ (s[:, None] * (U.T @ b))
        
        print(f"     Solution computed in {time.time()-start_time:.4f}s")
        return result


    def fit(self, X, y):
        """Train BLS model with optimized operations"""
        X = check_array(X, ensure_2d=True)
        
        if self.is_fitted:
            print("    Model already fitted. Resetting...")
            self.reset()
            
        # Automatic feature size determination
        if self.feature_size == 'auto':
            self.feature_size = X.shape[1]
        print(f"    Feature size is: {self.feature_size}")
        
        # Prepare labels
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            self.onehot_encoder.fit(y)
            y_encoded = self.onehot_encoder.transform(y)
        else:
            y_encoded = y
        
        # Generate mapping nodes
        print("  ·Generate mapping nodes ...")
        mapping_nodes = self.mapping_generator.generate_nodes(
            X, self.feature_size, self.feature_times
        )
        self._mapping_nodes = mapping_nodes
        
        # calculate sparse representation using pinv matrix
        if self.use_sparse:
            pinvX = self.compute_pinv(X)
            self.mapping_generator.spW = pinvX @ mapping_nodes
        
        # Generate enhancement nodes
        print("  ·Generate enhance nodes ...")
        enhance_nodes = self.enhance_generator.generate_nodes(
            mapping_nodes, self.feature_size, self.enhance_times
        )
        
        # Concatenate features
        A = np.hstack((mapping_nodes, enhance_nodes))
        
        # Compute output weights
        self.W = self.ridge_solve(A, y_encoded)
        self.is_fitted = True
        return self

    def add_enhancement_nodes(self, X, y, num_nodes=5):
        ''' add enhance node and update model weight '''
        print(f"> Generating {num_nodes} enhancement nodes ...")
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first")
        
        # Prepare labels
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            # self.onehot_encoder.fit(y)
            y_encoded = self.onehot_encoder.transform(y)
        else:
            y_encoded = y
        
        # get all feature nodes
        if self._mapping_nodes is None:
            mapping_nodes = self.mapping_generator.transform(X)
        else:
            mapping_nodes = self._mapping_nodes
        
        # get current output of all nodes
        current_enhance = self.enhance_generator.transform(mapping_nodes)
        A = np.hstack((mapping_nodes, current_enhance))

        W_new_list = []
        b_new_list = []
        new_nodes = np.zeros((mapping_nodes.shape[0], num_nodes))

        # generate weights and biases for new node
        input_dim = mapping_nodes.shape[1]
        for i in range(num_nodes):
            W_new = np.random.uniform(-1, 1, size=(input_dim, 1))
            b_new = np.random.uniform(-0.5, 0.5, size=(1,))
            h = self.enhance_generator.activation(mapping_nodes @ W_new + b_new)
            
            W_new_list.append(W_new)
            b_new_list.append(b_new)
            new_nodes[:, i] = h.flatten()
        
        # update enhance node generator
        self.enhance_generator.update(W_new_list, b_new_list)

        # recompute output weights
        A_extended = np.hstack((A, new_nodes))
        self.W = self.ridge_solve(A_extended, y_encoded)
        return self


    def predict_proba(self, X):
        ''' Predict class probabilities '''
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first")
            
        print(f"  ·Starting prediction for {X.shape[0]} samples")
            
        X = check_array(X, ensure_2d=True)
        
        # Transform
        mapping_nodes = self.mapping_generator.transform(X)
        enhance_nodes = self.enhance_generator.transform(mapping_nodes)
        H = np.hstack((mapping_nodes, enhance_nodes))
        
        # Prediction and softmax
        print("    Computing predictions ...", end=' ')
        logits = H @ self.W
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        print("Finish")
            
        return proba


    def predict(self, X):
        ''' Predict class labels '''
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


    def reset(self):
        ''' Reset model state '''
        self.W = None
        self.mapping_generator = NodeGenerator(self.mapping_function, verbose=self.verbose)
        self.enhance_generator = NodeGenerator(self.enhance_function, verbose=self.verbose)
        self.is_fitted = False
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
