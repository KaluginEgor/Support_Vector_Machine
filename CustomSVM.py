import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

class CustomSVM:
    """Custom implementation of Support Vector Machine using python library named cvxopt
    """

    def __init__(self):
        self._w = None  # np.array of length d (number of features)
        self._b = None  # float
        self._c = None  # float
        self._d = None  # int

        cvxopt_solvers.options['show_progress'] = False
        cvxopt_solvers.options['abstol'] = 1e-10
        cvxopt_solvers.options['reltol'] = 1e-10
        cvxopt_solvers.options['feastol'] = 1e-10

    def predict(self, x):
        """Predict an 0/1 class for samples
        
        Let's say n is number of samples, d is number of features.
        Raises RuntimeError if classifier was not fitted yet

        params:
            x: np.array with shape (n, d)
        returns: np.array of length n: predicted class 0/1 for each sample
        """
        if self._w is None and self._b is None:
            raise RuntimeError
        elif self._c is None and self._d is None:
            return np.where(np.array(cvxopt_matrix(x) @ self._w + self._b) > 0, 1, 0).reshape(-1)
        else:
            np.power(np.matmul(self.sv, x.T) + self._c, self._d) * self.alphas * self.sv_y
            pred = np.sum(np.power(np.matmul(self.sv, x.T) + self._c, self._d) * self.alphas * self.sv_y, axis=0) + self._b
            return np.where(pred > 0, 1, 0).reshape(-1)

    def fit_hard_svm_with_qp(self, x, y):
        """Train classifier by solving a prime quadratic optimization problem
        
        Works only if samples are linearly separable

        params:
            x: np.array with shape (n, d)
            y: np.array with shape (n) - 0/1
        """
        n, d = x.shape

        yc = np.where(y > 0.5, 1, -1)
        yc = yc.reshape(-1, 1) * 1.

        xc = np.c_[np.ones(n), x]

        P = cvxopt_matrix(np.diag(np.concatenate((np.array([0]), np.ones(d)))), tc = 'd')
        q = cvxopt_matrix(np.zeros(d + 1))
        G = cvxopt_matrix(np.multiply(yc, xc))
        h = cvxopt_matrix(-1 * np.ones(n).reshape(n, 1))

        sol = cvxopt_solvers.qp(P, q, G, h)
        self._w = -np.array(sol['x'])[1:]
        self._b = -np.array(sol['x'])[0]
        self._c = None
        self._d = None

    def fit_hard_svm_with_qp_dual(self, x, y):
        """Train classifier by solving a dual quadratic optimization problem
        
        Works only if samples are linearly separable

        params:
            x: np.array with shape (n, d)
            y: np.array with shape (n) - 0/1
        """
        n, d = x.shape
        yc = np.where(y > 0.5, 1, -1)
        yc = yc.reshape(-1, 1) * 1.
        X_dash = yc * x
        H = np.dot(X_dash , X_dash.T) * 1.
        
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((n, 1)))
        G = cvxopt_matrix(-np.eye(n))
        h = cvxopt_matrix(np.zeros(n))
        A = cvxopt_matrix(yc.reshape(1, -1))
        b = cvxopt_matrix(np.zeros(1))

        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        self._w = ((yc * alphas).T @ x).reshape(-1,1)

        S = (alphas > 1e-10).flatten()

        self._b = np.mean(yc[S] - np.dot(x[S], self._w))
        self._c = None
        self._d = None

    def fit_soft_svm_with_qp(self, x, y, C=1.):
        """Train classifier by solving a prime quadratic optimization problem
        
        0.5 || w ||^2 + C sum(ksi_i) 

        params:
            x: np.array with shape (n, d)
            y: np.array with shape (n) - 0/1
            C: float
        """
        n, d = x.shape
        d += 1

        yc = np.where(y > 0.5, 1, -1)
        yc = yc.reshape(-1, 1) * 1.

        xc = np.c_[np.ones(n), x]

        P = np.zeros((d + n, d + n))
        P[:d, :d] = np.identity(d)
        
        q = np.zeros((d + n, 1))
        q[d:] = C

        G = np.zeros((n * 2, d + n))
        G[:n, :d] = -np.multiply(yc, xc)
        G[:n, d:] = -np.identity(n)
        G[n:, d:] = -np.identity(n)

        h = np.zeros((n * 2, 1))
        h[:n] = -1


        P = cvxopt_matrix(P, tc = 'd')
        q = cvxopt_matrix(q, tc = 'd')
        G = cvxopt_matrix(G, tc = 'd')
        h = cvxopt_matrix(h, tc = 'd')

        sol = cvxopt_solvers.qp(P, q, G, h)
        self._w = np.array(sol['x'])[1:d]
        self._b = np.array(sol['x'])[0]
        self._c = None
        self._d = None

    def fit_soft_svm_with_qp_dual(self, x, y, C=1.):
        """Train classifier by solving a dual quadratic optimization problem
        
        params:
            x: np.array with shape (n, d)
            y: np.array with shape (n) - 0/1
            C: float
        """
        n, d = x.shape
        yc = np.where(y > 0.5, 1, -1)
        yc = yc.reshape(-1, 1) * 1.
        X_dash = yc * x
        H = np.dot(X_dash , X_dash.T) * 1.

        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((n, 1)))
        G = cvxopt_matrix(np.vstack((np.eye(n)*-1, np.eye(n))))
        h = cvxopt_matrix(np.hstack((np.zeros(n), np.ones(n) * C)))
        A = cvxopt_matrix(yc.reshape(1, -1))
        b = cvxopt_matrix(np.zeros(1))

        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        self._w = ((yc * alphas).T @ x).reshape(-1,1)
        S = ((alphas > 1e-10) & (alphas <= C)).flatten()
        self._b = np.mean(yc[S] - np.dot(x[S], self._w))
        self._c = None
        self._d = None

    def fit_soft_svm_with_sgd(self, x, y, C=1.):
        """Train classifier with SGD
        
        params:
            x: np.array with shape (n, d)
            y: np.array with shape (n) - 0/1
            C: float
        """
        n, d = x.shape
        d += 1

        yc = np.where(y > 0.5, 1, -1)
        yc = yc.reshape(-1, 1) * 1.

        xc = np.c_[np.ones(n), x]
        
        weights = []
        theta = np.zeros(d).reshape(-1, 1)
        epochs = n
        for epoch in range(epochs):
            w = theta / (epoch + 1)
            weights.append(w)
            index = np.random.randint(0, n)
            if yc[index] * (w.T @ xc[index]) < 1:
                theta += (yc[index] * xc[index]).reshape(-1, 1)
        
        final_weight = np.mean(weights, axis=0)
        self._w = final_weight[1:]
        self._b = final_weight[0]
        self._c = None
        self._d = None

    def fit_soft_svm_with_poly_kernel(self, x, y, C=1., c=1., d=3):
        """Train classifier with poly kernel

        K(a, b) = (<a, b> + c)^d
        
        params:
            x: np.array with shape (n, d)
            y: np.array with shape (n) - 0/1
            C: float
            c: float
            d: int
        """
        self._c = c
        self._d = d

        n, d = x.shape
        yc = np.where(y > 0.5, 1, -1)
        yc = yc.reshape(-1, 1) * 1.
        H = np.power(np.matmul(x, x.T) + self._c, self._d)

        P = cvxopt_matrix(np.matmul(yc, yc.T) * H)
        q = cvxopt_matrix(-np.ones((n, 1)))
        G = cvxopt_matrix(np.vstack((np.eye(n)*-1, np.eye(n))))
        h = cvxopt_matrix(np.hstack((np.zeros(n), np.ones(n) * C)))
        A = cvxopt_matrix(yc.reshape(1, -1))
        b = cvxopt_matrix(np.zeros(1))

        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        S = ((alphas > 1e-10) & (alphas <= C)).flatten()

        ind = np.arange(len(S))[S]
        self.alphas = alphas[S]
        self.sv = x[S]
        self.sv_y = yc[S]
        
        self._b = self.sv_y - np.sum(np.power(np.matmul(self.sv, self.sv.T) + self._c, self._d) * self.alphas * self.sv_y, axis=0)
        self._b = np.sum(self._b) / self._b.size
