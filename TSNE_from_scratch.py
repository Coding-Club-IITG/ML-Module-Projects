import numpy as np

class TSNE:
    def __init__(self, perplexity, n_iter=1000, n_components=2, learning_rate=200, early_ex=4):
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.early_ex = early_ex
        self.Y = None

    def binary_search(self, dist_i, i, max_iter=100, tolerance=1e-5):
        std_dis = np.std(dist_i)
        low = 0.01 * std_dis
        high = 5 * std_dis

        for _ in range(max_iter): #using binary search for finding the bronze
            mid = (low + high) / 2.0
            dist_sq_i = dist_i ** 2
            p = np.exp(-dist_sq_i / (2 * mid ** 2))
            p[i] = 0
            s= np.sum(p)
            p = np.maximum(p /s, np.nextafter(0, 1))
            H = -np.sum(p * np.log2(p))
            diff = np.log2(self.perplexity) - H

            if np.abs(diff) < tolerance:
                return mid

            if diff > 0:
                low = mid
            else:
                high = mid

        return mid

    def compute_high_prob(self, data):
        n = data.shape[0]
        prob_h = np.zeros((n, n))

        for i in range(n):
            diff = data[i] - data
            distance = np.linalg.norm(diff, axis=1)
            sigma_i = self.binary_search(distance, i)
            prob_h[i] = np.exp(-distance ** 2 / (2 * sigma_i ** 2))
            prob_h[i][i] = 0
            sum_i = np.sum(prob_h[i])
            prob_h[i] /= sum_i

        epsilon = np.nextafter(0, 1)
        prob_h = np.maximum(prob_h, epsilon)
        prob_h = (prob_h + prob_h.T) / (2 * n)
        return prob_h

    def compute_low_prob(self):
        n = self.Y.shape[0]
        prob_l = np.zeros((n, n))

        for i in range(n):
            diff = self.Y[i] - self.Y
            distance = np.linalg.norm(diff, axis=1)
            prob_l[i] = (1 + distance ** 2) ** -1
            prob_l[i][i] = 0

        prob_l = prob_l / np.sum(prob_l)
        epsilon = np.nextafter(0, 1)
        prob_l = np.maximum(prob_l, epsilon)
        return prob_l

    def compute_grad(self, prob_h, prob_l):
        n = prob_h.shape[0]
        gradient = np.zeros((n, self.n_components))

        for i in range(n):
            diff = self.Y[i] - self.Y 
            dP = np.array(prob_h[i, : ] - prob_l[i, : ])
            dQ = np.array((1 + np.linalg.norm(diff, axis=1) ** 2) ** -1)
            dY= dP*dQ
            
            
            gradient[i] = 4 * np.sum(np.dot(dY, diff))

        return gradient

    def fit_transform(self, data):
        n = data.shape[0]
        self.Y = np.random.randn(n, self.n_components)

        prob_h = self.compute_high_prob(data)

        for t in range(self.n_iter - 1):
            if t < 250:
                momentum = 0.5
                self.early_ex = 4
            else:
                momentum = 0.8
                self.early_ex = 1

            prob_l = self.compute_low_prob()
            grad = self.compute_grad(self.early_ex * prob_h, prob_l)

            if t == 0:
                prev_Y = self.Y.copy()
                self.Y -= self.learning_rate * grad
            else:
                self.Y -= self.learning_rate * grad + momentum * (self.Y - prev_Y)
                prev_Y = self.Y.copy()

        return self.Y