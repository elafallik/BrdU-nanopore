import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from GMM import *
PX = 0.000820905064105752

def forward(alpha, A, prob_xn_given_phi):
    # alpha(z1=i) = p(x1|z1=i)*p(z1=i) = pi_i*p(x1|phi_i), i=1,2
    # alpha(zn=i) = p(xn|zn=i)*( alpha(z(n-1)=1) * p(zn=i|z(n-1)=1) + alpha(z(n-1)=2) * p(zn=i|z(n-1)=2) ) =
    #             = p(xn|phi_i(n))*( alpha(z(n-1)=1) * A_i,1(n) + alpha(z(n-1)=2) * A_i,2(n) )
    return prob_xn_given_phi * (alpha[0] * A.transpose()[0] + alpha[1] * A.transpose()[1])

def backward(beta, A, prob_xn_given_phi):
    # beta(zN=i) = 1, i=1,2
    # beta(zn=i) = beta(z(n+1)=1) * p(x(n+1)|z(n+1)=1) * p(z(n+1)=1|zn=i) +
    #            + beta(z(n+1)=2) * p(x(n+1)|z(n+1)=2) * p(z(n+1)=2|zn=i) =
    #            = beta(z(n+1)=1) * p(x(n+1)|phi_1(n+1)) * A_1,i(n+1) +
    #            + beta(z(n+1)=2) * p(x(n+1)|phi_2(n+1)) * A_2,i(n+1)
    return beta[0] * prob_xn_given_phi[0] * A[0] + beta[1] * prob_xn_given_phi[1] * A[1]


def helper(A, prob_X_given_phi, eta, alpha, beta):

    N = len(prob_X_given_phi[0])
    gamma = np.zeros((N, 2))

    # E-step:
    for n in range(0, N-1):
        prob_xn_given_phi = prob_X_given_phi.transpose()[n+1]
        alpha[n+1] = forward(alpha[n], A, prob_xn_given_phi)
        beta[N-(n+1)] = backward(beta[N-n], A, prob_X_given_phi.transpose()[-(n+1)])
    for n in range(0, N-1):
        # gamma(zn=i) = p(zn=i|X) = (alpha(zn=i) * beta(zn=i)) / p(X)
        gamma[n] = (alpha[n] * beta[n+1])
        print(gamma[n] / PX)
        prob_xn_given_phi = prob_X_given_phi.transpose()[n+1]
        # print(np.sum(gamma[n]))
        # eta(z(n)=i,z(n+1)=j) = p(z(n)=i,z(n+1)=j|X) =
        #                    = (alpha(z(n)=i) * p(x(n+1)|z(n+1)=j) * p(z(n+1)=j|z(n)=i) * beta(z(n+1)=j)) / p(X) =
        #                    = (alpha(z(n)=i) * p(x(n+1)|phi_j) * A_j,i * beta(z(n+1)=j)) / p(X)
        # eta[n-1][i][j] = (alpha[i] * A.transpose()[i][j] * prob_X_given_phi.transpose()[n][j] * beta[j]) / p_X
        eta[n][0] = (alpha[n][0] * prob_xn_given_phi * A.transpose()[0] * beta[n+2])
        eta[n][1] = (alpha[n][1] * prob_xn_given_phi * A.transpose()[1] * beta[n+2])

    # p(X) = alpha(zN=1) + alpha(zN=2)
    p_X = np.sum(alpha[-1])
    gamma = gamma / p_X
    eta = eta / p_X

    # M-step:
    # pi_i = gamma(zn=i) / (gamma(z1=1) + gamma(z1=2))
    pi = gamma[0] / np.sum(gamma[0])
    # A_i,j = sum_k=1,...,N(eta(z(k-1)=i,zk=j)) / (sum_k=1,...,N(eta(z(k-1)=i,zk=1)) + sum_k=1,...,N(eta(z(k-1)=i,zk=2)))
    normalizer = np.sum(eta.transpose()[0] + eta.transpose()[1], axis=1)
    A = (np.sum(eta[0:N].transpose(), axis=2) / normalizer).transpose()
    return A, pi, eta, alpha, beta

def baum_welch(pi, A, prob_X_given_phi, T):
    # init: choose some A(1) and pi (what's 0 will stay 0)
    N = len(prob_X_given_phi[0])
    # init alpha: alpha(z1=i) = p(x1|z1=i)*p(z1=i) = pi_i*p(x1|phi_i), i=1,2
    alpha = np.zeros((N, 2))
    alpha[0] = pi * prob_X_given_phi.transpose()[0]
    # init beta: beta(zN=i) = 1, i=1,2
    beta = np.zeros((N+1, 2))
    beta[-1] = np.array([1, 1])
    # init eta: N * 2 * 2 np array
    eta = np.zeros((N, 2, 2))

    for i in range(T):
        A, pi, eta, alpha, beta = helper(A, prob_X_given_phi, eta, alpha, beta)

    return A, pi

#
# def func(pi, A, prob_X_given_phi):
#     N = 4
#     alpha = np.zeros((N, 2))
#     alpha[0] = pi * prob_X_given_phi.transpose()[0]
#     for i in range(2):
#         alpha[1][i] = prob_X_given_phi.transpose()[1][i] * (alpha[0][0]*A[i][0] + alpha[0][1]*A[i][1])
#     for i in range(2):
#         alpha[2][i] = prob_X_given_phi.transpose()[2][i] * (alpha[1][0]*A[i][0] + alpha[1][1]*A[i][1])
#     for i in range(2):
#         alpha[3][i] = prob_X_given_phi.transpose()[3][i] * (alpha[2][0]*A[i][0] + alpha[2][1]*A[i][1])
#     print(alpha)


if __name__ == '__main__':

    scores, readNum = getDetectData(100)
    dataMean = np.mean(scores)
    dataSigma = np.std(scores)
    threshold = '0'
    scores2 = scores[np.argwhere(scores > 0)].transpose()[0]
    data2Mean = np.mean(scores2)
    data2Sigma = np.std(scores2)
    mu1, sigma1, mu2, sigma2, alpha = EM(scores, dataMean, dataSigma, data2Mean, data2Sigma, ALPHA, 10)

    # p(xn|phi_i) = N(xn|mu_i,sigma_i) = stats.norm.pdf(X, mu_i, sigma_i)[n]
    prob_X_given_phi = np.array([stats.norm.pdf(scores, mu1, sigma1)[:4], stats.norm.pdf(scores, mu2, sigma2)[:4]])
    pi = np.array([0.5, 0.5])  # TODO: initialize somehow
    A = np.array([[0.8, 0.2],
                  [0.2, 0.8]])  # TODO: initialize somehow

    # func(pi, A, prob_X_given_phi)

    # test with simple distributions
    # x = np.array([0, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2])
    # prob_X_given_phi = np.array([stats.uniform.pdf(x, 0, 1), stats.uniform.pdf(x, 1, 1)])
    # pi = np.array([0.5, 0.5])  # TODO: initialize somehow
    # A = np.array([[0.2, 0.8],
    #               [0.5, 0.5]])  # TODO: initialize somehow


    A, pi = baum_welch(pi, A, prob_X_given_phi, 10)
    print(A)
    print(pi)

    a = 2