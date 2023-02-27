import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import logsumexp
from GMM import EM
from read_meta import save_reads
import os

ALPHA = 0.025

class HMM:
    def __init__(self, pi=np.array([0.8, 0.2, 0.0]),
                 trans_mat=np.array([[0.8, 0.2, 0.0],
                                     [0.0, 0.8, 0.2],
                                     [0.0, 0.0, 1]]),
                 mu1=0, sigma1=1, mu2=0, sigma2=1):
        self.pi = pi
        self.trans_mat = trans_mat
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.alpha = ALPHA

    def plot_gauss(self, scores, dataMean, dataSigma, data2Mean, data2Sigma):
        threshold = np.percentile(scores, 100 * (1 - ALPHA))
        scores2 = scores[np.argwhere(scores > threshold)].transpose()[0]
        threshold = round(threshold,2)

        # plot
        x = np.arange(-10, 10, 0.001)
        p1 = stats.norm.pdf(x, dataMean, dataSigma)
        p2 = stats.norm.pdf(x, data2Mean, data2Sigma)
        pMixture = ((1 - ALPHA) * p1) + (ALPHA * p2)
        print("From data, with all data hist, threshold=", threshold)
        print("mu1=", round(dataMean,2),", sigma1=", round(dataSigma,2))
        print("mu2=", round(data2Mean,2),", sigma2=", round(data2Sigma,2))
        txt = 'threshold=' + str(threshold) + '_From_data_with_all_data_hist'
        plt.title(txt +
                  '\nmu1=' + str(round(dataMean,2)) + ', sigma1=' + str(round(dataSigma,2)) +
                  '\nmu2=' + str(round(data2Mean,2)) + ', sigma2=' + str(round(data2Sigma,2)))
        plt.plot(x, p1, label='all data')
        plt.plot(x, p2, c='red', label='brdu group')
        plt.plot(x, pMixture, c='green', label='mixture w/ alpha=2.5%')
        histData = scores.reshape(-1, 1)
        plt.hist(histData, 200, density="True")
        plt.legend()
        plt.savefig('/cs/usr/elafallik/Documents/Project/BrdU/readsDist/GMM/' + txt + '.jpg')
        plt.show()

        # plot
        print("From data, with right data hist, threshold=", threshold)
        txt = 'threshold=' + str(threshold) + '_From_data_with_right_data_hist'
        plt.title(txt +
                  '\nmu1=' + str(round(dataMean,2)) + ', sigma1=' + str(round(dataSigma,2)) +
                  '\nmu2=' + str(round(data2Mean,2)) + ', sigma2=' + str(round(data2Sigma,2)))
        plt.plot(x, p1, label='all data')
        plt.plot(x, p2, c='red', label='brdu group')
        plt.plot(x, pMixture, c='green', label='mixture w/ alpha=2.5%')
        histData = scores2.reshape(-1, 1)
        plt.hist(histData, 200, density="True")
        plt.legend()
        plt.savefig('/cs/usr/elafallik/Documents/Project/BrdU/readsDist/GMM/' + txt + '.jpg')
        plt.show()

        # plot
        print("From data, threshold=", threshold)
        txt = 'threshold=' + str(threshold) + '_From_data'
        plt.title(txt +
                  '\nmu1=' + str(round(dataMean,2)) + ', sigma1=' + str(round(dataSigma,2)) +
                  '\nmu2=' + str(round(data2Mean,2)) + ', sigma2=' + str(round(data2Sigma,2)))
        plt.plot(x, p1, label='all data')
        plt.plot(x, p2, c='red', label='brdu group')
        plt.plot(x, pMixture, c='green', label='mixture w/ alpha=2.5%')
        plt.legend()
        plt.savefig('/cs/usr/elafallik/Documents/Project/BrdU/readsDist/GMM/' + txt + '.jpg')
        plt.show()

        mu1, sigma1, mu2, sigma2, alpha = self.mu1, self.sigma1, self.mu2, self.sigma2, self.alpha
        # plot
        p1 = stats.norm.pdf(x, mu1, sigma1)
        p2 = stats.norm.pdf(x, mu2, sigma2)
        pMixture = ((1 - alpha) * p1) + (alpha * p2)
        print("Alpha not constant, with all data hist, threshold=", threshold)
        print("mu1=", round(mu1,2),", sigma1=", round(sigma1,2))
        print("mu2=", round(mu2,2),", sigma2=", round(sigma2,2))
        txt = 'threshold=' + str(threshold) + '_Alpha_not_constant_with_all_data_hist'
        plt.title(txt +
                  '\nmu1=' + str(round(mu1,2)) + ', sigma1=' + str(round(sigma1,2)) +
                  '\nmu2=' + str(round(mu2,2)) + ', sigma2=' + str(round(sigma2,2)))
        plt.plot(x, p1, label='all data')
        plt.plot(x, p2, c='red', label='brdu group')
        plt.plot(x, pMixture, c='green', label='mixture w/ alpha=2.5%')
        histData = scores.reshape(-1, 1)
        plt.hist(histData, 200, density="True")
        plt.legend()
        plt.savefig('/cs/usr/elafallik/Documents/Project/BrdU/readsDist/GMM/' + txt + '.jpg')
        plt.show()

        print("Alpha not constant, threshold=", threshold)
        txt = 'threshold=' + str(threshold) + '_Alpha_not_constant'
        plt.title(txt +
                  '\nmu1=' + str(round(mu1,2)) + ', sigma1=' + str(round(sigma1,2)) +
                  '\nmu2=' + str(round(mu2,2)) + ', sigma2=' + str(round(sigma2,2)))
        plt.plot(x, p1, label='all data')
        plt.plot(x, p2, c='red', label='brdu group')
        plt.plot(x, pMixture, c='green', label='mixture w/ alpha=2.5%')
        plt.legend()
        plt.savefig('/cs/usr/elafallik/Documents/Project/BrdU/readsDist/GMM/' + txt + '.jpg')
        plt.show()

    def EM(self, X, mu1, sigma1, mu2, sigma2, T, alpha=ALPHA, changeAlpha=False):
        T -= 1
        if T < 0:  # T = num of iterations left, base case
            return mu1, sigma1, mu2, sigma2, alpha

        # E-step
        phi1 = stats.norm.pdf(X, mu1, sigma1)
        phi2 = stats.norm.pdf(X, mu2, sigma2)

        gamma = (alpha * phi2) / ((1 - alpha) * phi1 + alpha * phi2)

        # M-step
        mu1 = np.inner(1 - gamma, X) / np.sum(1 - gamma)
        sigma1 = np.sqrt(np.inner(1 - gamma, (X - mu1)**2) / np.sum(1 - gamma))
        mu2 = np.inner(gamma, X) / np.sum(gamma)
        sigma2 = np.sqrt(np.inner(gamma, (X - mu2)**2) / np.sum(gamma))
        if changeAlpha:
            alpha = np.sum(gamma) / len(X)
        # if T == 0:
        #     print("iter=", T)
        #     print("mu1=", round(mu1,2),", sigma1=", round(sigma1,2))
        #     print("mu2=", round(mu2,2),", sigma2=", round(sigma2,2))
        #     l = np.inner(1 - gamma, np.log(phi1)) + np.inner(gamma, np.log(phi2)) + \
        #         np.sum(np.log(alpha) * (1 - gamma)) + np.sum(np.log(1 - alpha) * gamma)
        #     print("log likelihood=", round(l,2))
        return self.EM(X, mu1, sigma1, mu2, sigma2, T, alpha, changeAlpha)

    def _forward(self, alpha, log_A, log_prob_xn_given_phi):
        # alpha(z1=i) = p(x1|z1=i)*p(z1=i) = pi_i*p(x1|phi_i), i=1,2
        # alpha(zn=i) = p(xn|zn=i)*( alpha(z(n-1)=1) * p(zn=i|z(n-1)=1) + alpha(z(n-1)=2) * p(zn=i|z(n-1)=2) ) =
        #             = p(xn|phi_i(n))*( alpha(z(n-1)=1) * A_i,1(n) + alpha(z(n-1)=2) * A_i,2(n) )
        # a = (alpha + log_A).transpose()
        return log_prob_xn_given_phi + logsumexp((alpha + log_A.transpose()).transpose(), axis=0)

    def _backward(self, beta, log_A, log_prob_xn_given_phi):
        # beta(zN=i) = 1, i=1,2
        # beta(zn=i) = beta(z(n+1)=1) * p(x(n+1)|z(n+1)=1) * p(z(n+1)=1|zn=i) +
        #            + beta(z(n+1)=2) * p(x(n+1)|z(n+1)=2) * p(z(n+1)=2|zn=i) =
        #            = beta(z(n+1)=1) * p(x(n+1)|phi_1(n+1)) * A_1,i(n+1) +
        #            + beta(z(n+1)=2) * p(x(n+1)|phi_2(n+1)) * A_2,i(n+1)
        # return np.log(np.exp(beta[0] + prob_xn_given_phi[0] + A[0]) + np.exp(beta[1] + prob_xn_given_phi[1] + A[1]))
        # a = (beta + log_prob_xn_given_phi + log_A.transpose()).transpose()
        return logsumexp((beta + log_prob_xn_given_phi + log_A).transpose(), axis=0)

    def _E_step(self, log_A, log_prob_X_given_phi, eta, gamma, alpha, beta):
        N = len(log_prob_X_given_phi[0])
        # E-step:
        for n in range(0, N-1):
            log_prob_xn_given_phi = log_prob_X_given_phi.transpose()[n + 1]
            alpha[n+1] = self._forward(alpha[n], log_A, log_prob_xn_given_phi)
            beta[N-(n+1)] = self._backward(beta[N - n], log_A, log_prob_X_given_phi.transpose()[-(n + 1)])

        # p(X) = alpha(zN=1) + alpha(zN=2)
        p_X = logsumexp(alpha[-1])
        # print("log(p(X|params))=log likelihood:")
        # print(p_X)

        for n in range(0, N-1):
            # gamma(zn=i) = p(zn=i|X) = (alpha(zn=i) * beta(zn=i)) / p(X)
            gamma[n] = (alpha[n] + beta[n+1]) - p_X
            log_prob_xn_given_phi = log_prob_X_given_phi.transpose()[n + 1]
            # eta(z(n-1)=i,zn=j) = p(z(n-1)=i,zn=j|X) =
            #                    = (alpha(z(n-1)=i) * p(xn|zn=j) * p(zn=j|z(n-1)=i) * beta(zn=j)) / p(X) =
            #                    = (alpha(z(n-1)=i) * p(xn|phi_j) * A_j,i * beta(zn=j)) / p(X)
            # eta[n-1][i][j] = (alpha[i] * A.transpose()[i][j] * prob_X_given_phi.transpose()[n][j] * beta[j]) / p_X
            eta[n] = ((alpha[n] + log_A.transpose()).transpose() + log_prob_xn_given_phi + beta[n+2]) - p_X

        return gamma, eta

    def _M_step(self, gamma, eta):
        # M-step:
        # pi_i = gamma(zn=i) / (gamma(z1=1) + gamma(z1=2))
        # pi = gamma[0] - np.log(np.sum(np.exp(gamma[0])))
        # m = np.max(gamma[0])
        # pi = gamma[0] - (m + np.log(np.sum(np.exp(gamma[0] - m))))
        pi = gamma[0]
        # A_i,j = sum_k=1,...,N(eta(z(k-1)=i,zk=j)) / (sum_k=1,...,N(eta(z(k-1)=i,zk=1)) + sum_k=1,...,N(eta(z(k-1)=i,zk=2)))
        # B = np.log(np.sum(np.exp(eta), axis=0))
        B = logsumexp(eta, axis=0)
        # A = B / np.log(np.sum(np.exp(B), axis=1))
        log_A = (B.transpose() - logsumexp(B, axis=1).transpose()).transpose()
        return log_A, pi

    def baum_welch(self, log_pi, log_A, log_prob_X_given_phi, T):
        # init: choose some A(1) and pi (what's 0 will stay 0)
        N = len(log_prob_X_given_phi[0])
        alpha = np.zeros((N, 3))
        beta = np.zeros((N + 1, 3))
        eta = np.zeros((N - 1, 3, 3))
        gamma = np.zeros((N - 1, 3))
        for i in range(T):
            # print("T =", i)
            # init alpha: alpha(z1=i) = p(x1|z1=i)*p(z1=i) = pi_i*p(x1|phi_i), i=0,1,2
            alpha[0] = log_pi + log_prob_X_given_phi.transpose()[0]
            # init beta: beta(zN=i) = 1, i=0,1,2
            beta[N] = np.log(np.array([1, 1, 1]))
            gamma, eta = self._E_step(log_A, log_prob_X_given_phi, eta, gamma, alpha, beta)
            temp = log_A
            log_A, log_pi = self._M_step(gamma, eta)
            # print("A:")
            # print(np.exp(log_A))
            # print("diff A:", np.sum(np.abs((np.exp(log_A) - np.exp(temp)).transpose()[0])))
            # print("pi:")
            # print(np.exp(log_pi))
        return log_A, log_pi


    def fit(self, scores, file_name, T_GMM=10, T_HMM=5):
        dataMean = np.mean(scores)
        dataSigma = np.std(scores)
        threshold = np.percentile(scores, 100 * (1 - ALPHA))
        scores2 = scores[np.argwhere(scores > threshold)].transpose()[0]
        data2Mean = np.mean(scores2)
        data2Sigma = np.std(scores2)
        self.mu1, self.sigma1, self.mu2, self.sigma2, self.alpha = \
            self.EM(scores, dataMean, dataSigma, data2Mean, data2Sigma, T_GMM, changeAlpha=True)
        # self.mu1, self.sigma1, self.mu2, self.sigma2, self.alpha = dataMean, dataSigma, data2Mean, data2Sigma, ALPHA

        self.plot_gauss(scores, dataMean, dataSigma, data2Mean, data2Sigma)

        # p(xn|phi_i) = N(xn|mu_i,sigma_i) = stats.norm.pdf(X, mu_i, sigma_i)[n]
        p1 = stats.norm.pdf(scores, self.mu1, self.sigma1)
        p2 = stats.norm.pdf(scores, self.mu2, self.sigma2)
        pMixture = ((1 - self.alpha) * p1) + (self.alpha * p2)
        log_prob_X_given_phi = np.log(np.array([p1, pMixture, p1]))

        # # test http://www.indiana.edu/~iulg/moss/hmmcalculations.pdf
        # self.pi = np.array([0.85, 0.15])
        # self.trans_mat = np.array([[0.3, 0.7],
        #                            [0.1, 0.9]])
        # # state 0: p(A)=.4, p(B)=.6, state 1: p(A)=.5, p(B)=.5
        # # word = ABBA
        # p0 = [0.4, 0.6, 0.6, 0.4]
        # p1 = [0.5, 0.5, 0.5, 0.5]
        # log_prob_X_given_phi = np.log(np.array([p0, p1]))

        log_pi = np.log(self.pi)  # TODO: initialize somehow
        log_A = np.log(self.trans_mat)  # TODO: initialize somehow

        log_A, log_pi = self.baum_welch(log_pi, log_A, log_prob_X_given_phi, T_HMM)
        self.trans_mat = np.exp(log_A)
        self.pi = np.exp(log_pi)

        with open(file_name + "model.txt", "w") as fb:
            fb.write("Model:" + "\n")
            fb.write("trans_mat:" + "\n")
            np.savetxt(fb, self.trans_mat, '%f')
            fb.write("pi:" + "\n")
            np.savetxt(fb, self.pi, '%f')
            fb.write("mu1:" + "\t")
            fb.write(str(self.mu1) + "\n")
            fb.write("sigma1:" + "\t")
            fb.write(str(self.sigma1) + "\n")
            fb.write("mu2:" + "\t")
            fb.write(str(self.mu2) + "\n")
            fb.write("sigma2:" + "\t")
            fb.write(str(self.sigma2) + "\n")
            fb.write("alpha:" + "\t")
            fb.write(str(self.alpha) + "\n")
            fb.write("threshold:" + "\t")
            fb.write(str(threshold) + "\n")
            fb.write("T_GMM:" + "\t")
            fb.write(str(T_GMM) + "\n")
            fb.write("T_HMM:" + "\t")
            fb.write(str(T_HMM) + "\n")
            fb.write("dataMean:" + "\t")
            fb.write(str(dataMean) + "\n")
            fb.write("dataSigma:" + "\t")
            fb.write(str(dataSigma) + "\n")
            fb.write("data2Mean:" + "\t")
            fb.write(str(data2Mean) + "\n")
            fb.write("data2Sigma:" + "\t")
            fb.write(str(data2Sigma) + "\n")

        return np.exp(log_A), np.exp(log_pi)

    def predict(self, observations):
        N = len(self.trans_mat[0])  # num of stats in the model
        T = len(observations)
        log_V = np.zeros((T, N))  # each row = the states values for o_t
        backP = np.zeros((T, N))

        p1 = stats.norm.pdf(observations, self.mu1, self.sigma1)
        p2 = stats.norm.pdf(observations, self.mu2, self.sigma2)
        pMixture = ((1 - self.alpha) * p1) + (self.alpha * p2)
        log_prob_X_given_phi = np.log(np.array([p1, pMixture, p1]).transpose())  # observations likelihood given the state
        log_A = np.log(self.trans_mat.transpose())

        log_V[0] = np.log(self.pi) + log_prob_X_given_phi[0]

        for t in range(1, T):
            temp = log_V[t-1] + log_A + log_prob_X_given_phi[t]
            backP[t] = np.argmax(temp, axis=1)
            log_V[t] = np.max(temp, axis=1)

        temp = log_V[T-1]
        best_pathP = int(np.argmax(temp))
        best_log_prob = temp[best_pathP]
        best_path = np.zeros((T))
        for t in range(1, T+1):
            best_path[T-t] = best_pathP
            best_pathP = int(backP[T-t, best_pathP])

        return best_path, best_log_prob


def plot_decode(fb, scores, best_path, reads_num, dir, reads_names, i, d):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Scores scatter by state
    ax1.set_title("Scores scatter by state")
    ax1.set_ylabel('score')
    ax1.set_xlabel('t')
    ax1.set_ylim((-10, 10))
    ax1.set_xlim((0, len(scores)))
    ax1.plot([0, len(scores)], [0, 0], color='k', linestyle='-', linewidth=0.6)
    ax1.scatter(np.argwhere(best_path == 0.0).transpose()[0], scores[best_path == 0.0], s=0.4, color="red", label="0=T")
    ax1.scatter(np.argwhere(best_path == 1.0).transpose()[0], scores[best_path == 1.0], s=0.4, color="blue", label="1=T+BrdU")
    ax1.scatter(np.argwhere(best_path == 2.0).transpose()[0], scores[best_path == 2.0], s=0.4, color="green", label="2=T")
    ax1.legend(markerscale=5)


    # Read mixture
    ax2.set_title("Read mixture")
    ax2.set_ylabel('density')
    ax2.set_xlabel('score')
    ax2.set_xlim((-10, 10))

    dataMean = np.mean(scores)
    dataSigma = np.std(scores)
    threshold = np.percentile(scores, 100 * (1 - ALPHA))
    score2 = scores[np.argwhere(scores > threshold)].transpose()[0]
    data2Mean = np.mean(score2)
    data2Sigma = np.std(score2)

    mu1, sigma1, mu2, sigma2, alpha = EM(scores, dataMean, dataSigma, data2Mean, data2Sigma, 10, ALPHA, changeAlpha=True)

    y = np.arange(-10, 10, 0.001)
    p1 = stats.norm.pdf(y, mu1, sigma1)
    p2 = stats.norm.pdf(y, mu2, sigma2)
    pMixture = ((1 - alpha) * p1) + (alpha * p2)
    # ax2.title("read " + reads_names[i])
    histData = scores.reshape(-1, 1)
    ax2.hist(histData, 200, density=1, zorder=1, alpha=0.7)
    ax2.plot(y, p1, c='green', label='all data', linewidth=1.5, zorder=3)
    ax2.plot(y, p2, c='yellow', label='brdu group', linewidth=1.5, zorder=2)
    ax2.plot(y, pMixture, c='red', label='mixture w/ alpha=2.5%', linewidth=1.5, zorder=4)
    ax2.legend()

    f.suptitle("read " + reads_names[i])

    plt.savefig("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/readScatter" + reads_num + "/out/" + dir + reads_names[i] + ".jpg", dpi=100)
    # plt.show()

    if len(d) > 1 and d[1.0] > 10:
        print(reads_names[i], "\t", d)
        plt.savefig("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/readScatter" + reads_num + "out/" + dir + "0/" + reads_names[i] + ".jpg", dpi=100)
        fb.write(reads_names[i] + "\t" + str(d) + "\n")

    plt.close(f)


def run_predict(pi, trans_mat, reads_num, dir):
    if not os.path.exists("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/readScatter" + reads_num):
        os.mkdir("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/readScatter" + reads_num)
    if not os.path.exists("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/readScatter" + reads_num + "out/"):
        os.mkdir("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/readScatter" + reads_num + "out/")
    if not os.path.exists("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/readScatter" + reads_num + "out/" + dir):
        os.mkdir("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/readScatter" + reads_num + "out/" + dir)
    if not os.path.exists("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/readScatter" + reads_num + "out/" + dir + "0/"):
        os.mkdir("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/readScatter" + reads_num + "out/" + dir + "0/")

    if not os.path.exists("/cs/usr/elafallik/Documents/Project/BrdU/out/HMM_predict/" + reads_num):
        os.mkdir("/cs/usr/elafallik/Documents/Project/BrdU/out/HMM_predict/" + reads_num)
    if not os.path.exists("/cs/usr/elafallik/Documents/Project/BrdU/out/HMM_predict/" + reads_num + dir):
        os.mkdir("/cs/usr/elafallik/Documents/Project/BrdU/out/HMM_predict/" + reads_num + dir)

    obj = HMM()
    obj.pi = pi
    obj.trans_mat = trans_mat
    obj.mu1 = -2.1738628932512962
    obj.sigma1 = 2.1755906853793423
    obj.mu2 = 6.489612953297095
    obj.sigma2 = 1.5611384198432174
    obj.alpha = 0.0031451602165976414

    obj.alpha = ALPHA

    log_probs = []
    temp = []
    reads_names = np.genfromtxt("/cs/usr/elafallik/Documents/Project/BrdU/data/meta/reads" + reads_num + "reads_names.txt", dtype='str')

    with open("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/readScatter" + reads_num + "out/" + dir + "0/0.txt", "w") as fb:

        fb.write("pi " + str(pi) + "\n")
        fb.write("trans_mat " + str(trans_mat) + "\n")
        fb.write("reads w/ over 10 points of T+BrdU group:\n")

        for i in range(100):
            scores = np.loadtxt("/cs/usr/elafallik/Documents/Project/BrdU/data/meta/reads" + reads_num + "scores_read_" + reads_names[i] + ".txt")
            best_path, best_log_prob = obj.predict(scores)
            temp.append(best_path)
            np.savetxt("/cs/usr/elafallik/Documents/Project/BrdU/out/HMM_predict/" + reads_num + dir + "best_path_read_" + reads_names[i] + ".txt", best_path, fmt='%s')
            log_probs.append(best_log_prob)
            unique, counts = np.unique(best_path, return_counts=True)
            d = dict(zip(unique, counts))

            plot_decode(fb, scores, best_path, reads_num, dir, reads_names, i, d)

            # print(reads_names[i], "\t", d)

    np.savetxt("/cs/usr/elafallik/Documents/Project/BrdU/out/HMM_predict/" + reads_num + dir + "log_probs.txt", np.array(log_probs), fmt='%s')

if __name__ == '__main__':

    # scores, readNum = getDetectData(10000)
    # 10000 reads = 15086606 samples
    # 2000 reads = 3063348 samples
    # np.savetxt("/cs/usr/elafallik/Documents/Project/BrdU/data/meta/scores10000.txt", np.array(scores), fmt='%s')

    reads_num = "400-499/"
    dir = "state2-0-2-8_pi-8-2-0/"
    # dir = "pi-8-2-0/"
    #
    # pi = np.array([0.800604, 0.199396, 0.0])
    # trans_mat = np.array([[0.80000153, 0.19999847, 0],
    #                       [0, 0.77519868, 0.22480132],
    #                       [0, 0, 1]])


    pi = np.array([0.807092, 0.192908, 0.0])
    trans_mat = np.array([[0.807648, 0.192352, 0],
                          [0, 0.803630, 0.196370],
                          [0, 0.203381, 0.796619]])

    # save_reads(5000, 6000)


    # reads_names = np.genfromtxt("/cs/usr/elafallik/Documents/Project/BrdU/data/meta/reads" + reads_num + "reads_names.txt", dtype='str')
    #
    # if not os.path.exists("/cs/usr/elafallik/Documents/Project/BrdU/out/HMM_predict/" + reads_num):
    #     os.mkdir("/cs/usr/elafallik/Documents/Project/BrdU/out/HMM_predict/" + reads_num)
    #
    # obj = HMM()
    # scores = np.array([])
    # for i in range(100):
    #     temp = np.loadtxt("/cs/usr/elafallik/Documents/Project/BrdU/data/meta/reads" + reads_num + "scores_read_" + reads_names[i] + ".txt")
    #     scores = np.append(scores, temp)
    #
    # obj.fit(scores, "/cs/usr/elafallik/Documents/Project/BrdU/out/HMM_predict/" + reads_num)

    run_predict(pi, trans_mat, reads_num, dir)



    a = 2



















