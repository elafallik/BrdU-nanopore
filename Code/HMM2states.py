import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import logsumexp
import os
import re

ALPHA = 0.025
PATH = '/cs/usr/elafallik/Documents/Project/temp/'

############################################
################# get data #################
############################################
PTR_DETECT_6MER = "(\d+)\t(-?\d.?\d+)\t[CATG]{6}\t[CATG]{6}" # 6mer from .detect file g1: location, g2: score
PTR_DETECT_START_READ = "^\>(.*) chr(.*):(\d+)-\d+" # g1: read name, g2: chr num (roman), g3: start point
PTR_ORIDB = "chr(.*)\t(\d+)\t(\d+)\tchr\d+:\d+-\d+\t\d"  # g1: chr num (roman), g2: start, g3: end

# get list, every element is the log likelihood scores for a read
def get_detect_data(file, end=155945, byRead=False):
    scores = []
    readNames = []
    locations = []
    readNum = -1
    with open(file, 'r') as fp:
        line = fp.readline()
        while line and readNum < end:  # total num of reads: 155945
            x = re.search(PTR_DETECT_6MER, line)
            y = re.search(PTR_DETECT_START_READ, line)
            if x is not None:
                if byRead:
                    scores[readNum] += [float(x.group(2))]
                    locations[readNum] += [int(x.group(1))]
                else:
                    scores.append(float(x.group(2)))
                    locations.append(int(x.group(1)))
            elif y is not None:
                readNum += 1
                if byRead:
                    scores.append([])
                    locations.append([])
                    readNames.append(np.array([y.group(1), y.group(2), y.group(3)]))
            line = fp.readline()
    return np.array(scores[:-1]), readNum, readNames, np.array(locations)

def get_reads(start, end, path_res, save_data, save_read_names):
    scores, readNum, readNames, locations = get_detect_data('/cs/zbio/tommy/nanopore/Seq_Jun19_yeast/Itamar_Simon_gDNAyeast-2-5min/2-5min_BrdU.detect', end, True)
    # scores, readNum, readNames, locations = get_detect_data('/cs/zbio/tommy/nanopore/Seq_Jun19_yeast/Itamar_Simon_gDNAyeast-5min/Itamar_Simon_gDNAyeast-5min/20190605_1029_GA30000_FAK67655_241b1c9f/5min_BrdU.detect', end, True)

    if save_data:
        if not os.path.exists(path_res + "reads" + str(start) + "-" + str(end - 1) + "/"):
            os.mkdir(path_res + "reads" + str(start) + "-" + str(end - 1) + "/")

        for i in range(start, end):
            np.savetxt(path_res + "reads" + str(start) + "-" + str(end - 1) + "/scores_read_" + readNames[i][0] + ".txt", np.array(scores[i]), fmt='%s')
            np.savetxt(path_res + "reads" + str(start) + "-" + str(end - 1) + "/loc_read_" + readNames[i][0] + ".txt", np.array(locations[i]), fmt='%s')

        np.savetxt(path_res + "reads" + str(start) + "-" + str(end - 1) + "/reads_names.txt", np.array(readNames[start:end]), fmt='%s')
        return None, None, None
    else:
        if save_read_names:
            np.savetxt(path_res + "reads" + str(start) + "-" + str(end - 1) + "/reads_names.txt", np.array(readNames[start:end]), fmt='%s')
        return np.array(scores), np.array(locations), np.array(readNames)


# get list, every element is the log likelihood scores for a read
def get_Ori_data(file_path):
    res = {}
    with open(file_path, 'r') as fp:
        line = fp.readline()
        while line:
            x = re.search(PTR_ORIDB, line)
            if x is not None:
                if x.group(1) not in res:
                    res[x.group(1)] = [[int(x.group(2)), int(x.group(3))]]
                else:
                    res[x.group(1)].append([int(x.group(2)), int(x.group(3))])
            line = fp.readline()
    return res


def get_Ori():
    confirmed = get_Ori_data("/cs/usr/elafallik/Documents/Project/BrdU/data/OriDB/origin_yeast_OriDB_sacCe3_Confirmed.bed")
    likely = get_Ori_data("/cs/usr/elafallik/Documents/Project/BrdU/data/OriDB/origin_yeast_OriDB_sacCe3_Likely.bed")
    dubious = get_Ori_data("/cs/usr/elafallik/Documents/Project/BrdU/data/OriDB/origin_yeast_OriDB_sacCe3_Dubious.bed")

    return confirmed, likely, dubious

############################################


class HMM:
    def __init__(self, pi=np.array([0.8, 0.2]),
                 trans_mat=np.array([[0.8, 0.2], [0.2, 0.8]]),
                 mu1=0, sigma1=1, mu2=0, sigma2=1):
        self.pi = pi
        self.trans_mat = trans_mat
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.alpha = ALPHA

    def _plot_gauss_helper(self, txt, file_path, scores, mu1, sigma1, mu2, sigma2):
        x = np.arange(-10, 10, 0.001)
        p1 = stats.norm.pdf(x, mu1, sigma1)
        p2 = stats.norm.pdf(x, mu2, sigma2)
        pMixture = ((1 - ALPHA) * p1) + (ALPHA * p2)
        plt.title(txt +
                  '\nmu1=' + str(round(mu1, 2)) + ', sigma1=' + str(round(sigma1, 2)) +
                  '\nmu2=' + str(round(mu2,2)) + ', sigma2=' + str(round(sigma2,2)))
        plt.plot(x, p1, label='all data')
        plt.plot(x, p2, c='red', label='brdu group')
        plt.plot(x, pMixture, c='green', label='mixture w/ alpha=' + str(ALPHA * 100) + '%')
        plt.xlabel('scores')
        plt.ylabel('density')
        if scores is not None:
            histData = scores.reshape(-1, 1)
            plt.hist(histData, 200, density="True")
        plt.legend()
        plt.savefig(file_path + '.jpg')
        plt.close()

    def _plot_gauss(self, file_path, scores, dataMean, dataSigma, data2Mean, data2Sigma):

        threshold = np.percentile(scores, 100 * (1 - ALPHA))
        scores2 = scores[np.argwhere(scores > threshold)].transpose()[0]

        txt = 'Initial params with data hist'
        self._plot_gauss_helper(txt, file_path + 'Initial_params_with_data_hist', scores, dataMean, dataSigma, data2Mean, data2Sigma)

        txt = 'Initial params with BrdU group hist'
        self._plot_gauss_helper(txt, file_path + 'Initial_params_with_BrdU_group_hist', scores2, dataMean, dataSigma, data2Mean, data2Sigma)

        txt = 'Initial_params'
        self._plot_gauss_helper(txt, file_path + 'Initial_params', None, dataMean, dataSigma, data2Mean, data2Sigma)

        mu1, sigma1, mu2, sigma2, alpha = self.mu1, self.sigma1, self.mu2, self.sigma2, self.alpha

        txt = 'Trained params with data hist'
        self._plot_gauss_helper(txt, file_path + 'Trained_params_with_data_hist', scores, mu1, sigma1, mu2, sigma2)

        txt = 'Trained params'
        self._plot_gauss_helper(txt, file_path + 'Trained_params', None, mu1, sigma1, mu2, sigma2)

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
        return log_prob_xn_given_phi + logsumexp((alpha + log_A.transpose()).transpose(), axis=0)

    def _backward(self, beta, log_A, log_prob_xn_given_phi):
        # beta(zN=i) = 1, i=1,2
        # beta(zn=i) = beta(z(n+1)=1) * p(x(n+1)|z(n+1)=1) * p(z(n+1)=1|zn=i) +
        #            + beta(z(n+1)=2) * p(x(n+1)|z(n+1)=2) * p(z(n+1)=2|zn=i) =
        #            = beta(z(n+1)=1) * p(x(n+1)|phi_1(n+1)) * A_1,i(n+1) +
        #            + beta(z(n+1)=2) * p(x(n+1)|phi_2(n+1)) * A_2,i(n+1)
        # return np.log(np.exp(beta[0] + prob_xn_given_phi[0] + A[0]) + np.exp(beta[1] + prob_xn_given_phi[1] + A[1]))
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
        pi = gamma[0]
        # A_i,j = sum_k=1,...,N(eta(z(k-1)=i,zk=j)) / (sum_k=1,...,N(eta(z(k-1)=i,zk=1)) + sum_k=1,...,N(eta(z(k-1)=i,zk=2)))
        B = logsumexp(eta, axis=0)
        log_A = (B.transpose() - logsumexp(B, axis=1).transpose()).transpose()
        return log_A, pi

    def _baum_welch(self, log_pi, log_A, log_prob_X_given_phi, T):
        # init: choose some A(1) and pi (what's 0 will stay 0)
        N = len(log_prob_X_given_phi[0])
        K = len(log_prob_X_given_phi)
        alpha = np.zeros((N, K))
        beta = np.zeros((N + 1, K))
        eta = np.zeros((N - 1, K, K))
        gamma = np.zeros((N - 1, K))
        for i in range(T):
            # init alpha: alpha(z1=i) = p(x1|z1=i)*p(z1=i) = pi_i*p(x1|phi_i), i=0,1,2
            alpha[0] = log_pi + log_prob_X_given_phi.transpose()[0]
            # init beta: beta(zN=i) = 1, i=0,1,2
            beta[N] = np.log(np.array([1, 1]))
            gamma, eta = self._E_step(log_A, log_prob_X_given_phi, eta, gamma, alpha, beta)
            log_A, log_pi = self._M_step(gamma, eta)
        return log_A, log_pi


    def fit(self, scores, file_path, T_GMM=10, T_HMM=5):
        dataMean = np.mean(scores)
        dataSigma = np.std(scores)
        threshold = np.percentile(scores, 100 * (1 - ALPHA))
        scores2 = scores[np.argwhere(scores > threshold)].transpose()[0]
        data2Mean = np.mean(scores2)
        data2Sigma = np.std(scores2)
        self.mu1, self.sigma1, self.mu2, self.sigma2, self.alpha = \
            self.EM(scores, dataMean, dataSigma, data2Mean, data2Sigma, T_GMM, changeAlpha=True)

        self._plot_gauss(file_path, scores, dataMean, dataSigma, data2Mean, data2Sigma)

        # p(xn|phi_i) = N(xn|mu_i,sigma_i) = stats.norm.pdf(X, mu_i, sigma_i)[n]
        p1 = stats.norm.pdf(scores, self.mu1, self.sigma1)
        p2 = stats.norm.pdf(scores, self.mu2, self.sigma2)
        pMixture = ((1 - self.alpha) * p1) + (self.alpha * p2)
        log_prob_X_given_phi = np.log(np.array([p1, pMixture]))

        # # test http://www.indiana.edu/~iulg/moss/hmmcalculations.pdf
        # self.pi = np.array([0.85, 0.15])
        # self.trans_mat = np.array([[0.3, 0.7],
        #                            [0.1, 0.9]])
        # # state 0: p(A)=.4, p(B)=.6, state 1: p(A)=.5, p(B)=.5
        # # word = ABBA
        # p0 = [0.4, 0.6, 0.6, 0.4]
        # p1 = [0.5, 0.5, 0.5, 0.5]
        # log_prob_X_given_phi = np.log(np.array([p0, p1]))

        log_pi = np.log(self.pi)
        log_A = np.log(self.trans_mat)

        log_A, log_pi = self._baum_welch(log_pi, log_A, log_prob_X_given_phi, T_HMM)
        self.trans_mat = np.exp(log_A)
        self.pi = np.exp(log_pi)

        with open(file_path + "model.txt", "w") as fb:
            fb.write("Model:" + "\n")
            fb.write("trans_mat:" + "\n")
            np.savetxt(fb, self.trans_mat, '%f')
            fb.write("pi:" + "\n")
            np.savetxt(fb, self.pi, '%f')
            fb.write("mu1:" + "\n")
            fb.write(str(self.mu1) + "\n")
            fb.write("sigma1:" + "\n")
            fb.write(str(self.sigma1) + "\n")
            fb.write("mu2:" + "\n")
            fb.write(str(self.mu2) + "\n")
            fb.write("sigma2:" + "\n")
            fb.write(str(self.sigma2) + "\n")
            fb.write("alpha:" + "\n")
            fb.write(str(self.alpha) + "\n")
            fb.write("threshold:" + "\n")
            fb.write(str(threshold) + "\n")
            fb.write("T_GMM:" + "\n")
            fb.write(str(T_GMM) + "\n")
            fb.write("T_HMM:" + "\n")
            fb.write(str(T_HMM) + "\n")
            fb.write("dataMean:" + "\n")
            fb.write(str(dataMean) + "\n")
            fb.write("dataSigma:" + "\n")
            fb.write(str(dataSigma) + "\n")
            fb.write("data2Mean:" + "\n")
            fb.write(str(data2Mean) + "\n")
            fb.write("data2Sigma:" + "\n")
            fb.write(str(data2Sigma) + "\n")

        return np.exp(log_A), np.exp(log_pi)

    def predict(self, observations):
        N = len(self.trans_mat[0])  # num of states in the model
        T = len(observations)
        log_V = np.zeros((T, N))  # each row = the states values for o_t
        backP = np.zeros((T, N))

        p1 = stats.norm.pdf(observations, self.mu1, self.sigma1)
        p2 = stats.norm.pdf(observations, self.mu2, self.sigma2)
        pMixture = ((1 - self.alpha) * p1) + (self.alpha * p2)
        log_prob_X_given_phi = np.log(np.array([p1, pMixture]).transpose())  # observations likelihood given the state
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


############################################
#################### run ###################
############################################

def plot_decode(model, scores, locations, best_path, root, read_name):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Scores scatter by state
    ax1.set_title("Scores scatter by state")
    ax1.set_ylabel('score')
    ax1.set_xlabel('t')
    ax1.set_ylim((-10, 10))
    s = locations[0]
    e = locations[-1] + len(scores)
    ax1.set_xlim((s, e))
    ax1.plot([s, e], [0, 0], color='k', linestyle='-', linewidth=0.6)
    idx = np.argwhere(best_path == 0.0).transpose()[0]
    ax1.scatter(locations[idx] + idx, scores[idx], s=0.4, color="green", label="0=T")
    idx = np.argwhere(best_path == 1.0).transpose()[0]
    ax1.scatter(locations[idx] + idx, scores[idx], s=0.4, color="blue", label="1=T+BrdU")
    leg1 = ax1.legend(loc='lower left', markerscale=5)
    ax1.legend(markerscale=5)

    confirmed, likely, dubious = get_Ori()
    chr = read_name[1]
    read_start = int(read_name[2])
    if chr in confirmed:
        for ori in confirmed[chr]:
            if (ori[0] >= s and ori[0] < e) or (ori[1] > s and ori[1] <= e):
                ax1.plot([ori[0], ori[0]], [-10, 10], linewidth=0.6, color="red", label="confirmed")
                # ax1.plot([ori[1], ori[1]], [-10, 10], linewidth=0.6, color="red")
                # ax1.plot([ori[0], ori[1]], [5, 5], linewidth=0.6, color="red")
            elif ori[0] > e:
                break
    if chr in likely:
        for ori in likely[chr]:
            if (ori[0] >= s and ori[0] < e) or (ori[1] > s and ori[1] <= e):
                ax1.plot([ori[0], ori[0]], [-10, 10], linewidth=0.6, color="orange", label="likely")
                # ax1.plot([ori[1], ori[1]], [-10, 10], linewidth=0.6, color="orange")
                # ax1.plot([ori[0], ori[1]], [5, 5], linewidth=0.6, color="orange")
            elif ori[0] > e:
                break
    if chr in dubious:
        for ori in dubious[chr]:
            if (ori[0] >= s and ori[0] < e) or (ori[1] > s and ori[1] <= e):
                ax1.plot([ori[0], ori[0]], [-10, 10], linewidth=0.6, color="yellow", label="dubious")
                # ax1.plot([ori[1], ori[1]], [-10, 10], linewidth=0.6, color="yellow")
                # ax1.plot([ori[0], ori[1]], [5, 5], linewidth=0.6, color="yellow")
            elif ori[0] > e:
                break

    leg2 = ax1.legend([ax1.plot([0, 0], [0, 0], color="red")[0],
                       ax1.plot([0, 0], [0, 0], color="orange")[0],
                       ax1.plot([0, 0], [0, 0], color="yellow")[0]],
                      ['confirmed', 'likely', 'dubious'], loc='upper right', title="Origin")
    ax1.add_artist(leg1)
    ax1.add_artist(leg2)

    # Read mixture
    ax2.set_title("Read with the model mixture")
    ax2.set_ylabel('density')
    ax2.set_xlabel('score')
    ax2.set_xlim((-10, 10))
    mu1, sigma1, mu2, sigma2, alpha = model.mu1, model.sigma1, model.mu2, model.sigma2, model.alpha
    y = np.arange(-10, 10, 0.001)
    p1 = stats.norm.pdf(y, mu1, sigma1)
    p2 = stats.norm.pdf(y, mu2, sigma2)
    pMixture = ((1 - alpha) * p1) + (alpha * p2)
    histData = scores.reshape(-1, 1)
    ax2.hist(histData, 200, density=1, zorder=1, alpha=0.7)
    ax2.plot(y, p1, c='green', label='all data', linewidth=1.5, zorder=3)
    ax2.plot(y, p2, c='yellow', label='brdu group', linewidth=1.5, zorder=2)
    ax2.plot(y, pMixture, c='red', label='mixture w/ alpha=2.5%', linewidth=1.5, zorder=4)
    ax2.legend()

    f.suptitle("read " + read_name[0] + ", chr=" + chr)
    plt.savefig(root + "/readScatter/" + read_name[0] + ".jpg", dpi=100)
    plt.close(f)


def run_fit(reads_num, train_scores, T_GMM=10, T_HMM=5, sample_size=100):
    reads_names = np.genfromtxt(PATH + "data/reads" + reads_num + "reads_names.txt", dtype='str')

    obj = HMM()
    if train_scores is None:
        train_scores = np.array([])
        for i in range(sample_size):
            temp = np.loadtxt(PATH + "data/reads" + reads_num + "scores_read_" + reads_names[i][0] + ".txt")
            train_scores = np.append(train_scores, temp)
    else:
        temp = train_scores
        train_scores = np.array([])
        for i in range(sample_size):
            train_scores = np.append(train_scores, np.array(temp[i]))
    obj.fit(train_scores, PATH + "out/HMM2states/" + reads_num, T_GMM=T_GMM, T_HMM=T_HMM)
    return obj


def run_predict(model, test_scores, test_locations, reads_names, res_path, reads_num, sample_size=100):

    if not os.path.exists(res_path + "readScatter/"):
        os.mkdir(res_path + "readScatter/")
    if not os.path.exists(res_path + "predict_res/"):
        os.mkdir(res_path + "predict_res/")

    model.alpha = ALPHA

    log_probs = []
    temp = []
    temp2 = []
    if reads_names is None:
        reads_names = np.genfromtxt(PATH + "data/reads" + reads_num + "reads_names.txt", dtype='str')

    with open(res_path + "readScatter/0.txt", "w") as fb:
        fb.write("pi " + str(model.pi) + "\n")
        fb.write("trans_mat " + str(model.trans_mat) + "\n")
    for i in range(sample_size):
        if test_scores is None:
            test_score = np.loadtxt(PATH + "data/reads" + reads_num + "scores_read_" + reads_names[i][0] + ".txt")
        else:
            test_score = np.array(test_scores[i])
        if test_locations is None:
            test_location = np.loadtxt(PATH + "data/reads" + reads_num + "loc_read_" + reads_names[i][0] + ".txt")
        else:
            test_location = np.array(test_locations[i])
        best_path, best_log_prob = model.predict(test_score)
        a = best_path[0]
        count = 0
        for j in range(1, len(best_path)):
            if best_path[j] == 0.0 and a == 1.0:  # T+BrdU -> T state
                temp2.append(count)
                count = 0
            elif best_path[j] == 1.0:  # T+BrdU state
                count += 1
            a = best_path[j]

        temp.append(best_path)
        np.savetxt(res_path + "predict_res/" + reads_names[i][0] + ".txt", best_path, fmt='%s')
        log_probs.append(best_log_prob)
        plot_decode(model, test_score, test_location, best_path, res_path, reads_names[i])


    np.savetxt(res_path + "predict_res/" + "log_probs.txt", np.array(log_probs), fmt='%s')
    return temp2


def run(start, end, train_start, train_end, T_GMM=10, T_HMM=5, save_data=True):

    reads_num = str(start) + "-" + str(end-1) + "/"
    train_reads_num = str(train_start) + "-" + str(train_end-1) + "/"

    if not os.path.exists(PATH + "data/"):
        os.mkdir(PATH + "data/")
    if not os.path.exists(PATH + "out/"):
        os.mkdir(PATH + "out/")
    if not os.path.exists(PATH + "out/HMM2states/"):
        os.mkdir(PATH + "out/HMM2states/")
    if not os.path.exists(PATH + "out/HMM2states/" + train_reads_num):
        os.mkdir(PATH + "out/HMM2states/" + train_reads_num)

    path_predict_res = PATH + "out/predict2states/"
    if not os.path.exists(path_predict_res):
        os.mkdir(path_predict_res)
    if not os.path.exists(path_predict_res + reads_num):
        os.mkdir(path_predict_res + reads_num)



    # get test data
    test_scores, test_locations, reads_names = None, None, None
    if (not os.path.exists(PATH + "data/reads" + reads_num + "reads_names.txt")) or len(os.listdir(PATH + "data/reads" + reads_num)) == 1:
        if not os.path.exists(PATH + "data/reads" + reads_num):
            os.mkdir(PATH + "data/reads" + reads_num)
        test_scores, test_locations, reads_names = get_reads(start, end, PATH + "data/", True, True)

    if not os.path.exists(PATH + "out/HMM2states/" + train_reads_num + "model.txt"):  # run_fit
        # get run_fit data
        train_scores = None
        if (not os.path.exists(PATH + "data/reads" + train_reads_num + "reads_names.txt")) or len(os.listdir(PATH + "data/reads" + train_reads_num)) == 1:
            if not os.path.exists(PATH + "data/reads" + train_reads_num):
                os.mkdir(PATH + "data/reads" + train_reads_num)
            train_scores, _, _ = get_reads(train_start, train_end, PATH + "data/", save_data, True)
        model = run_fit(train_reads_num, train_scores, T_GMM=T_GMM, T_HMM=T_HMM)
    else:
        model = HMM()
        with open(PATH + "out/HMM2states/" + train_reads_num + "model.txt", "r") as fb:
            fb.readline()
            fb.readline()
            model.trans_mat = np.loadtxt(fb, ndmin=2, max_rows=2)
            fb.readline()
            model.pi = np.loadtxt(fb, ndmin=1, max_rows=2)
            fb.readline()
            model.mu1 = float(fb.readline())
            fb.readline()
            model.sigma1 = float(fb.readline())
            fb.readline()
            model.mu2 = float(fb.readline())
            fb.readline()
            model.sigma2 = float(fb.readline())
            fb.readline()
            model.alpha = float(fb.readline())

    temp = run_predict(model, test_scores, test_locations, reads_names, path_predict_res + reads_num, reads_num)

    histData = np.array(temp).reshape(-1, 1)
    plt.hist(histData, 20, density=1)

    plt.title("")
    # plt.savefig(root + "/readScatter/" + read_name[0] + ".jpg", dpi=100)
    plt.show()


if __name__ == '__main__':

    start, end = 1300, 1400  # at least 100 samples, or change the param in run_predict
    train_start, train_end = 0, 1000
    run(start, end, train_start, train_end, save_data=False)


    a = 2

