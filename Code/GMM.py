import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import re
PTR_DETECT_6MER = "(\d+)\t(-?\d.?\d+)\t([CATG]{6})\t([CATG]{6})" # 6mer from .detect file
PTR_DETECT_START_READ = "^\>"
ALPHA = 0.025


# get list, every element is the log likelihood scores for a read
def getDetectData(end=155945, byRead=False):
    scores = []
    readNames = []
    readNum = -1
    with open('/vol/sci/bio/data/itamar.simon/itamarsi/storage/Nanopore_Oriya/Seq_Jun19_yeast/Itamar_Simon_gDNAyeast-2-5min/'
              'Itamar_Simon_gDNAyeast-2-5min/20190605_1029_GA10000_FAK67866_28476ad6/2-5min_BrdU.detect', 'r') as fp:
        line = fp.readline()
        while line and readNum < end:  # total num of reads: 155945
            x = re.search(PTR_DETECT_6MER, line)
            y = re.search(PTR_DETECT_START_READ, line)
            if x is not None:
                if byRead:
                    scores[readNum] += [float(x.group(2))]
                else:
                    scores.append(float(x.group(2)))
            elif y is not None:
                readNum += 1
                if byRead:
                    scores.append([])
                    readNames.append(line.split(" ")[0])
            line = fp.readline()
    return np.array(scores[:-1]), readNum, readNames

# Model:
# GMM, with 2 components: (1-ALPHA) * N(mu, sigma) + ALPHA * N(mu2, sigma2),
# ALPHA is a small constant.
# So model parameters: mu1, sigma1, mu2, sigma2. Hyperparameter: ALPHA
# We have a sequence of observations X.
# We want to find the the model parameters that maximize the likelihood of the data X.
#
# EM algorithm:
#
#
#
#
#


def EM(X, mu1, sigma1, mu2, sigma2, T, alpha, changeAlpha=False):
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
    #     print("alpha=", alpha)
    return EM(X, mu1, sigma1, mu2, sigma2, T, alpha)


if __name__ == '__main__':

    scores, readNum, readNames = getDetectData(500, True)
    M = max([max(s) for s in scores[400:500]])
    m = min([min(s) for s in scores[400:500]])
    for i in range(400, 500):
        score = scores[i]
        plt.title("read " + readNames[i])
        plt.ylim((m, M))
        plt.plot([0, len(score)], [0, 0], color='k', linestyle='-', linewidth=0.6)
        x = np.arange(0, len(score), 1)
        plt.scatter(x, score, s=0.4)
        plt.savefig("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/readScatter400-499/" + readNames[i] + ".jpg")
        plt.show()


    # scores, readNum, readNames = getDetectData(500, True)
    # j = 0
    # x = np.arange(-10, 10, 0.001)
    # for i in range(400, 500):
    #     score = np.array(scores[i])
    #     histData = score.reshape(-1, 1)
    #     plt.hist(histData, 200, density="True")
    #
    #     dataMean = np.mean(score)
    #     dataSigma = np.std(score)
    #     threshold = np.percentile(score, 100 * (1 - ALPHA))
    #     score2 = score[np.argwhere(score > threshold)].transpose()[0]
    #     data2Mean = np.mean(score2)
    #     data2Sigma = np.std(score2)
    #
    #     mu1, sigma1, mu2, sigma2, alpha = EM(score, dataMean, dataSigma, data2Mean, data2Sigma, 10, ALPHA, changeAlpha=True)
    #
    #     if mu1 > mu2:
    #         j+=1
    #
    #     p1 = stats.norm.pdf(x, mu1, sigma1)
    #     p2 = stats.norm.pdf(x, mu2, sigma2)
    #     pMixture = ((1 - alpha) * p1) + (alpha * p2)
    #     plt.title("read " + readNames[i])
    #     plt.plot(x, p1, label='all data')
    #     plt.plot(x, p2, c='yellow', label='brdu group')
    #     plt.plot(x, pMixture, c='red', label='mixture w/ alpha=2.5%')
    #     histData = score.reshape(-1, 1)
    #     plt.hist(histData, 200, density="True")
    #     plt.legend()
    #     plt.savefig("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/readHist400-499/" + readNames[i] + ".jpg")
    #     plt.show()
    #
    #     np.savetxt("/cs/usr/elafallik/Documents/Project/BrdU/data/meta/reads400-499/scores_read_" + readNames[i] + ".txt", np.array(scores[i]), fmt='%s')
    # print("num of bad reads: ", j)
    # np.savetxt("/cs/usr/elafallik/Documents/Project/BrdU/data/meta/reads400-499/reads_names.txt", np.array(readNames[400:500]), fmt='%s')

    # scores, readNum = getDetectData(100)
    # np.savetxt("/cs/usr/elafallik/Documents/Project/BrdU/data/meta/scores100.txt", np.array(scores), fmt='%s')
    # scores = np.loadtxt("/cs/usr/elafallik/Documents/Project/BrdU/data/meta/scores2000.txt")
    # dataMean = np.mean(scores)
    # dataSigma = np.std(scores)
    # threshold = np.percentile(scores, 100 * (1 - ALPHA))
    # scores2 = scores[np.argwhere(scores > threshold)].transpose()[0]
    # data2Mean = np.mean(scores2)
    # data2Sigma = np.std(scores2)
    # threshold = round(threshold,2)
    #
    # # plot
    # x = np.arange(-10, 10, 0.001)
    # p1 = stats.norm.pdf(x, dataMean, dataSigma)
    # p2 = stats.norm.pdf(x, data2Mean, data2Sigma)
    # pMixture = ((1 - ALPHA) * p1) + (ALPHA * p2)
    # print("From data, with all data hist, threshold=", threshold)
    # print("mu1=", round(dataMean,2),", sigma1=", round(dataSigma,2))
    # print("mu2=", round(data2Mean,2),", sigma2=", round(data2Sigma,2))
    # txt = 'threshold=' + str(threshold) + '_From_data_with_all_data_hist'
    # plt.title(txt +
    #           '\nmu1=' + str(round(dataMean,2)) + ', sigma1=' + str(round(dataSigma,2)) +
    #           '\nmu2=' + str(round(data2Mean,2)) + ', sigma2=' + str(round(data2Sigma,2)))
    # plt.plot(x, p1, label='all data')
    # plt.plot(x, p2, c='red', label='brdu group')
    # plt.plot(x, pMixture, c='green', label='mixture w/ alpha=5%')
    # histData = scores.reshape(-1, 1)
    # plt.hist(histData, 200, density="True")
    # plt.savefig('/cs/usr/elafallik/Documents/Project/BrdU/readsDist/GMM/' + txt + '.jpg')
    # plt.legend()
    # plt.show()
    #
    # # plot
    # print("From data, with right data hist, threshold=", threshold)
    # txt = 'threshold=' + str(threshold) + '_From_data_with_right_data_hist'
    # plt.title(txt +
    #           '\nmu1=' + str(round(dataMean,2)) + ', sigma1=' + str(round(dataSigma,2)) +
    #           '\nmu2=' + str(round(data2Mean,2)) + ', sigma2=' + str(round(data2Sigma,2)))
    # plt.plot(x, p1, label='all data')
    # plt.plot(x, p2, c='red', label='brdu group')
    # plt.plot(x, pMixture, c='green', label='mixture w/ alpha=5%')
    # histData = scores2.reshape(-1, 1)
    # plt.hist(histData, 200, density="True")
    # plt.savefig('/cs/usr/elafallik/Documents/Project/BrdU/readsDist/GMM/' + txt + '.jpg')
    # plt.legend()
    # plt.show()
    #
    # # plot
    # print("From data, threshold=", threshold)
    # txt = 'threshold=' + str(threshold) + '_From_data'
    # plt.title(txt +
    #           '\nmu1=' + str(round(dataMean,2)) + ', sigma1=' + str(round(dataSigma,2)) +
    #           '\nmu2=' + str(round(data2Mean,2)) + ', sigma2=' + str(round(data2Sigma,2)))
    # plt.plot(x, p1, label='all data')
    # plt.plot(x, p2, c='red', label='brdu group')
    # plt.plot(x, pMixture, c='green', label='mixture w/ alpha=5%')
    # plt.savefig('/cs/usr/elafallik/Documents/Project/BrdU/readsDist/GMM/' + txt + '.jpg')
    # plt.legend()
    # plt.show()
    #
    # mu1, sigma1, mu2, sigma2, alpha = EM(scores, dataMean, dataSigma, data2Mean, data2Sigma,
    #                                       ALPHA, 100, changeAlpha=True)
    # # plot
    # p1 = stats.norm.pdf(x, mu1, sigma1)
    # p2 = stats.norm.pdf(x, mu2, sigma2)
    # pMixture = ((1 - ALPHA) * p1) + (ALPHA * p2)
    # print("Alpha constant, with all data hist, threshold=", threshold)
    # print("mu1=", round(mu1,2),", sigma1=", round(sigma1,2))
    # print("mu2=", round(mu2,2),", sigma2=", round(sigma2,2))
    # txt = 'threshold=' + str(threshold) + '_Alpha_constant_with_all_data_hist'
    # plt.title(txt +
    #           '\nmu1=' + str(round(mu1,2)) + ', sigma1=' + str(round(sigma1,2)) +
    #           '\nmu2=' + str(round(mu2,2)) + ', sigma2=' + str(round(sigma2,2)))
    # plt.plot(x, p1, label='all data')
    # plt.plot(x, p2, c='red', label='brdu group')
    # plt.plot(x, pMixture, c='green', label='mixture w/ alpha=5%')
    # histData = scores.reshape(-1, 1)
    # plt.hist(histData, 200, density="True")
    # plt.savefig('/cs/usr/elafallik/Documents/Project/BrdU/readsDist/GMM/' + txt + '.jpg')
    # plt.legend()
    # plt.show()
    #
    # print("Alpha constant, threshold=", threshold)
    # txt = 'threshold=' + str(threshold) + '_Alpha_constant'
    # plt.title(txt +
    #           '\nmu1=' + str(round(mu1,2)) + ', sigma1=' + str(round(sigma1,2)) +
    #           '\nmu2=' + str(round(mu2,2)) + ', sigma2=' + str(round(sigma2,2)))
    # plt.plot(x, p1, label='all data')
    # plt.plot(x, p2, c='red', label='brdu group')
    # plt.plot(x, pMixture, c='green', label='mixture w/ alpha=5%')
    # plt.savefig('/cs/usr/elafallik/Documents/Project/BrdU/readsDist/GMM/' + txt + '.jpg')
    # plt.legend()
    # plt.show()


a = 2