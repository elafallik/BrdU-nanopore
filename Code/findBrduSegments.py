import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from hmmlearn import hmm

import re
PTR_DETECT_6MER = "(\d+)\t(-?\d.?\d+)\t([CATG]{6})\t([CATG]{6})" # 6mer from .detect file
PTR_DETECT_START_READ = "^\>"
ALPHA = 0.05


# get list, every element is the log likelihood scores for a read
def getDetectData(end=155945):
    scores = []
    # readNames = []
    readNum = -1
    with open('/vol/sci/bio/data/itamar.simon/itamarsi/storage/Nanopore_Oriya/Seq_Jun19_yeast/Itamar_Simon_gDNAyeast-2-5min/'
              'Itamar_Simon_gDNAyeast-2-5min/20190605_1029_GA10000_FAK67866_28476ad6/2-5min_BrdU.detect', 'r') as fp:
        line = fp.readline()
        while line and readNum < end:  # total num of reads: 155945, length of first read 31414
            x = re.search(PTR_DETECT_6MER, line)
            y = re.search(PTR_DETECT_START_READ, line)
            if x is not None:
                # scores[readNum] += [float(x.group(2))]
                scores.append(float(x.group(2)))
            elif y is not None:
                readNum += 1
                # scores.append([])
                # readNames.append(line.split(" ")[0])
            line = fp.readline()
    return np.array(scores), readNum


def plotGaussian(mu, sigma, text=None, histData=None, plot=True, saveName=None):
    if histData is not None:
        histData = histData.reshape(-1, 1)
        t, y, z = plt.hist(histData, 400, density="True")

    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    if text is not None:
        plt.title(text)
    if saveName is not None:
        plt.savefig(saveName)
    if plot:
        plt.show()


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


# get data and split to run_fit & test
scores, readNum = getDetectData(100)
# testScores = scores[:2000]
# scores = scores[2000:]

# estimate normal distribution fit for the data and plot:
dataMean = np.mean(scores)
dataSigma = np.std(scores)
# plotGaussian(dataMean, dataSigma,
#              histData=scores,
#              text="mu=" + str(round(dataMean,2)) + ", sigma=" + str(round(dataSigma,2)),
#              plot=False)


# fit the mixture model to find the dist for the complete model:
modelGMM = hmm.GMMHMM(n_components=1, n_mix=2, covariance_type="diag", n_iter=10,
                      init_params='stmc')
modelGMM.weights_ = [[1 - ALPHA, ALPHA]]  # TODO: something is very weird here...
scores2 = scores[np.argwhere(scores > dataMean)].transpose()[0]  # TODO ?

modelGMM.fit(scores2.reshape(-1, 1))


# plotGaussian(model.means_[0][0], model.covars_[0][0])
#
# plotGaussian(dataMean, dataSigma,
#              text="mu=" + str(round(dataMean,2)) + ", sigma=" + str(round(dataSigma,2)),
#              plot=False)
# plotGaussian(model.means_[1], model.covars_[1][0])
#

# check which dist represent the T and which the BrdU with KL - divergence:
x = np.arange(-10, 10, 0.001)
origin = stats.norm.pdf(x, dataMean, dataSigma)
p0 = stats.norm.pdf(x, modelGMM.means_[0][0][0], modelGMM.covars_[0][0][0])
p1 = stats.norm.pdf(x, modelGMM.means_[0][1][0], modelGMM.covars_[0][1][0])
d0 = kl_divergence(origin, p0)
plt.title('KL(origin||p0) = %1.3f' % d0)
plt.plot(x, origin)
plt.plot(x, p0, c='red')
plt.show()
d1 = kl_divergence(origin, p1)
plt.title('KL(origin||p1) = %1.3f' % d1)
plt.plot(x, origin)
plt.plot(x, p1, c='red')
plt.show()
# We get one is way closer to origin(~203 vs. ~796). Because we have more T then BrdU,
# it's make sense that the closer dist will represent the T.
# i = np.argmin(np.array([d0, d1]))
# j = np.argmax(np.array([d0, d1]))
i = 0
j = 1
muT = modelGMM.means_[0][i][0]
sigmaT = modelGMM.covars_[0][i][0]
muBrdu = modelGMM.means_[0][j][0]
sigmaBrdu = modelGMM.covars_[0][j][0]

# TODO numbers changed due to to weights, looks like i = 0 & j = 1, and that what we want!
pMixture = ((1 - ALPHA) * p0) + (ALPHA * p1)
print(kl_divergence(pMixture, p0))
plt.title('origin||pMixture')
plt.plot(x, origin)
plt.plot(x, pMixture, c='red')
plt.show()

print("T: mu=", muT, ", sigma=", sigmaT)
# plotGaussian(model.means_[0][1], model.covars_[0][1])
print("BrdU: mu=", muBrdu, ", sigma=", sigmaBrdu)

# # create the complete HMM model
model = hmm.GMMHMM(n_components=2, n_mix=2, covariance_type="diag", init_params='st')
# model = hmm.GaussianHMM(n_components=2, covariance_type="diag", params='st')
model.means_ = np.array([[[muT], [muT]],
                         [[muT], [muBrdu]]])
model.covars_ = np.array([[[sigmaT], [sigmaT]],
                          [[sigmaT], [sigmaBrdu]]])
model.weights_ = [[1, 0],
                  [1 - ALPHA, ALPHA]]

model.fit(scores.reshape(-1, 1))
#
#
# # plotGaussian(dataMean, dataSigma,
# #              histData=scores,
# #              text="mu=" + str(round(dataMean,2)) + ", sigma=" + str(round(dataSigma,2)),
# #              plot=False)
# # plotGaussian(model.means_[0][0], model.covars_[0][0],
# #              plot=False)
# # print("0: mu=", model.means_[0][i][0], ", sigma=", model.covars_[0][i][0])
# # plotGaussian(model.means_[0][1], model.covars_[0][1])
# # print("1: mu=", model.means_[0][j][0], ", sigma=", model.covars_[0][j][0])
print(model.transmat_)


a = 2