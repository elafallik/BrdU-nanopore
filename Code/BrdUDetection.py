import numpy as np
import matplotlib.pyplot as plt

from hmmlearn import hmm

import re
PTR_MODEL = "{ \"([GATC]{6})\", { (-?\d.?\d+) ,(-?\d.?\d+)} }" # read from poreModel files
PTR_DETECT = "(\d+)\t(-?\d.?\d+)\t([CATG]{6})\t([CATG]{6})" # read from .detect file

# read data and get log likelihood of each 6-mer
scores = []
with open('/cs/usr/elafallik/Documents/Project/BrdU/data/test3.txt', 'r') as fp:
    line = fp.readline()
    while line:
        if re.search(PTR_DETECT, line) is not None:
            x = re.split("\t", line)
            # print(x)
            scores += [float(x[1])]
        line = fp.readline()
print(len(scores))
scores = np.array(scores).reshape(-1, 1)
# count, bins, ignored = plt.hist(scores, 300, density=True)
# plt.show()

# read models
poreModelsBrdU = {}
with open('/cs/usr/elafallik/Documents/Project/BrdU/Code/poreModelsBrdU.txt', 'r') as fp:
    line = fp.readline()
    while line:
        x = re.search(PTR_MODEL, line)
        if x is not None:
            sixMer = x.group(1)
            mu = x.group(2)
            sigma = x.group(3)
            poreModelsBrdU[sixMer] = [mu, sigma]
        line = fp.readline()

poreModelONT = {}
with open('/cs/usr/elafallik/Documents/Project/BrdU/Code/poreModelONT.txt', 'r') as fp:
    line = fp.readline()
    while line:
        x = re.search(PTR_MODEL, line)
        if x is not None:
            sixMer = x.group(1)
            mu = x.group(2)
            sigma = x.group(3)
            poreModelONT[sixMer] = [mu, sigma]
        line = fp.readline()
# scores = np.array(scores).reshape(-1, 1)
# count, bins, ignored = plt.hist(scores, 300, density=True)
# plt.show()


# plot normal dist
# example for "GAACGT", { 83.6619055624 ,1.15168920165} } (BrdU) and "GAACGT", { 85.417027 ,1.554534} }, (ONT)
sigma1 = 1.15168920165
mu1 = 83.6619055624
s1 = np.random.normal(mu1, sigma1, 10000)
sigma2 = 1.554534
mu2 = 85.417027
s2 = np.random.normal(mu2 ,sigma2, 10000)
count, bins1, ignored = plt.hist(s1, 300, density=True)
count, bins2, ignored = plt.hist(s2, 300, density=True)
plt.show()
s = np.append(s1, s2)
count, bins2, ignored = plt.hist(s, 300, density=True)
plt.show()

# sample from gaussianHMM with 2 states and plot
# example for "GAACGT", { 83.6619055624 ,1.15168920165} } (BrdU) and "GAACGT", { 85.417027 ,1.554534} }, (ONT)
model = hmm.GaussianHMM(n_components=2, covariance_type="diag")
model.fit(scores)
model.means_ = np.array([[mu1],
                         [mu2]])
model.covars_ = np.array([[sigma1], [sigma2]])
logit, states = model.decode(s2.reshape(-1, 1), algorithm="viterbi")
# count, bins, ignored = plt.hist(X, 500, density=True)
# plt.show()


# # for each 6-mer, sample from gaussianHMM with 2 states and plot
# model = hmm.GaussianHMM(n_components=2, covariance_type="diag")
# model.fit(scores)
# X, Z = model.sample(10000)
# logit, states = model.decode(X, algorithm="viterbi")
# k = model.get_stationary_distribution()
# count, bins, ignored = plt.hist(X, 500, density=True)
# plt.show()






a=2
