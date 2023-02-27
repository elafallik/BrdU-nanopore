import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

import re
PTR_DETECT_6MER = "(\d+)\t(-?\d.?\d+)\t([CATG]{6})\t([CATG]{6})" # 6mer from .detect file
PTR_DETECT_START_READ = "^\>"

scores = []
readnames = []
readNum = -1
with open('/vol/sci/bio/data/itamar.simon/itamarsi/storage/Nanopore_Oriya/Seq_Jun19_yeast/Itamar_Simon_gDNAyeast-2-5min/'
          'Itamar_Simon_gDNAyeast-2-5min/20190605_1029_GA10000_FAK67866_28476ad6/2-5min_BrdU.detect', 'r') as fp:
    line = fp.readline()
    while line and readNum < 5:  # total num of reads: 155945, length of first read 31414
        x = re.search(PTR_DETECT_6MER, line)
        y = re.search(PTR_DETECT_START_READ, line)
        if x is not None:
            lineNum = int(x.group(1))
            # if float(x.group(2)) >= 2.5:
            scores[readNum] += [float(x.group(2))]
        elif y is not None:
            readNum += 1
            scores.append([])
            readnames.append(line.split(" ")[0])
        line = fp.readline()


# measurements = np.random.normal(loc = 20, scale = 5, size=100)
for i in range(readNum):
    measurements = np.array(scores[i])
    stats.probplot(measurements, dist="norm", plot=plt)

    plt.savefig("/cs/usr/elafallik/Documents/Project/BrdU/readsDist/compareDataToGaussian/ProbabilityPlots/" +
                str(readnames[i]) + ".png")
    plt.show()