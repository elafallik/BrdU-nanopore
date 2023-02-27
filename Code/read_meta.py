import numpy as np
import re
import os

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

def save_reads(start, end, path):
    scores, readNum, readNames = getDetectData(end, True)

    if not os.path.exists(path + "reads" + str(start) + "-" + str(end-1) + "/"):
        os.mkdir(path + "reads" +  str(start) + "-" + str(end-1) + "/")

    for i in range(start, end):
        np.savetxt(path + "reads" + str(start) + "-" + str(end-1) + "/scores_read_" + readNames[i] + ".txt", np.array(scores[i]), fmt='%s')

    np.savetxt(path + "reads" + str(start) + "-" + str(end-1) + "/reads_names.txt", np.array(readNames[start:end]), fmt='%s')

if __name__ == '__main__':

    scores, readNum, readNames = getDetectData(500, True)
    for i in range(400, 500):
        np.savetxt("/cs/usr/elafallik/Documents/Project/BrdU/data/meta/reads400-499/scores_read_" + readNames[i] + ".txt", np.array(scores[i]), fmt='%s')

    np.savetxt("/cs/usr/elafallik/Documents/Project/BrdU/data/meta/reads400-499/reads_names.txt", np.array(readNames[400:500]), fmt='%s')
