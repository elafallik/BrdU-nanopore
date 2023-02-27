import numpy as np
import matplotlib.pyplot as plt
import sys
import os

UNDER_BOUND = 3

print("hello world")


def eval_var(arr, flag=1):
    if flag == 1 and len(arr) < UNDER_BOUND:
        return np.inf
    if len(arr) == 0:
        return np.inf
    return np.var(arr)*len(arr)


def eval_std(arr):
    if len(arr) < UNDER_BOUND:
        return np.inf
    return np.std(arr)


def eval_median(arr):
    if len(arr) < UNDER_BOUND:
        return np.inf
    return np.abs(np.mean(arr) - np.median(arr))

def val_mean(arr):
    if len(arr) == 0:
        return 0
    return np.mean(arr)


# segmenting X into k segments
def segmentation(X, n, k, eval, val_func):

    T = np.full((n, k), np.inf)  # keep opt: T[i,j] = opt segmentation (0,...,i) into j segments.
    S = np.zeros((n, k))  # where the j-th segment begin.

    for j in range(0, k - 1):
        for i in range(UNDER_BOUND*j+(UNDER_BOUND-1), n):
            if j == 0:  # place the first | after i: (0,...,i)
                T[i, j] = eval(X[0: i+1])
            else:  # min (3(j-1)<=t<i-2), opt of (0,...,t) + val of (t+1,...,i)
                # arr = np.array([T[t, j-1] + eval(X[t+1: i+1]) for t in range(max(i-13,UNDER_BOUND*(j-1)+(UNDER_BOUND-1)), i-(UNDER_BOUND-1))])
                arr = np.array([T[t, j-1] + eval(X[t+1: i+1]) for t in range(UNDER_BOUND*(j-1)+(
                        UNDER_BOUND-1), i-(UNDER_BOUND-1))])
                arr_t = [t for t in range(UNDER_BOUND*(j-1)+(UNDER_BOUND-1), i-(UNDER_BOUND-1))]
                # min_arr = np.where(arr == arr.min())
                # t = np.argmax(min_arr) + UNDER_BOUND*(j-1)+(UNDER_BOUND-1)
                t = np.argmin(arr) + UNDER_BOUND*(j-1)+(UNDER_BOUND-1)
                T[i, j] = T[t, j-1] + eval(X[t+1: i+1])
                S[i, j] = t

    for i in range(UNDER_BOUND*(k-1), n-(UNDER_BOUND-1)):
        T[i, k-1] = T[i-1, k-1-1] + eval(X[i: n])  # opt of (0,...,i-1) + val of (i,...,n-1)
        S[i, k-1] = i

    start = np.argmin(T.transpose()[k-1])
    res = [start]
    vals = [val_func(X[start:])]
    errors = [eval(X[start:], 0)]
    end = start  # not included
    start = int(S[start - 1, k - 2]) + 1
    # segmentation of (0,...,start-1) into k-1 segments:
    for j in range(k - 2, 0, -1):

        if (j == 1):
            res = [start] + res
            vals = [val_func(X[start: end])] + vals
            errors = [eval(X[start:end], 0)] + errors
            end = start  # not included
            if end != 0:
                res = [0] + res
                vals = [val_func(X[0: end])] + vals
                errors = [eval(X[0:end], 0)] + errors
        else:
            if start != end:
                res = [start] + res
                vals = [val_func(X[start: end])] + vals
                errors = [eval(X[start:end], 0)] + errors

        end = start  # not included
        start = int(S[start, j-1]) + 1
    res = res + [n - 1]
    return res, vals, errors


def print_segments_for(X, start_data, end_data, k, dir_name):
    n = len(X)

    print("eval for k=" + str(k))
    res, vals, errors = segmentation(X, n, k, eval_var, val_mean)
    print(res)
    print(vals)
    print(errors)
    print('')
    error = np.sum(errors)

    plt.subplot(111)
    plt.plot(range(n), X, marker='.')
    for i in range(len(res) - 2):
        r = res[i]
        r2 = res[i+1] - 1
        m = vals[i]
        plt.axvline(x=r)
        plt.plot([r, r2], [m, m])

    r = res[len(res)-2]
    r2 = res[len(res)-1]
    m = vals[len(res)-2]
    plt.axvline(x=r)
    plt.plot([r, r2], [m, m])

    plt.title('segmentation of X[' + str(start_data) + ':' + str(end_data) + '] for k=' + str(k))
    plt.xlabel(res)
    plt.legend()
    plt.show()
    # plt.savefig(dir_name + 'test1_k=' + str(k) + '.png', dpi=100)
    plt.clf()
    # print("check var")
    # check_var(X)
    return error


def choose_k(X, start_data, end_data, start, end, step=1):
    errors = np.zeros(end - start)
    dir_name = '/cs/usr/elafallik/Documents/Project/dynamic_prog/results/' + str(start_data) + '-' + str(end_data) + '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for k in range(start, end, step):
        errors[k/step-start] = print_segments_for(X, start_data, end_data, k, dir_name) / k
    print(errors)
    plt.subplot(111)
    plt.plot(range(start, len(errors) + start), errors, marker='.', label='data')
    plt.title('segmentation for k=' + str(start) + ' - ' + str(end) + ' with step ' + str(step))
    plt.legend()
    plt.show()
    # plt.savefig(dir_name + 'test1_choose_k.png', dpi=100)
    plt.clf()


def check_var(X):
    while True:
        temp = sys.stdin.readline()
        if temp == '\n':
            break
        start = int(temp)
        end = int(sys.stdin.readline())
        print(X[start:end])
        print(np.var(X[start:end]))


# X = [1, 1, 2, 3, 3, 7, 7]
# X = [1, 2, 3, 3, 7]
# X = [2,2,7,7,7,7]
# X = [1,2,3,4,5,6,7]
X = np.loadtxt("/cs/usr/elafallik/Documents/Project/dynamic_prog/data/data1.txt")
start_data = 400
end_data = start_data + 100
X = X[start_data:end_data]
# X = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,]
# X = [0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1]
# X= [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
# print(print_segments_for(X, start_data, end_data, 13, '/cs/usr/elafallik/Documents/Project/dynamic_prog/results/'))
choose_k(X, start_data, end_data, 10, 20)

