import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax

print("hello world")


def eval_var(arr):
    if len(arr) == 0:
        return 0
    return np.var(arr)

def eval_std(arr):
    if len(arr) == 0:
        return 0
    return np.std(arr)


def eval_median(arr):
    if len(arr) == 0:
        return 0
    return np.abs(np.mean(arr) - np.median(arr))

def segmentation(X, n, k, eval, val_func):
    T = np.zeros((n, k))  # keep best value: T[i,j] = best value for | after element t , assuming
    # there're j | left and there's a | after i. 0<=t<=i
    S = np.zeros((n, k))  # keep best index

    for j in range(k):
        for i in range(n):
            if j == 0:
                T[i, j] = eval(X[0:i + 1])
            elif j == k - 1:
                arr = np.array([T[t, j - 1] + eval(X[t + 1:i + 1]) + eval(X[i + 1:n])
                                for t in range(i + 1)])
                T[i, j] = np.min(arr)
                S[i, j] = np.argmin(arr)
            else:
                arr = np.array([T[t, j - 1] + eval(X[t + 1:i + 1]) for t in range(i + 1)])
                T[i, j] = np.min(arr)
                S[i, j] = np.argmin(arr)

    end = np.argmin(T.transpose()[k-1])
    res = [end + 1]
    vals = [val_func(X[end + 1:])]
    errors = [eval(X[end + 1:])]
    for j in range(k):
        if j == k - 1:
            res = [0] + res
            vals = [val_func(X[0:end + 1])] + vals
            errors = [eval(X[0:end + 1])] + errors
        else:
            temp = int(S[end, k-j-1])
            if temp != end:
                res = [temp + 1] + res
                vals = [val_func(X[temp + 1:end + 1])] + vals
                errors = [eval(X[temp + 1:end + 1])] + errors
            end = temp
    return res, vals, errors



# X = [1, 1, 1, 3, 3, 7, 7]
# X = [1, 2, 3, 3, 7]
# X = [2,2,7,7,7,7]
# X = [1,2,3,4,5,6,7]
X = np.loadtxt("/cs/usr/elafallik/Documents/Project/dynamic_prog/data/data1.txt")
X = X[900:1000]
k = 30
n = len(X)

res, vals, errors = segmentation(X, n, k, eval_median, np.mean)
print(res)
# plt.plot(range(n),X)

plt.subplot(111)
plt.plot(range(n),X, marker='.', label='data')
for i in range(len(res)-1):
    r = res[i]
    r2 = res[i+1]
    m = vals[i]
    plt.axvline(x=r)
    plt.plot([r, r2], [m, m])
    # plt.axhline(y=m, xmin=r, xmax=r2)

    # ax.Axes.axvline()
# # plt.plot(range(100), testErr, label='test error')
# # plt.title('Price Prediction Error')
# # plt.ylabel('Train and Test error')
# # plt.xlabel('training data % from data')
plt.legend()
plt.show()
plt.savefig('/cs/usr/elafallik/Documents/Project/dynamic_prog/results/test1.2.png')


np.savetxt("/cs/usr/elafallik/Documents/Project/dynamic_prog/results/data1.txt", vals)