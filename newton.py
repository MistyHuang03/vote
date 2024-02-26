# Author: Misty
# Last Modified: 2023-2-3
# Description: 
#   implement Newton's method to find the root of a specific function which is from model v1.2

import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

def f(x, y, d, n):
    """
    Args:
        x: bar x *
        y: bar y *
        d: degree
        n: number of participants is 2n+1
    Returns:
        value of f
    """
    sumK = 0
    for K in range(0,n+1):
        maxk = np.min([d,2*K])
        coeK = math.comb(2*n,2*K)*math.comb(2*n-2*K,n-K)
        sumk = 0
        for k in range(0,maxk+1):
            coek = math.comb(2*n-2*K,d-k)*math.comb(2*K,k)
            sumk += coek/ 2**(2*n-2*K) * x**(2*K-k) * (1-x)**(2*n-2*K-d+k) * y**k *(1-y)**(d-k)
        sumK += coeK * sumk
    return sumK/math.comb(2*n,d)

def diffx_f(x,y,d,n):
    """
    Returns:
        value of partial derivative of f with respect to x
    """
    sumK = 0
    for K in range(0,n+1):
        maxk = np.min([d,2*K])
        coeK = math.comb(2*n,2*K)*math.comb(2*n-2*K,n-K)
        sumk = 0
        for k in range(0,maxk+1):
            coek = math.comb(2*n-2*K,d-k)*math.comb(2*K,k)
            sumk += coek/ 2**(2*n-2*K) * (2*K-k) * x**(2*K-k-1) * (1-x)**(2*n-2*K-d+k) * y**k * (1-y)**(d-k)
            sumk -= coek/ 2**(2*n-2*K) * x**(2*K-k) * (2*n-2*K-d+k) * (1-x)**(2*n-2*K-d+k-1) * y**k * (1-y)**(d-k)
        sumK += coeK * sumk
    return sumK/math.comb(2*n,d)

def diffy_f(x,y,d,n):
    """
    Returns:
        value of partial derivative of f with respect to y
    """
    sumK = 0
    for K in range(0,n+1):
        maxk = np.min([d,2*K])
        coeK = math.comb(2*n,2*K)*math.comb(2*n-2*K,n-K)
        sumk = 0
        for k in range(0,maxk+1):
            coek = math.comb(2*n-2*K,d-k)*math.comb(2*K,k)
            sumk += coek/ 2**(2*n-2*K) * x**(2*K-k) * (1-x)**(2*n-2*K-d+k) * k * y**(k-1) * (1-y)**(d-k)
            sumk -= coek/ 2**(2*n-2*K) * x**(2*K-k) * (1-x)**(2*n-2*K-d+k) * y**k * (d-k) * (1-y)**(d-k-1)
        sumK += coeK * sumk
    return sumK/math.comb(2*n,d)

def invJ_FG(x, y, c, gamma, n, pdlist):
    """
    Args:
        x: bar x *
        y: bar y *
        c: cost of every vote
        gamma: coefficient of utility function
        n: number of participants is 2n+1
        pdlist: array of probability distribution of degrees
    Returns:
        inverse of Jacobian matrix and value of F,G
    """
    # calculate flist, fxlist, fylist
    flist = np.array([f(x,y,d,n) for d in range(0, 2*n+1)])
    fxlist = np.array([diffx_f(x,y,d,n) for d in range(0, 2*n+1)])
    fylist = np.array([diffy_f(x,y,d,n) for d in range(0, 2*n+1)])
    # calculate the Jacobian matrix [A,B;C,D]
    A = 1 + c/gamma * np.sum(pdlist*fxlist/flist**2)
    B = c/gamma * np.sum(pdlist*fylist/flist**2)
    C = c/gamma * np.sum(pdlist[0:-1]*fxlist[1:]/flist[1:]**2)
    D = 1 + c/gamma * np.sum(pdlist[0:-1]*fylist[1:]/flist[1:]**2)
    invJ = 1/(A*D-B*C) * np.array([[D, -B], [-C, A]])
    # calculate F and G
    F = x - c/gamma * np.sum(pdlist/flist)
    G = y - c/gamma * np.sum(pdlist[0:-1]/flist[1:])
    return invJ, F, G

def newton(x0, y0, c, gamma, n, pdlist, tol, max_iter):
    """
    Args:
        x0: initial guess of x
        y0: initial guess of y
        c: cost of every vote
        gamma: coefficient of utility function
        n: number of participants is 2n+1
        pdlist: array of probability distribution of degrees
        tol: tolerance
        max_iter: maximum number of iterations
    Returns:
        x: approximation of x
        y: approximation of y
        k: number of iterations
    """
    k = 0
    x = x0
    y = y0
    for k in range(max_iter):
        k += 1
        invJ, F, G = invJ_FG(x, y, c, gamma, n, pdlist)
        x1 = x - invJ[0,0]*F - invJ[0,1]*G
        y1 = y - invJ[1,0]*F - invJ[1,1]*G
        err = np.max([abs(x1-x), abs(y1-y)])
        if err < tol:
            return x1, y1, k, err
        x = x1
        y = y1
    return x, y, k, err

def main_powerlaw(x0, y0, n, c, gamma, bias, exponent, plotflag):
    """
    Args:
        x0: initial guess of x
        y0: initial guess of y
        n: number of participants is 2n+1
        c: cost of every vote
        gamma: coefficient of utility function
        bias: bias of powerlaw probability distribution
        exponent: exponent of powerlaw probability distribution
    Returns:
        xdlist: xd under the powerlaw distribution
        xbar: sum of xdlist*pdlist
    """
    tol = 1e-6
    max_iter = 100
    #! pdlist generation 0值做了偏移处理
    pdlist = np.array([1/((d+bias)**exponent) for d in range(0,2*n+1)])
    pdlist = pdlist/np.sum(pdlist)
    x, y, numiter, err = newton(x0, y0, c, gamma, n, pdlist, tol, max_iter)
    if err < tol:
        print(f'x0={x0}, y0={y0}, x={x}, y={y}, numiter={numiter}, err={err}')
    else:
        print('not converge')
        return
    flist = np.array([f(x,y,d,n) for d in range(0,2*n+1)])
    xdlist = c/gamma/flist
    xbar = np.sum(xdlist*pdlist)
    if plotflag:
        plt.plot(range(0,2*n+1),xdlist)
        plt.axhline(xbar)
        plt.show()
    return xdlist, xbar

def mainvisual_powerlaw(x0, y0, bias, exponent, plotflag):
    nlist = np.arange(10,21,1)
    klist = np.linspace(0.1,1,10) #c/gamma
    gamma = 0.5 #?
    arr_xbar = np.zeros([len(nlist),len(klist)])
    for i in range(len(nlist)):
        for j in range(len(klist)):
            n = nlist[i]
            k = klist[j]
            _, xbar = main_powerlaw(x0, y0, n, gamma*k, gamma, bias, exponent, plotflag)
            arr_xbar[i,j] = xbar
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(klist, nlist)
    ax.plot_surface(X, Y, arr_xbar, cmap='viridis')
    ax.set_xlabel('c/gamma')
    ax.set_ylabel('n')
    ax.set_zlabel('xbar')
    plt.show()

""" 
x0list = np.linspace(0.1,0.9,9)
y0list = np.linspace(0.1,0.9,9)
for x0 in x0list:
    for y0 in y0list:
        print(x0,y0)
        main(x0, y0, n=10, c=1, gamma=0.5, bias=1, exponent=2)
"""

def main_uniform(x0, y0, n, c, gamma):
    """
    Args:
        x0: initial guess of x
        y0: initial guess of y
        n: number of participants is 2n+1
        c: cost of every vote
        gamma: coefficient of utility function
    Returns:
        xdlist: xd under the uniform distribution
        xbar: sum of xdlist*pdlist
    """
    tol = 1e-6
    max_iter = 100
    pdlist = np.array([1/(2*n+1) for d in range(0,2*n+1)])
    x, y, numiter, err = newton(x0, y0, c, gamma, n, pdlist, tol, max_iter)
    if err < tol:
        print(f'x0={x0}, y0={y0}, x={x}, y={y}, numiter={numiter}, err={err}')
    else:
        print('not converge')
        return
    flist = np.array([f(x,y,d,n) for d in range(0,2*n+1)])
    xdlist = c/gamma/flist
    xbar = np.sum(xdlist*pdlist)
    """
    plt.plot(range(0,2*n+1),xdlist)
    plt.show()
    """
    return xdlist, xbar