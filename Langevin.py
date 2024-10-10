import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import sys

# Modelling Langevin's Stochastic Differential Equation and Gibbs's final measure: dX_t = -grad V(X_t)dt + sigma*dB_t

class SDE_Trajectories:
    """Class for computation of approximated solutions of the Langevin's SDE"""
    def V(self, x):
        """Potential involved in the equation (force term).
        Inputs:
        - x: Array of shape (p,) - Space variable. p: Number of the components of the vector (component wise computation)"""
        ones = np.ones_like(x)
        x = 0.2*x
        #return np.log(x**4-2.5*x**3+x**2+x+3)
        return x**2

    def Nabla_V(self, x):
        """Gradient of the Potential function (force term).
        Inputs:
        - x: Array of shape (p,) - Space variable. p: Number of the components of the vector (component wise computation)"""
        eta = 1e-3
        ones = np.ones_like(x)
        return (self.V(x + eta*ones) - self.V(x - eta*ones))/(2*eta)

    def Trajectories(self, J = 1000, T = 100, h = 0.1, sigma = 1):
        """Computation of trajectories, approximations of solutions of the Langevin's SDE.
        Numerical scheme is Euler-Maruyama method.
        Inputs:
        - J: Int - Number of trajectories. Default: 1000
        - T: Float - Length of the time interval. Default: 100
        - h: Float - Time step. Default: 0.1
        - sigma: Float - Intensity of Gaussian white noise. Default: 1"""

        TT = np.arange(0 , T , h)
        N = np.size(TT)
        X = np.zeros((J,N))
        X[:,0] = np.random.uniform(low=-10, high=10, size=(J,))

        print("Iterations:")
        for n in range(N-1):
            nn = n + 2
            sys.stdout.write("\r%d " % nn + "/" + str(N))
            sys.stdout.flush()
            X[:,n+1] = X[:,n] - h*self.Nabla_V(X[:,n]) + sigma*np.random.normal(loc=0, scale=np.sqrt(h), size=(J,))

        np.save("Langevin_SDE_Trajectories.npy",X)
        np.save("Langevin_SDE_Parameters.npy",(T, h, sigma))
        pass

class SDE_Plot(SDE_Trajectories):
    """Class for plotting trajectories of Langevin's SDE and observe Gibbs's final measure."""
    def plot(self, save = False):
        """Plots on a video the evolution of trajectories and comparison with Gibbs's final measure.
        Inputs:
        - save: Boolean. Saves the figure or not. Default: False"""
        X = np.load("Langevin_SDE_Trajectories.npy")
        T, h, sigma = np.load("Langevin_SDE_Parameters.npy")
        J, N = X.shape
        TT = np.arange(0 , T , h)

        tt = []
        xx = []
        colors = []
        fig = plt.figure(figsize=(10, 4))

        XX = np.linspace(np.min(X), np.max(X), 1000)
        Delta_x = XX[1] - XX[0]
        VV = np.exp(-2 * self.V(XX) / sigma ** 2)
        VV = VV / (Delta_x * np.sum(VV))

        def animation_func(n):
            time = np.round(n*h, int(np.log10(N)))


            plt.clf()
            plt.subplot(1, 2, 1)
            tt = TT[:n]
            xx = X[:,:n]
            plt.title("$t = $ "+str(time))
            for j in range(np.min([J,25])):
                plt.plot(tt, xx[j,:])
            plt.grid()
            plt.xlabel("$t$")
            plt.ylabel("$X_t$")

            plt.subplot(1, 2, 2)
            plt.title("Distributions")
            plt.hist(X[:,n], bins=np.linspace(np.min(X), np.max(X), 50), density=True, color="green", label="SDE's")
            plt.plot(XX, VV, color="red", label="Final (theory)")
            plt.ylim((0,1.25*np.max(VV)))
            plt.grid()
            plt.xlabel("$X_t$")
            plt.ylabel("$IP_X$")
            plt.legend(loc = "upper right")

        animation = FuncAnimation(fig, animation_func, interval=100, blit=False, repeat=True, frames=N)
        if save == True:
            animation.save("Langevin_SDE_J="+str(J)+"_T="+str(T)+"_h="+str(h)+"_sigma="+str(sigma)+".gif", writer="pillow")
        else:
            fig.tight_layout()
            plt.show()
        pass