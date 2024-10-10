import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pylab as p
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import sys

# Modelling a system of particles in a fluid with external force [modelled with Brownian motion and damping force]

# Parameter
L = 5 # Half-length of sides of the box

class Particle_ODE:
    """Class for computation of Particles's trajectories"""
    # def V(self, y1, y2, y3):
    #     """Lennard-Jones potential: Modelling of collision in a system of J particles.
    #     Inputs:
    #     - y1: Array of shape (J,) - First space coordinate of all particles
    #     - y2: Array of shape (J,) - Second space coordinate of all particles
    #     - y2: Array of shape (J,) - Third space coordinate of all particles"""
    #     alpha, delta = 1e10, 1e-1
    #     J = np.size(y1) # Number of particles
    #     y1, y2, y3 = y1.reshape(J,1), y2.reshape(J,1), y3.reshape(J,1)
    #     ones = np.ones_like(y1)
    #     D1, D2, D3 = (y1@ones.T - ones@y1.T + np.eye(J))**2, (y2@ones.T - ones@y2.T + np.eye(J))**2, (y3@ones.T - ones@y3.T + np.eye(J))**2
    #     V = alpha*np.sum( (delta**2/(D1+D2+D3))**6 - (delta**2/(D1+D2+D3))**3 )
    #     return V
    #
    # def Nabla_V(self, y1, y2, y3):
    #     """Gradient of Lennard-Jones potential: Modelling of collision in a system of J particles.
    #     Inputs:
    #     - y1: Array of shape (J,) - First space coordinate of all particles
    #     - y2: Array of shape (J,) - Second space coordinate of all particles
    #     - y2: Array of shape (J,) - Third space coordinate of all particles"""
    #     eta = 1e-3
    #     J = np.size(y1)  # Number of particles
    #     y1, y2, y3 = y1.reshape(J, 1), y2.reshape(J, 1), y3.reshape(J, 1)
    #     ones = np.ones_like(y1)
    #     nabla_V_1 = (1/(2*eta))*(self.V(y1 + eta*ones, y2, y3) - self.V(y1 - eta*ones, y2, y3))
    #     nabla_V_2 = (1/(2*eta))*(self.V(y1, y2 + eta*ones, y3) - self.V(y1, y2 - eta*ones, y3))
    #     nabla_V_3 = (1/(2*eta))*(self.V(y1, y2, y3 + eta*ones) - self.V(y1, y2, y3 - eta*ones))
    #     return nabla_V_1, nabla_V_2, nabla_V_3

    def Bound(self, X):
        """Function to take into account collision with bounds of box."""
        X = X+L
        return 4*L*np.abs(X/(4*L) - np.floor(X/(4*L)+1/2))-L

    def Trajectories(self, J = 1000, T = 100, h = 0.1, sigma = 1, Lambda = 1):
        """Computation of trajectories, approximations of trajectories of particles.
        Numerical scheme is Euler-Maruyama method.
        Inputs:
        - J: Int - Number of particles. Default: 1000
        - T: Float - Length of the time interval. Default: 100
        - h: Float - Time step. Default: 0.1
        - sigma: Float - Intensity of Gaussian white noise. Default: 1
        - Lambda: Float - Damping coefficient. Default: 1"""

        TT = np.arange(0, T, h)
        N = np.size(TT)

        X1, X2, X3 = np.zeros((J, N)), np.zeros((J, N)), np.zeros((J, N))
        V1, V2, V3 = np.zeros((J, N)), np.zeros((J, N)), np.zeros((J, N))
        X1[:, 0], X2[:, 0], X3[:, 0] = np.random.uniform(low=-L, high=L, size=(J,)), np.random.uniform(low=-L, high=L, size=(J,)), np.random.uniform(low=-L, high=L, size=(J,))
        v_max = 5 # Maximal initial velocity (absolute value)
        V1[:, 0], V2[:, 0], V3[:, 0] = np.random.uniform(low=-v_max, high=v_max, size=(J,)), np.random.uniform(low=-v_max, high=v_max, size=(J,)), np.random.uniform(low=-v_max, high=v_max, size=(J,))

        # grad_V = self.Nabla_V(X1[:, 0], X2[:, 0], X3[:, 0])
        #
        # X1[:, 1] = self.Bound(X1[:, 0] + h * V1[:, 0] + (h ** 2 / 2) * (-grad_V[0]))
        # X2[:, 1] = self.Bound(X2[:, 0] + h * V2[:, 0] + (h ** 2 / 2) * (-grad_V[1]))
        # X3[:, 1] = self.Bound(X3[:, 0] + h * V3[:, 0] + (h ** 2 / 2) * (-grad_V[2]))
        #X1[:, 1] = X1[:, 0] + h * V1_0 + (h ** 2 / 2) * (-grad_V[0])
        #X2[:, 1] = X2[:, 0] + h * V2_0 + (h ** 2 / 2) * (-grad_V[1])
        #X3[:, 1] = X3[:, 0] + h * V3_0 + (h ** 2 / 2) * (-grad_V[2])

        #V1[:, 1], V2[:, 1], V3[:, 1] = -h * grad_V[0], -h * grad_V[1], -h * grad_V[2]

        print("Iterations:")
        for n in range(N-1):
            nn = n+2
            sys.stdout.write("\r%d " % nn + "/" + str(N))
            sys.stdout.flush()
            #grad_V = self.Nabla_V(X1[:, n+1], X2[:, n+1], X3[:, n+1])
            #print(grad_V)
            #X1[:, n+2] = self.Bound(2*X1[:, n+1] - X1[:, n] + h**2 * (-grad_V[0]))
            #X2[:, n+2] = self.Bound(2*X2[:, n+1] - X2[:, n] + h**2 * (-grad_V[1]))
            #X3[:, n+2] = self.Bound(2*X3[:, n+1] - X3[:, n] + h**2 * (-grad_V[2]))
            #X1[:, n + 2] = 2 * X1[:, n + 1] - X1[:, n] + h ** 2 * (-grad_V[0])
            #X2[:, n + 2] = 2 * X2[:, n + 1] - X2[:, n] + h ** 2 * (-grad_V[1])
            #X3[:, n + 2] = 2 * X3[:, n + 1] - X3[:, n] + h ** 2 * (-grad_V[2])
            #V1[:, n+1], V2[:, n+1], V3[:, n+1] = -h * grad_V[0], -h * grad_V[1], -h * grad_V[2]

            X1[:, n + 1], X2[:, n + 1], X3[:, n + 1] = X1[:, n] + h * V1[:, n], X2[:, n] + h * V2[:, n], X3[:, n] + h * V3[:, n]
            V1[:, n + 1], V2[:, n + 1], V3[:, n + 1] = -np.sign((X1[:, n + 1]-self.Bound(X1[:, n + 1]))**2-0.1)*(V1[:, n] - Lambda*h*V1[:,n] + sigma * np.random.normal(loc=0, scale=np.sqrt(h), size=(J,))), -np.sign((X2[:, n + 1]-self.Bound(X2[:, n + 1]))**2-0.1)*(V2[:,n] - Lambda*h*V2[:,n] + sigma * np.random.normal(loc=0, scale=np.sqrt(h), size=(J,))), -np.sign((X3[:, n + 1]-self.Bound(X3[:, n + 1]))**2-0.1)*(V3[:, n] - Lambda*h*V3[:,n] + sigma * np.random.normal(loc=0, scale=np.sqrt(h), size=(J,)))
            X1[:, n + 1], X2[:, n + 1], X3[:, n + 1] = self.Bound(X1[:, n+1]), self.Bound(X2[:, n+1]), self.Bound(X3[:, n+1])






        #V1, V2, V3 = (X1[:,1:]-X1[:,:-1])/h, (X2[:,1:]-X2[:,:-1])/h, (X3[:,1:]-X3[:,:-1])/h

        np.save("Particles_Trajectories.npy", (X1, X2, X3))
        np.save("Particles_Velocities.npy", (V1, V2, V3))
        np.save("Particles_Parameters.npy", (T, h, sigma, Lambda))

        pass

class Particle_Plot(Particle_ODE):
    """Class for plotting trajectories of particles and observe velocities distributions."""
    def plot(self, name_traj = "Particles_Trajectories.npy", save = False):
        """Plots on a video the evolution of trajectories and comparison with Gibbs's final measure.
        Inputs:
        - name_traj: Character str - Name of the file containing trajectories which is loaded. Default: "Particles_Trajectories.npy".
        - save: Boolean. Saves the figure or not. Default: False"""
        name_model = name_traj[:-17]
        X1, X2, X3 = np.load(name_model+"_Trajectories.npy")
        V1, V2, V3 = np.load(name_model+"_Velocities.npy")
        T, h, sigma, Lambda = np.load(name_model+"_Parameters.npy")
        J, N = X1.shape
        TT = np.arange(0 , T , h)

        tt = []
        xx = []
        colors = []
        fig = plt.figure(figsize=(13, 8))

        # Density of velocities
        VV1, VV2, VV3 = np.linspace(np.min(V1), np.max(V1), 1000), np.linspace(np.min(V2), np.max(V2), 1000), np.linspace(np.min(V3), np.max(V3), 1000)
        Delta_V1, Delta_V2, Delta_V3  = VV1[1] - VV1[0], VV2[1] - VV2[0], VV3[1] - VV3[0]
        DV1, DV2, DV3 = np.exp(-Lambda*VV1**2/ sigma ** 2), np.exp(-Lambda*VV2**2/ sigma ** 2), np.exp(-Lambda*VV3**2/ sigma ** 2)
        DV1, DV2, DV3 = DV1/(Delta_V1*np.sum(DV1)), DV2/(Delta_V2*np.sum(DV2)), DV3/(Delta_V3*np.sum(DV3))

        # Kinetic energy
        Ec = np.sum(V1 ** 2 + V2 ** 2 + V3 ** 2, axis=0) / (2*J)

        # Potential energy
        #Ep = Lambda*np.sum(X1 ** 2 + X2 ** 2 + X3 ** 2, axis=0) / (2*J)



        colors = np.random.randint(low=1, high=100, size=(J,))

        def animation_func(n):
            time = np.round(n*h, int(np.log10(N)))


            plt.clf()

            ax = fig.add_subplot(2, 3, 1, projection='3d')
            xx, yy, zz = X1[:,n], X2[:,n], X3[:,n]
            ax.set_title("$t = $ "+str(time))
            ax.scatter(xx, yy, zz, c=colors, depthshade=0, cmap="rainbow")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_zlabel("$z$")
            ax.set_xlim(-L, L)
            ax.set_ylim(-L, L)
            ax.set_zlim(-L, L)
            ax.grid()

            plt.subplot(2, 3, 2)
            tt = TT[:n]
            EEc = Ec[:n]
            #EEp = Ep[:n]
            #EE = (Ec+Ep)[:n]
            plt.title("Kinetic Energy - $t = $ " + str(time))
            plt.plot(tt, EEc, label="$E_k(t) = \\frac{1}{2}\\langle v(t)^2\\rangle$", color="green")
            #plt.plot(tt, EEp, label="$E_p(t) = \\frac{\\lambda}{2}\\langle x(t)^2\\rangle$", color="red")
            #plt.plot(tt, EE, label="$E(t) = E_k(t) + E_p(t)$", color="orange")
            plt.grid()
            plt.xlabel("$t$")
            plt.ylabel("$E_k$")
            plt.legend(loc = "upper right")


            plt.subplot(2, 3, 4)
            plt.title("Distribution of $v_x$")
            plt.hist(V1[:,n], bins=np.linspace(np.min(V1), np.max(V1), 50), density=True, color="green", label="SDE's")
            plt.plot(VV1, DV1, color="red", label="Final (theory)")
            plt.ylim((0,1.25*np.max(DV1)))
            plt.grid()
            plt.xlabel("$v_x$")
            plt.ylabel("$IP_{v_x}$")
            plt.legend(loc = "upper right")

            plt.subplot(2, 3, 5)
            plt.title("Distribution of $v_y$")
            plt.hist(V1[:, n], bins=np.linspace(np.min(V2), np.max(V2), 50), density=True, color="green", label="SDE's")
            plt.plot(VV2, DV2, color="red", label="Final (theory)")
            plt.ylim((0,1.25*np.max(DV2)))
            plt.grid()
            plt.xlabel("$v_y$")
            plt.ylabel("$IP_{v_y}$")
            plt.legend(loc="upper right")

            plt.subplot(2, 3, 6)
            plt.title("Distribution of $v_z$")
            plt.hist(V3[:, n], bins=np.linspace(np.min(V3), np.max(V3), 50), density=True, color="green", label="SDE's")
            plt.plot(VV3, DV3, color="red", label="Final (theory)")
            plt.ylim((0,1.25*np.max(DV3)))
            plt.grid()
            plt.xlabel("$v_z$")
            plt.ylabel("$IP_{v_z}$")
            plt.legend(loc="upper right")

        animation = FuncAnimation(fig, animation_func, interval=100, blit=False, repeat=True, frames=N)
        if save == True:
            animation.save("Particles_J="+str(J)+"_T="+str(T)+"_h="+str(h)+"_sigma="+str(sigma)+"_Lambda="+str(Lambda)+".gif", writer="pillow")
        else:
            fig.tight_layout()
            plt.show()
        pass
