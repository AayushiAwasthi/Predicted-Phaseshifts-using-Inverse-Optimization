

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import pi, abs, mean, sin, cos, exp, sqrt, cosh, log
from scipy.special import erf
from scipy.optimize import differential_evolution, NonlinearConstraint, minimize, basinhopping, Bounds
import numba
import matplotlib.pyplot as plt
import os

M_d = 1875.612928  # MeV / c ^ 2
M_n = 939.565379  # MeV / c ^ 2
hbarc = 197.3269718  # MeV fm
mu = (M_n * M_d) / (M_n + M_d)
den = (hbarc ** 2) / ( mu)
den2 = (hbarc ** 2) / (2 * mu)
E1S0 = np.array([0.0015,	0.08,	0.15,	0.3,	0.45,	0.6,	0.75,	0.9,	1.2,	1.35,	1.5]
) ####For2S1/2
#P3S1 = np.array([179.77,	178.01,	176.72,	174.32,	172.05,	169.9,	167.9,	166,	162.5,	160.9,	159.3])####For2S1/2
P3S1 = np.array([177.91,	165.4,	159.6,	151.7,	146,	141.5,	137.6,	134.3,	128.8,	126.4,	124.2])####For4S1/2
abs_P3S1_inv = abs(P3S1 ** - 1)

k = sqrt(E1S0 / den)
com = M_d/(M_d+M_n)
k = sqrt(E1S0*com / den2)
k_1 = k ** -1
den_1 = -den2 ** -1
h = 0.01
h_4 = h / 4
h_8 = h / 8
h_2 = h / 2
h_34 = h * 0.75
h_3_16 = h * 3 / 16
h_9_16 = h * 9 / 16
h_3_7 = h * 3 / 7
h_12_7 = h * 12 / 7
h_90 = h / 90
h_8_7 = h * 8 / 7
h_2_7 = h * 2 / 7
rad2deg = 180. / pi

####For 20exp######

######Optimised parameters for 2S1/2

# a1 = 1.7455912897602868  
# a2 = 2.86267139217461 
# a3 = 0.32745549420377185 
# a4 = 0.7050472036358678   
# a5= 1.7735550770415607 
# a6 = 3.1874645135661646 
# a7 = 1e-06 
# a8 = 1.9084928396224667 
# a9 = 4.3121667264162475 
# d0 = 61.809161182619924

##########Optimised parameters for 4S1/2
a1 = 0.39220224093257766  
a2 = 1.3896561975678758 
a3 = 1.1115573106956607 
a4 = 4.823198432910238   
a5 = 1.2735317471095542 
a6 = 4.192030205228069 
a7 = 1e-05 
a8 = 0.331799746688496 
a9 = 1.9236997918747352 
D0 = 86.77158340575632


x1 = np.arange(0.01, a8, 0.0025)
y1 = np.arange(a8, a9 , 0.0025)
z1 = np.arange(a9, 20, 0.0025)

 #####for 3S1##########
f0 = exp(-2*a1*(a8 - a4)) - (2*exp(-a1*(a8 - a4)))
f1 = exp(-2*a2*(a8 - a5)) - (2*exp(-a2*(a8 - a5)))

g0 = exp(-2 *a1 *(a8 - a4)) - (exp(-a1 *(a8 - a4)))
g1 = exp(-2 *a2 *(a8 - a5)) - (exp(-a2 *(a8 - a5)))

k1 = exp(-2*a2*(a9-a5))-2*exp(-a2*(a9-a5))
k2 = exp(-2*a3*(a9-a6))-2*exp(-a3*(a9-a6))

l1 = exp(-2*a2*(a9-a5)) - exp(-a2*(a9-a5))
l2 = exp(-2*a3*(a9-a6)) - exp(-a3*(a9-a6))


D1 = (a1 * D0 * g0)/(a2 * g1)
D2 = (a2 * D1 *l1)/(l2 * a3)
#v0 = D1 * f1 - D0 * f0 + a7
v1 = a7 + D2*k2 - D1*k1
v0 = v1 + D1*f1 - D0*f0


V_0 = v0 + D0 *(exp(-2*a1*(x1-a4)) - 2*exp(-a1*(x1-a4)))
V_1 = v1 + D1 *(exp(-2*a2*(y1-a5)) - 2*exp(-a2*(y1-a5)))
V_2 = a7 + D2*(exp(-2*a3*(z1-a6)) - 2*exp(-a3*(z1-a6)))




v_new = np.append(np.append(V_0, V_1), V_2)


z = np.append(np.append(x1, y1), z1)


def f(x_in, y_in, vx):
    E1 = np.arange(0.01,5.01,0.01)  
    k1 = sqrt(E1*com / den2)
    k_2 = k1 ** -1
    den_2 = -(den2 ** -1)
    kx = k1 * x_in
    res = den_2 * k_2 * vx * (cos(y_in) * sin(kx) + sin(y_in) * cos(kx)) ** 2
    return res



def rk_method(v_new):
    idx = 0
    x = 0.01
    E1 = np.arange(0.01,5.01,0.01)  
    k1 = sqrt(E1*com / den2)
    y = np.zeros_like(k1)
    for _ in range(1998):
        k1 = f(x, y, v_new[idx])
        k2 = f(x + h_4, y + h_4 * k1, v_new[idx + 1])
        k3 = f(x + h_4, y + h_8 * (k1 + k2), v_new[idx + 1])
        k4 = f(x + h_2, y - h_2 * k2 + h * k3, v_new[idx + 2])
        k5 = f(x + h_34, y + h_3_16 * k1 + h_9_16 * k4, v_new[idx + 3])
        k6 = f(x + h, y - h_3_7 * k1 + h_2_7 * k2 + h_12_7 * (k3 - k4) + h_8_7 * k5, v_new[idx + 4])
        y = y + h_90 * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6)
        idx += 4
        x += h
    return y

delta_best1 = rk_method(v_new)
delta_best11 = delta_best1* rad2deg


def f1(x_in, y_in, vx):
     
    k1 = sqrt(E1S0*com / den2)
    k_2 = k1 ** -1
    den_2 = -(den2 ** -1)
    kx = k1 * x_in
    res = den_2 * k_2 * vx * (cos(y_in) * sin(kx) + sin(y_in) * cos(kx)) ** 2
    return res



def rk_method1(v_new):
    idx = 0
    x = 0.01
    k1 = sqrt(E1S0*com / den2)
    y = np.zeros_like(k1)
    for _ in range(1998):
        k1 = f1(x, y, v_new[idx])
        k2 = f1(x + h_4, y + h_4 * k1, v_new[idx + 1])
        k3 = f1(x + h_4, y + h_8 * (k1 + k2), v_new[idx + 1])
        k4 = f1(x + h_2, y - h_2 * k2 + h * k3, v_new[idx + 2])
        k5 = f1(x + h_34, y + h_3_16 * k1 + h_9_16 * k4, v_new[idx + 3])
        k6 = f1(x + h, y - h_3_7 * k1 + h_2_7 * k2 + h_12_7 * (k3 - k4) + h_8_7 * k5, v_new[idx + 4])
        y = y + h_90 * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6)
        idx += 4
        x += h
    return y

delta_best12 = rk_method1(v_new)
delta_best112 = delta_best12* rad2deg


print('idx   Energy   1S0-Phase-Exp.      1S0-Phase-Sim.')
for idx, _ in enumerate(E1S0):
    print(f'{idx + 1:2}   {int(E1S0[idx]):2}       {P3S1[idx]:.3f}           {delta_best112[idx]:.3f}')
@numba.jit(nopython=True)
def metric_absPercentError(delta_best11):
        return mean((abs(delta_best11 - P3S1) * abs_P3S1_inv)*100)

print(f'metric\nAbsolute Percent Error:\t{metric_absPercentError(delta_best112)}\n')



E1 = np.arange(0.01,5.01,0.01)

plt.figure()
plt.plot(z, v_new)
plt.legend(['$^1S_0$'])
plt.xlabel('r (fm)')
plt.ylabel('V(r) (MeV)')
plt.axis([0, 5, -400, 400])
plt.grid()
plt.title('1S0 np Morse Potential')
plt.savefig('Potential_1', dpi=300)

plt.figure()
plt.plot(E1S0, P3S1, 'ro')
plt.grid()
plt.plot(E1, delta_best11, 'g-')
plt.xlabel('Energy MeV')
plt.ylabel('Phase Shift (degrees)')
plt.legend(['Experimental', 'Simulated'])
plt.title('3s1 np - Phase Shift')
plt.savefig('Phase Shift_1', dpi=300)   


plt.show()