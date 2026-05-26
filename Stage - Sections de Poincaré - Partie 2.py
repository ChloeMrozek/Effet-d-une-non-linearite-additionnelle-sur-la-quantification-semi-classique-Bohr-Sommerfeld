# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:33:40 2026

@author: chloe
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from math import atan, sin, cos, tan, exp, sqrt
from scipy.optimize import brentq
import random
from scipy import optimize
from scipy.integrate import odeint, trapezoid
x

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def f(Y, t):
    pr, pth, pphi, r, th, phi = Y
    dhdr  = 1/r/r-(pth*pth+pphi*pphi/sin(th)/sin(th))/r/r/r
    dhdth = -pphi*pphi*cos(th)/(sin(th)*sin(th)*sin(th))/r/r
    dhdphi = 0.0
    dhdpr   = pr
    dhdpth  = pth/r/r
    dhdpphi = pphi/(r*r*sin(th)*sin(th))
    return[-dhdr,-dhdth,-dhdphi,dhdpr,dhdpth,dhdpphi]


def fperturb(Y, t):
    pr, pth, pphi, r, th, phi = Y
    global alpha
    dhdr   = 1/r**2-(pth**2+pphi**2/sin(th)**2)/r**3+3*alpha*r**2
    dhdth  = -pphi**2*cos(th)/(sin(th)**3*r**2)
    dhdphi = 0.0
    dhdpr   = pr
    dhdpth  = pth/r**2
    dhdpphi = pphi/(r**2*sin(th)**2)
    return[-dhdr,-dhdth,-dhdphi,dhdpr,dhdpth,dhdpphi]


def E(Y, t):
    pr, pth, pphi, r, th, phi = Y
    global alpha
    return 0.5*(pr**2+pth**2/r**2+pphi**2/(r**2*sin(th)**2))-1/r+alpha*r**3


def drv(Y):
    pr, pth, pphi, r, th, phi = Y
    rp=pr; thp=pth/r/r; phip=pphi/r/r/sin(th)/sin(th)
    erx=sin(th)*cos(phi); ery=sin(th)*sin(phi); erz=cos(th)
    ethx=cos(th)*cos(phi); ethy=cos(th)*sin(phi); ethz=-sin(th)
    ephix=-sin(phi); ephiy=cos(phi); ephiz=0.0
    vx=rp*erx+thp*r*ethx+phip*sin(th)*r*ephix
    vy=rp*ery+thp*r*ethy+phip*sin(th)*r*ephiy
    vz=rp*erz+thp*r*ethz+phip*sin(th)*r*ephiz
    return[r*erx,r*ery,r*erz,vx,vy,vz]


def distance(a, b):
    d = 0
    for i in range(len(a)-1):
        diff = np.array(a[i])-np.array(b[i])
        d += diff*diff
    return d


def traj(t1):
    global t0, Y
    t = np.linspace(t0, t1, 500)
    ys = odeint(f, Y, t)
    return distance(drv(ys[499,:]), drv(ys[0,:]))


def traj2(t1):
    global t0, Y
    t = np.linspace(t0, t1, 500)
    return odeint(f, Y, t)


def trajelec(t1):
    global t0, Y
    t = np.linspace(t0, t1, 1500)
    return odeint(fperturb, Y, t, mxstep=50000)


def trajelecf(t1):
    global t0, Y
    t = np.linspace(t0, t1, 500)
    ys = odeint(fperturb, Y, t, mxstep=50000)
    return distance(drv(ys[499,:]), drv(ys[0,:]))


def traj3(t1):
    global t0, Y
    t = np.linspace(t0, t1, 500)
    ys = odeint(f, Y, t)
    return ys[len(t)-1]


def adiabint(falpha, dra, precision, ax, color='blue', pphi0_in=1.0, pr0_in=0.0):       # on enregistre (r, pr) chaque fois que la trajectoire traverse le plan phi = 0.
    global alpha, tsse, cts, t0, Y, pr0

    alpha = falpha
    m = 1.0
    l = 0.0
    pi = 4*atan(1.0)
    t0    = 0.0
    pr0   = pr0_in
    pth0  = l
    pphi0 = pphi0_in
    r0    = 1.0
    th0   = pi/2
    phi0  = 0.0

    intt  = np.empty(3, dtype=float)
    intte = np.empty(3, dtype=float)
    Y = [pr0, pth0, pphi0, r0, th0, phi0]

    if cts == 0:
        cts = 1
        tsse = optimize.minimize_scalar(traj, bounds=[pi/4, 2*pi-0.01], method='bounded')       # Trouve la période de Kepler par minimisation.
        yex = traj2(tsse.x)
        for iii in range(3):
            intt[iii] = trapezoid(yex[:,iii], yex[:,iii+3])/2/pi       

        print("exact", intt)

    Y = [pr0, pth0, pphi0, r0, th0, phi0]

    T_long = tsse.x * 250                                   # Intègre sur 250 périodes pour accumuler suffisamment de points de section. ( plus c'est trop )
    t_arr  = np.linspace(t0, T_long, 40000)
    yex    = odeint(fperturb, Y, t_arr, mxstep=50000)

    Eee = E(yex[-1], T_long)

    if Eee < -0.01:
        phi_mod  = np.mod(yex[:,5], 2*pi)
        r_arr    = yex[:,3]
        pr_arr   = yex[:,0]
        pphi_arr = yex[:,2]
        th_arr   = yex[:,4]

        for i in range(len(yex)-1):                         # Détecte chaque franchissement de phi = π → 0 (section de Poincaré) et trace le point (r, pr) interpolé.
            if phi_mod[i] > pi and phi_mod[i+1] < pi:
                dphi = pphi_arr[i]/(r_arr[i]**2*sin(th_arr[i])**2)
                if dphi > 0:
                    frac   = (2*pi-phi_mod[i])/(2*pi-phi_mod[i]+phi_mod[i+1])
                    r_int  = r_arr[i]  + frac*(r_arr[i+1]-r_arr[i])
                    pr_int = pr_arr[i] + frac*(pr_arr[i+1]-pr_arr[i])
                    ax.plot(r_int, pr_int, '.', markersize=1.2, color=color)

    if Eee < -0.01:
        return intte[0], intte[1], intte[2]
    else:
        return 0.5, 0.5, 0.5


def is_approx_integer(val, precision):
    return abs(val - round(val)) < precision

def is_valid_tuple(tuple_vals, precision):
    return all(0 <= val <= 10 and is_approx_integer(val, precision) for val in tuple_vals)

def refine_search(a, range_min, range_max, precision, precision_lower_bound, precision_factor, results):
    if precision < precision_lower_bound:
        result = adiabint(a, 1, precision)
        print(precision, result)
        print(f"alpha={a:.4f}")
        results.append((a, result))
        return
    for _ in range(1):
        a_new = random.uniform(max(a-precision, range_min), min(a+precision, range_max))
        result = adiabint(a_new, 0, precision)
        if is_valid_tuple(result, precision):
            refine_search(a_new, range_min, range_max, precision/precision_factor,
                          precision_lower_bound, precision_factor, results)


def find_valid_combinations_scan(alpha_val, ax, precision, precision_lower_bound, precision_factor):
    pi = 4*atan(1.0)
    E0 = -0.5               # énergie du niveau fondamental de l'hydrogène (unités atomiques)
    global alpha, tsse, t0, Y
    alpha = alpha_val       # met à jour le global utilisé par fperturb et E

    #  alpha=0 : courbe analytique exacte 
    if alpha_val == 0:
        r_vals = np.linspace(0.01, 3.0, 5000)       # grille de r, évite r=0 (singularité 1/r)
        for pphi_val in np.linspace(0.1, 0.95, 22):  # 22 valeurs de moment angulaire azimutal
            pr2  = 2*(E0 + 1/r_vals - pphi_val**2/(2*r_vals**2))  # pr² par conservation de E
            mask = pr2 >= 0                           # garde uniquement les r classiquement accessibles
            if mask.sum() < 2:                        # moins de 2 points : pas de courbe possible
                continue
            ax.plot(r_vals[mask],  np.sqrt(pr2[mask]), '-', color='steelblue', lw=0.9)  # branche pr>0
            ax.plot(r_vals[mask], -np.sqrt(pr2[mask]), '-', color='steelblue', lw=0.9)  # branche pr<0
        ax.set_ylim(-4, 4)   # limite l'axe car pr→∞ quand r→0 pour les orbites quasi-radiales
        ax.set_xlim(0, 2.1)

    # alpha!=0 : simulation numérique 
    else:
        for pphi_val in np.linspace(0.3, 1.4, 22):
            val = 2*(E0 + 1.0 - pphi_val**2/2.0 - alpha_val)  # pr² à r=1 par conservation de E
            if val < 0:       # énergie inaccessible à r=1 pour ce pphi : on passe
                continue
            pr0_val = sqrt(val)
            for sign in [+1, -1]:  # pr>0 (s'éloigne de r=1) et pr<0 (se rapproche)
                adiabint(alpha_val, 0, precision,
                         ax=ax,
                         color='steelblue',
                         pphi0_in=pphi_val,
                         pr0_in=sign*pr0_val)

    ax.set_xlabel('r')
    ax.set_ylabel('$p_r$')
    ax.set_title('alpha=0 (Kepler pur)' if alpha_val == 0 else f'alpha={alpha_val}')
    ax.grid(True, alpha=0.3)
    print(f"alpha={alpha_val} trace")


def trace_trajectoires_reelles(alpha_val, ax):
    global alpha
    alpha = alpha_val        # met à jour le global pour fperturb
    pi = 4*atan(1.0)
    E0 = -0.5

    colors = plt.cm.cool(np.linspace(0, 1, 12))  # 12 couleurs cyan→magenta, une par orbite

    for k, pphi_val in enumerate(np.linspace(0.1, 0.92, 12)):  # 12 valeurs de pphi
        val = 2*(E0 + 1.0 - pphi_val**2/2.0 - alpha_val)  # pr² à r=1 par conservation de E
        if val < 0:           # énergie inaccessible : on passe
            continue
        pr0_val = sqrt(val)

        Y_loc = [pr0_val, 0.0, pphi_val, 1.0, pi/2, 0.0]  # CI : pr>0, pth=0, r=1, th=pi/2, phi=0
        n_periodes = 3 if alpha_val == 0 else 20  # 3 périodes suffisent pour Kepler (orbites fermées)
                                                   # 20 périodes pour voir la précession si alpha!=0
        t_arr = np.linspace(0, tsse.x * n_periodes, 3000)  # grille temporelle

        if alpha_val == 0:
            yex = odeint(f, Y_loc, t_arr, mxstep=50000)         # Kepler pur
        else:
            yex = odeint(fperturb, Y_loc, t_arr, mxstep=50000)  # Kepler perturbé

        if E(yex[-1], 0) >= -0.01:  # énergie finale positive : orbite ionisée, on rejette
            continue

        # conversion sphérique → cartésien via drv, on prend x=indice 0, y=indice 1
        xv = [drv(yex[i,:])[0] for i in range(len(t_arr))]
        yv = [drv(yex[i,:])[1] for i in range(len(t_arr))]
        ax.plot(xv, yv, '-', color=colors[k], lw=0.7, alpha=0.85)

    ax.plot(0, 0, 'ko', markersize=5, label='noyau')  # noyau à l'origine
    ax.set_aspect('equal')   # axes x et y à la même échelle : les cercles apparaissent ronds
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('alpha=0 (ellipses de Kepler)' if alpha_val == 0 else f'alpha={alpha_val}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

# ── Bloc principal ────────────────────────────────────────────────────────────
t0    = 0.0
r0    = 1.0
m     = 1.0
pi    = 4*atan(1.0)
pr0   = 0.0
th0   = pi/2
phi0  = pi/2
l     = 0
pphi0 = m
pth0  = l
Y     = [pr0, pth0, pphi0, r0, th0, phi0]
cts   = 0
alpha = 0.01

tsse = optimize.minimize_scalar(traj, bounds=[pi/4, 2*pi-0.01], method='bounded')
print(f"Periode de Kepler : {tsse.x:.4f}")
cts = 1

initial_precision, precision_lower_bound, precision_factor = 0.015, 0.0005, 2

alphas_a_tracer = [0.0, -0.005, 0.2, 0.4]

# ── Figure 1 : sections de Poincaré (4 alphas) ───────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
fig1.suptitle("Section de Poincare  (r, pr),  phi=0,  E=-0.5", fontsize=12)
axes1 = axes1.flatten()

for k, alpha_val in enumerate(alphas_a_tracer):
    find_valid_combinations_scan(
        alpha_val             = alpha_val,
        ax                    = axes1[k],
        precision             = initial_precision,
        precision_lower_bound = precision_lower_bound,
        precision_factor      = precision_factor
    )
fig1.tight_layout()

# ── Figure 2 : trajectoires réelles (x,y) pour les mêmes 4 alphas ────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle("Trajectoires reelles (x, y)", fontsize=12)
axes2 = axes2.flatten()

for k, alpha_val in enumerate(alphas_a_tracer):
    trace_trajectoires_reelles(alpha_val, axes2[k])
fig2.tight_layout()

plt.show()