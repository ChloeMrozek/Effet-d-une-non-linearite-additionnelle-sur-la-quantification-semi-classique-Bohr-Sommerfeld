#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:16:42 2024

@author: florentcalvayrac & ChloeMrozek
"""


import numpy as np
import matplotlib.pyplot as plt
import math
from math import atan,sin,cos,tan,exp
from scipy.optimize import brentq
import random
from scipy import optimize
from scipy.integrate import odeint
from scipy.integrate import trapezoid


def find_nearest(array,value):
        idx=(np.abs(array-value)).argmin()
        return array[idx], idx


def f(Y, t):
    pr, pth ,pphi, r,th,phi = Y
    
    dhdr=1/r/r-(pth*pth+pphi*pphi/sin(th)/sin(th))/r/r/r
    dhdth=-pphi*pphi*cos(th)/(sin(th)*sin(th)*sin(th))/r/r
    dhdphi=0.0
            
    dhdpr=pr
    dhdpth=pth/r/r
    dhdpphi=pphi/(r*r*sin(th)*sin(th))

    return[-dhdr,-dhdth,-dhdphi,dhdpr,dhdpth,dhdpphi]


def felec(Y, t):
    pr, pth ,pphi, r,th,phi = Y
    global elx,pe,omeg,sigma,collected_data,condition_fn
   
    rx=r*sin(th)*cos(phi)-elx
    ry=r*sin(th)*sin(phi)
    rz=r*cos(th)
                                                
    pef=-pe*exp(-2*(t-tmax)*(t-tmax)/sigma/sigma)
    omega=omeg*(1+t/tmax)
    collected_data.append([t,pef*cos(omega*t)])
    
    rr=rx*rx+ry*ry+rz*rz
    rr=math.sqrt(rr)
    
    gate = 1.0 if condition_fn(Y, t) else 0.0
    champ = gate * pef * cos(omega*t)
    
    dhdr=1/r/r-(pth*pth+pphi*pphi/sin(th)/sin(th))/r/r/r+2*champ*cos(th)/rr/rr/rr
    dhdth=-pphi*pphi*cos(th)/(sin(th)*sin(th)*sin(th))/r/r+champ*sin(th)/rr/rr
    dhdphi=0.0
    
    dhdpr=pr
    dhdpth=pth/r/r
    dhdpphi=pphi/(r*r*sin(th)*sin(th))

    return[-dhdr,-dhdth,-dhdphi,dhdpr,dhdpth,dhdpphi]


def E(Y,t):
    pr, pth ,pphi, r,th,phi = Y
    rx=r*sin(th)*cos(phi)-elx
    ry=r*sin(th)*sin(phi)
    rz=r*cos(th)
    rr=rx*rx+ry*ry+rz*rz
    rr=math.sqrt(rr)

    res=0.5*(pr*pr+pth*pth/r/r+pphi*pphi/r/r/sin(th)/sin(th))-1/r
    return res


def drv(Y):
    pr, pth ,pphi, r,th,phi = Y
    
    r_safe = max(abs(r), 1e-10)
    sin_th_safe = max(abs(sin(th)), 1e-10)

    rp = pr
    thp = pth / r_safe / r_safe
    phip = pphi / r_safe / r_safe / sin_th_safe / sin_th_safe
    
    rp=pr;thp=pth/r/r;phip=pphi/r/r/sin(th)/sin(th);
    
    erx=sin(th)*cos(phi)
    ery=sin(th)*sin(phi)
    erz=cos(th)
    
    ethx=cos(th)*cos(phi)
    ethy=cos(th)*sin(phi)
    ethz=-sin(th)
    
    ephix=-sin(phi)
    ephiy=cos(phi)
    ephiz=0.0
    
    vx=rp*erx
    vy=rp*ery
    vz=rp*erz
    
    vx=vx+thp*r*ethx
    vy=vy+thp*r*ethy
    vz=vz+thp*r*ethz
    
    vx=vx+phip*sin(th)*r*ephix
    vy=vy+phip*sin(th)*r*ephiy
    vz=vz+phip*sin(th)*r*ephiz
    
    elx=r*erx
    ely=r*ery
    elz=r*erz
    
    return[elx,ely,elz,vx,vy,vz]


def section(Y, t):
    y1, y2 ,y3, y4 = Y
    res = y1
    return res


def distance(a, b):
    distance = 0
    for i in range(len(a) - 1):
        diff = np.array(a[i]) - np.array(b[i])
        distance += diff*diff
    return distance 


def traj(t1):
    global t0,Y,pr0,pth0,pphi0,r0,th0,phi0
    t = np.linspace(t0, t1, 1000)
    ys = odeint(f, Y, t)
    pi=4*atan(1.0)
    dist = distance(drv(ys[999, :]), drv(ys[0, :]))
    return(dist)
    

def traj5(t1):
     global t0,Y,pr0,pth0,pphi0,r0,th0,phi0
     t = np.linspace(t0, t1, 100)
     ys = odeint(f, Y, t)
     pi=4*atan(1.0)
     quantity=ys[len(t)-1,5]-phi0
     while (quantity>pi):
         quantity -=2*math.pi  
     while (quantity<-pi):
         quantity +=2*math.pi   
     return(quantity)
 
    
def traj2(t1):
     global t0,Y
     t = np.linspace(t0, t1, 1000)
     ys = odeint(f, Y, t)
     return(ys)

def trajelec(t1):
     global t0,Y
     t = np.linspace(t0, t1*3, 150000)
     ys = odeint(felec, Y, t,mxstep=50000)
     return(ys)

def trajelecf(t1):
     global t0,Y
     t = np.linspace(t0, t1*3, 1000)
     ys = odeint(felec, Y, t, mxstep=50000)
     dist = distance(drv(ys[999, :]), drv(ys[0, :]))        
     return(dist)
 

def traj3(t1):
     global t0,Y
     t = np.linspace(t0, t1, 1000)
     ys = odeint(f, Y, t)
     return(ys[len(t)-1])


def cond_toujours(Y, t):
    # Champ toujours actif — comportement original
    return True

def cond_y_positif(Y, t):
    # Kick actif uniquement quand y > 0
    pos = drv(Y)
    return pos[1] > 0          # y = pos[1]



def adiabint(fomeg, fpe, fsigma,ftmax,felx,dra,precision):
    global omeg, pe, sigma,tmax,elx,collected_data,tsse,cts,t0,Y,pr0,tsse
    omeg=fomeg
    pe=fpe
    sigma=fsigma
    tmax=ftmax
    elx=felx

    m = 1.0
    o = 1.0
    l=0.0
    
    pi=4*atan(1.0)
    t0=0.0
    pr0=0.0
    pth0=l
    pphi0=m
    r0=1.0
    th0=pi/2
    phi0=pi/2
    
    intt=0.0
    intt= np.empty( 3, dtype=float)
    
    intte= np.empty( 3, dtype=float)
    Y=[pr0,pth0,pphi0,r0,th0,phi0]
    
    if(cts==0):
        cts=1
        tsse = optimize.minimize_scalar(traj, bounds=[pi/4, 2*pi-0.01], method='bounded')
        yex=traj2(tsse.x)
        for iii in range(0,3,1):
            intt[iii]=trapezoid(yex[:,iii],yex[:,iii+3])  /2/pi
        print("exact",intt)
        
    Y=[pr0,pth0,pphi0,r0,th0,phi0]
    
    yex=trajelec(tsse.x)
    Eee=E(yex[len(yex)-1],tsse.x)
    if(Eee<-0.01):
        if abs(Eee +0.5) > 0.12:
    
            Y=[pr0,pth0,pphi0,r0,th0,phi0]
                    
            
            ts = optimize.minimize_scalar(trajelecf, bounds=[tsse.x, tsse.x*40], method='bounded')
            Y=[pr0,pth0,pphi0,r0,th0,phi0]
            collected_data = []

            yex=trajelec(ts.x)
            
            Eee=E(yex[len(yex)-1],ts.x)
            if(Eee<-0.01):
                if abs(Eee +0.5) > 0.1:
                    
                    for iii in range(0,3,1):
                        intte[iii]=trapezoid(yex[:,iii],yex[:,iii+3])  /2/pi
                    if(dra==1):
                        print(1.0/math.sqrt(-2*Eee))
                        Y=[pr0,pth0,pphi0,r0,th0,phi0]

                        times_laser = np.linspace(t0, tsse.x*3, 100000)
                        pef_arr = -pe * np.exp(-2*(times_laser - tmax)**2 / sigma**2)
                        omega_arr = omeg * (1 + times_laser / tmax)
                        data_values_laser = pef_arr * np.cos(omega_arr * times_laser)
                        
                        plt.figure(1)
                        plt.title("Caractéristiques du laser")
                        plt.xlabel('temps (u.a.)')
                        plt.ylabel('E(t)')
                        plt.plot(times_laser, data_values_laser)

                        
                        xv = np.empty((len(yex), 6), dtype=float)
                        Ee= np.empty((len(yex), 1), dtype=float)
                
                        for i, x in enumerate(yex):
                            xv[i] = drv(x)
                            Ee[i]=E(x,0.0)
                       
                
                        plt.figure(2)
                        plt.plot(xv[:,0], xv[:,1], 'b-')
                        plt.title("trajectoire de l'éléctron")
                        plt.axis('equal')
                        plt.show()
                        
                    if(dra==1):
                        xv = np.empty((len(yex), 6), dtype=float)
                        Ee = np.empty((len(yex), 1), dtype=float)
                        t  = np.linspace(t0, ts.x*3, len(yex))

                    
                        for i, x in enumerate(yex):
                            xv[i] = drv(x)
                            Ee[i] = E(x, t[i])
                    
                        ua_to_as = 24.188
                        ua_to_eV = 27.211
                    
                        t_as = t  * ua_to_as
                        E_eV = Ee * ua_to_eV
                        
                        E_n1 = -0.5   * ua_to_eV
                        E_n2 = -0.125 * ua_to_eV
                        E_n3 = -0.055 * ua_to_eV
                        
                        E_array = E_eV[:,0]
                        
                        seuil_depart = E_n1 * 0.85
                        
                        t_depart = None
                        fenetre = 500
                        for i in range(len(E_array) - fenetre):
                            if E_array[i] > seuil_depart:
                                if np.all(E_array[i:i+fenetre] > seuil_depart):
                                    t_depart = t_as[i]
                                    idx_depart = i
                                    break
                        
                        t_arrivee_n2 = None
                        t_arrivee_n3 = None
                        
                        # Trouver la stabilisation finale en remontant depuis la fin
                        idx_stable_final = len(E_array) - fenetre
                        while idx_stable_final > idx_depart:
                            segment = E_array[idx_stable_final:idx_stable_final + fenetre]
                            if np.std(segment) < 0.05 * abs(np.mean(segment)):
                                break
                            idx_stable_final -= 1
                        
                        E_finale = np.mean(E_array[idx_stable_final:idx_stable_final + fenetre])
                        
                        # Chercher la première stabilisation sur l'orbite finale uniquement
                        for E_ref, label in [(E_n2, 'n2'), (E_n3, 'n3')]:
                            if abs(E_finale - E_ref) < 0.15 * abs(E_ref):
                                for i in range(idx_depart, idx_stable_final):
                                    segment = E_array[i:i+fenetre]
                                    if (abs(np.mean(segment) - E_ref) < 0.15 * abs(E_ref)
                                            and np.std(segment) < 0.05 * abs(E_ref)):
                                        if label == 'n2':
                                            t_arrivee_n2 = t_as[i]
                                        else:
                                            t_arrivee_n3 = t_as[i]
                                        break
                        
                        print(f"Départ de n=1        : {t_depart:.2f} as = {t_depart/1000:.4f} fs")
                        if t_arrivee_n2:
                            print(f"Arrivée à n=2        : {t_arrivee_n2:.2f} as = {t_arrivee_n2/1000:.4f} fs")
                            print(f"Durée transition n=2 : {(t_arrivee_n2-t_depart):.2f} as = {(t_arrivee_n2-t_depart)/1000:.4f} fs")
                        if t_arrivee_n3:
                            print(f"Arrivée à n=3        : {t_arrivee_n3:.2f} as = {t_arrivee_n3/1000:.4f} fs")
                            print(f"Durée transition n=3 : {(t_arrivee_n3-t_depart):.2f} as = {(t_arrivee_n3-t_depart)/1000:.4f} fs")
                        
                        plt.figure(3)
                        if t_depart:
                            plt.axvline(x=t_depart, color='k', linestyle=':', label=f'Départ n=1 : {t_depart:.0f} as')
                        if t_arrivee_n2:
                            plt.axvline(x=t_arrivee_n2, color='b', linestyle=':', label=f'Arrivée n=2 : {t_arrivee_n2:.0f} as')
                        if t_arrivee_n3:
                            plt.axvline(x=t_arrivee_n3, color='m', linestyle=':', label=f'Arrivée n=3 : {t_arrivee_n3:.0f} as')
                        plt.legend()
                        plt.title("Temps de transition entre les orbites")
                        plt.xlabel("temps (as)")
                        
                        plt.figure(4)
                        plt.plot(t_as, E_eV, 'teal')
                        plt.axhline(y=-0.5  * ua_to_eV, color='purple', linestyle='--', label=f'n=1 ({-0.5*ua_to_eV:.2f} eV)')
                        plt.axhline(y=-0.125* ua_to_eV, color='m', linestyle='--', label=f'n=2 ({-0.125*ua_to_eV:.2f} eV)')
                        plt.axhline(y=-0.055* ua_to_eV, color='orchid', linestyle='--', label=f'n=3 ({-0.055*ua_to_eV:.2f} eV)')
                        plt.xlim(0,150)
                        plt.xlabel('Temps (as)')
                        plt.ylabel('Énergie (eV)')
                        plt.legend()
                        plt.title("Énergie de l'électron au cours du temps")
                        plt.show()
                    
                        
    if abs(Eee +0.5) < 0.12: 
        Eee=1.0
    
    if(Eee<-0.01):
        return intte[0],intte[1],intte[2]
    else:
        return 0.5,0.5,0.5
   
    
import random


def is_approx_integer(val, precision):
    return abs(val - round(val)) < precision

def is_valid_tuple(tuple_vals, precision):
    return all(0 <= val <= 10 and is_approx_integer(val, precision) for val in tuple_vals)

def refine_search(a, b, c, d, e, range_min, range_max, precision, precision_lower_bound, precision_factor, results):
    if precision < precision_lower_bound:
        result = adiabint(a, b, c, d ,e,1,precision)
        print(precision,result)
        print(f"ω={a:.4f}, pe={b:.4f}, σ={c:.4f}, tmax={d:.4f}, elx={e:.4f}")
        results.append((a, b, c, d, e, result))
        return

    for _ in range(1):
        a_new = random.uniform(max(a - precision, range_min), min(a + precision, range_max))
        b_new = random.uniform(max(b - precision, range_min), min(b + precision, range_max))
        c_new = random.uniform(max(c - precision, range_min), min(c + precision, range_max))
        d_new = random.uniform(max(d - precision, range_min), min(d + precision, range_max))
        e_new = random.uniform(max(e - precision, range_min), min(e + precision, range_max))
        
        result = adiabint(a_new, b_new, c_new, d_new, e_new,0,precision)
        if is_valid_tuple(result, precision):
            refine_search(a_new, b_new, c_new, d_new, e_new, range_min, range_max, precision / precision_factor, precision_lower_bound, precision_factor, results)

def find_valid_combinations_randomly(range_min, range_max, num_samples, precision, precision_lower_bound, precision_factor):
    valid_combinations = []
    for _ in range(num_samples):
        a = random.uniform(range_min, range_max)
        b = random.uniform(range_min, range_max)
        c = random.uniform(range_min, range_max)
        d = random.uniform(0, 10)
        e = random.uniform(range_min, range_max)
        result = adiabint(a, b, c, d, e,0,precision)
        if is_valid_tuple(result, precision):
            refine_search(a, b, c, d, e, range_min, range_max, precision / precision_factor, precision_lower_bound, precision_factor, valid_combinations)
    return valid_combinations


def frange(start, stop, step):
    while start < stop:
        yield round(start, 2)
        start += step


t0=0.0
r0=1.0
m=1.0
pi=4*atan(1.0)
pr0=0.0
th0=pi/2
phi0=pi/2
l=0
collected_data = []
pphi0=m
pth0=l
Y=[pr0,pth0,pphi0,r0,th0,phi0]
cts=0


# -Choix de la condition-

condition_fn = cond_y_positif
# condition_fn = cond_toujours       


range_min, range_max = -10.0, 10.0
num_samples, initial_precision, precision_lower_bound, precision_factor = 20000, 0.15, 0.05, 2

valid_combinations = find_valid_combinations_randomly(range_min, range_max, num_samples, initial_precision, precision_lower_bound, precision_factor)
for combo in valid_combinations:
    print(f"Variables: {combo[:5]}, Result: {combo[5]}")
    
    
    
    