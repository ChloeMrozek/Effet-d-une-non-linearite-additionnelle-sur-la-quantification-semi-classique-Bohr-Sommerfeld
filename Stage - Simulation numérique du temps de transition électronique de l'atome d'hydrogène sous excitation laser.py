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



def f(Y, t):                                                #Equations du mouvement de Kepler retourne les equations de Hamilton (dTOTO / dTITI)

    pr, pth ,pphi, r,th,phi = Y
    
    dhdr=1/r/r-(pth*pth+pphi*pphi/sin(th)/sin(th))/r/r/r    # ∂H/∂r
    dhdth=-pphi*pphi*cos(th)/(sin(th)*sin(th)*sin(th))/r/r  # ∂H/∂θ
    dhdphi=0.0                                              # H ne dépend pas de φ → pphi conservé
            
    dhdpr=pr                                                # ∂H/∂pr  → ṙ
    dhdpth=pth/r/r                                          # ∂H/∂pth → θ̇
    dhdpphi=pphi/(r*r*sin(th)*sin(th))                      # ∂H/∂pphi → φ̇

    return[-dhdr,-dhdth,-dhdphi,dhdpr,dhdpth,dhdpphi]



def felec(Y, t):                                            #Equations du mouvement avec un champ LASER dépendant du temps
    pr, pth ,pphi, r,th,phi = Y
    global elx,pe,omeg,sigma,collected_data
   
    
    
    rx=r*sin(th)*cos(phi)-elx                               # La position de l'électron est exprimée dans le repère du centre de masse décalé par elx 
    ry=r*sin(th)*sin(phi)                                   # La position de l'électron est exprimée dans le repère du centre de masse 
    rz=r*cos(th)                                            # La position de l'électron est exprimée dans le repère du centre de masse
                                                
    pef=-pe*exp(-2*(t-tmax)*(t-tmax)/sigma/sigma)           # Enveloppe gaussienne
    omega=omeg*(1+t/tmax)                                   # Fréquence à dérive linéaire (chirp)
    collected_data.append([t,pef*cos(omega*t)])
    

    
    rr=rx*rx+ry*ry+rz*rz
    rr=math.sqrt(rr)                                        # distance effective de l'électron au centre du laser
    
    dhdr=1/r/r-(pth*pth+pphi*pphi/sin(th)/sin(th))/r/r/r+2*pef*cos(th)/rr/rr/rr*cos(omega*t)        # Calcul des dérivées hamiltoniennes
    dhdth=-pphi*pphi*cos(th)/(sin(th)*sin(th)*sin(th))/r/r+pef*sin(th)/rr/rr*cos(omega*t)
    dhdphi=0.0
    
    dhdpr=pr
    dhdpth=pth/r/r
    dhdpphi=pphi/(r*r*sin(th)*sin(th))


    return[-dhdr,-dhdth,-dhdphi,dhdpr,dhdpth,dhdpphi]




def E(Y,t):                                             #Calcule l'energie mécanique du niveau n


    pr, pth ,pphi, r,th,phi = Y
    rx=r*sin(th)*cos(phi)-elx
    ry=r*sin(th)*sin(phi)
    rz=r*cos(th)
    rr=rx*rx+ry*ry+rz*rz
    rr=math.sqrt(rr)

    res=0.5*(pr*pr+pth*pth/r/r+pphi*pphi/r/r/sin(th)/sin(th))-1/r   #Calcule l'énergie cinétique + potentielle coulombienne (en unités atomiques)
    return res



def drv(Y):                                             #Convertis le spherique en cartésien 
    
    pr, pth ,pphi, r,th,phi = Y
    
    #
    r_safe = max(abs(r), 1e-10)
    sin_th_safe = max(abs(sin(th)), 1e-10)

    rp = pr
    thp = pth / r_safe / r_safe
    phip = pphi / r_safe / r_safe / sin_th_safe / sin_th_safe
    #
    
    rp=pr;thp=pth/r/r;phip=pphi/r/r/sin(th)/sin(th);
    
    	
    erx=sin(th)*cos(phi)                                # vecteur er
    ery=sin(th)*sin(phi)                                
    erz=cos(th)                                         
    	
    ethx=cos(th)*cos(phi)                               # vecteur eθ
    ethy=cos(th)*sin(phi)
    ethz=-sin(th)
    	
    ephix=-sin(phi)                                     # vecteur eφ
    ephiy=cos(phi)
    ephiz=0.0
    
    vx=rp*erx                                           # vitesse cartésienne v=r˙e^r​+rθ˙e^θ​+rsinθϕ˙​e^ϕ​
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


def section(Y, t):                                      #


    y1, y2 ,y3, y4 = Y
    res = y1
    return res


def  distance(a, b):                                    #calcul la distance euclidienne au carré
   
    distance = 0
    for i in range(len(a) - 1):
        # Calculate the difference between consecutive points
        diff = np.array(a[i]) - np.array(b[i])
        # Compute the Euclidean distance for these two points and add to the accumulator
        distance += diff*diff
  
    return  distance 


def traj(t1):                                           #Intègre le mouvement de Kepler de t0 à t1 et retourne la distance entre le point final et initial en cartésien
    global t0,Y,pr0,pth0,pphi0,r0,th0,phi0
    t = np.linspace(t0, t1, 500)
    ys = odeint(f, Y, t)
    pi=4*atan(1.0)
    dist = distance(drv(ys[499, :]), drv(ys[0, :]))
           
    return(dist)
    


def traj5(t1):                                          #Intègre le mouvement et retourne la variation d'angle azimutal 
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
 
    
 
    
def traj2(t1):                                          #Intègre le mouvement de Kepler et retourne toute la trajectoire
     global t0,Y
     t = np.linspace(t0, t1, 500)
     ys = odeint(f, Y, t)
     return(ys)

def trajelec(t1):                                       #Intègre avec le champ laser et retourne la distance entre point final et initial 
     global t0,Y
     t = np.linspace(t0, t1, 1500)
     ys = odeint(felec, Y, t,mxstep=50000)
     return(ys)

def trajelecf(t1):                                      # Distance point final/initial avec laser, sert à trouver la nouvelle période sous laser.
     global t0,Y
     t = np.linspace(t0, t1, 500)
     ys = odeint(felec, Y, t, mxstep=50000)
     dist = distance(drv(ys[499, :]), drv(ys[0, :]))        
     return(dist)
 

def traj3(t1):                                          # Intègre le mouvement de Kepler et retourne uniquement le dernier point de la trajectoire. Utile pour le tracer
     global t0,Y
     t = np.linspace(t0, t1, 500)
     ys = odeint(f, Y, t)
     return(ys[len(t)-1])






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
    intt= np.empty( 3, dtype=float)  # Pre-allocating array for performance
    
    
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
                    
                
            collected_data = []
            
            ts = optimize.minimize_scalar(trajelecf, bounds=[tsse.x, tsse.x*40], method='bounded')
            Y=[pr0,pth0,pphi0,r0,th0,phi0]
            
            yex=trajelec(ts.x)
        
            
            Eee=E(yex[len(yex)-1],ts.x)
            if(Eee<-0.01):
                if abs(Eee +0.5) > 0.1:
                    
                    for iii in range(0,3,1):
                        intte[iii]=trapezoid(yex[:,iii],yex[:,iii+3])  /2/pi
                    if(dra==1):
                        print(1.0/math.sqrt(-2*Eee))
                        Y=[pr0,pth0,pphi0,r0,th0,phi0]
                        #plt.plot(ys[:,0], ys[:,3], 'b-') # pat
                        #plt.plot(ys[:,1], ys[:,4], 'b-') # path
                        #plt.plot(ys[:,2], ys[:,5], 'b-') # path
                        plt.figure(1)
                        plt.xlabel('$y_1$')
                        plt.ylabel('$y_2$')
                        
                        times, data_values = zip(*collected_data)
                         
                        plt.plot( times, data_values,'.')
                        
                        xv = np.empty((len(yex), 6), dtype=float)  # Pre-allocating array for performance
                        Ee= np.empty((len(yex), 1), dtype=float)  # Pre-allocating array for performance
                
                        for i, x in enumerate(yex):
                            xv[i] = drv(x)
                            Ee[i]=E(x,0.0)
                       
                
                        plt.figure(2)
                        plt.plot(xv[:,0], xv[:,1], 'b-')
                        plt.title("trajectoire de l'éléctron")
                        plt.axis('equal')
                        plt.show()
                        
                        ###########################################################"
                    if(dra==1):
                        xv = np.empty((len(yex), 6), dtype=float)
                        Ee = np.empty((len(yex), 1), dtype=float)
                        t  = np.linspace(t0, ts.x, 1500)
                    
                        for i, x in enumerate(yex):
                            xv[i] = drv(x)
                            Ee[i] = E(x, t[i])
                    
                        # Conversions
                        ua_to_as = 24.188   # 1 u.a. = 24.188 as
                        ua_to_eV = 27.211   # 1 u.a. = 27.211 eV
                    
                        t_as = t  * ua_to_as        # temps en attosecondes
                        E_eV = Ee * ua_to_eV        # énergie en eV
                        
                        # Niveaux d'énergie en eV
                        E_n1 = -0.5   * ua_to_eV   # -13.61 eV
                        E_n2 = -0.125 * ua_to_eV   # -3.40 eV
                        E_n3 = -0.055 * ua_to_eV   # -1.51 eV
                        
                        E_array = E_eV[:,0]
                        
                        # Seuil de départ : l'électron quitte n=1
                        seuil_depart = E_n1 + 0.05 * abs(E_n3 - E_n1)
                        
                        # 1. Trouver t_depart : premier instant où l'énergie quitte n=1
                        t_depart = None
                        for i in range(len(E_array)):
                            if E_array[i] > seuil_depart:
                                t_depart = t_as[i]
                                idx_depart = i
                                break
                        
                        # 2. Trouver t_arrivee : premier instant où l'énergie croise n=2 ou n=3
                        t_arrivee_n2 = None
                        t_arrivee_n3 = None
                        for i in range(idx_depart, len(E_array)):
                            if t_arrivee_n2 is None and E_array[i] >= E_n2:
                                t_arrivee_n2 = t_as[i]
                            if t_arrivee_n3 is None and E_array[i] >= E_n3:
                                t_arrivee_n3 = t_as[i]
                        
                        # 3. Affichage
                        print(f"Départ de n=1        : {t_depart:.2f} as = {t_depart/1000:.4f} fs")
                        if t_arrivee_n2:
                            print(f"Arrivée à n=2        : {t_arrivee_n2:.2f} as = {t_arrivee_n2/1000:.4f} fs")
                            print(f"Durée transition n=2 : {(t_arrivee_n2-t_depart):.2f} as = {(t_arrivee_n2-t_depart)/1000:.4f} fs")
                        if t_arrivee_n3:
                            print(f"Arrivée à n=3        : {t_arrivee_n3:.2f} as = {t_arrivee_n3/1000:.4f} fs")
                            print(f"Durée transition n=3 : {(t_arrivee_n3-t_depart):.2f} as = {(t_arrivee_n3-t_depart)/1000:.4f} fs")
                        
                        # 4. Marquer sur le graphique
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
                        
                        
                        # Figure 3 : énergie en fonction du temps
                        plt.figure(4)
                        plt.plot(t_as, E_eV, 'teal')
                        plt.axhline(y=-0.5  * ua_to_eV, color='purple', linestyle='--', label=f'n=1 ({-0.5*ua_to_eV:.2f} eV)')
                        plt.axhline(y=-0.125* ua_to_eV, color='m', linestyle='--', label=f'n=2 ({-0.125*ua_to_eV:.2f} eV)')
                        plt.axhline(y=-0.055* ua_to_eV, color='orchid', linestyle='--', label=f'n=3 ({-0.055*ua_to_eV:.2f} eV)')
                        plt.xlabel('Temps (as)')
                        plt.ylabel('Énergie (eV)')
                        plt.legend()
                        plt.title("Énergie de l'électron au cours du temps")
                        plt.show()
                    
                        # Extraction du temps de transition
                        E_transition = (-0.5 + (-0.055)) / 2 * ua_to_eV
                        idx = np.where(np.diff(np.sign(E_eV[:,0] - E_transition)))[0]
                        if len(idx) > 0:

                            t_transition = t[idx[0]]
                            print(f"Temps de transition : {t_transition:.2f} u.a.")
                            t_transition_as = t_as[idx[0]]
                            print(f"Temps de transition : {t_transition_as:.2f} as  =  {t_transition_as/1000:.4f} fs")  
                        #################################################################
                        
                        
                        
                # =============================================================================
                #         plt.figure(3)
                #         plt.plot(Ee, 'b-')
                # =============================================================================
        
    if abs(Eee +0.5) < 0.12: 
        Eee=1.0
    
    if(Eee<-0.01):
    
        return intte[0],intte[1],intte[2]
    else:
    
        return 0.5,0.5,0.5
   

 


import random


def is_approx_integer(val, precision):                      #Teste si une valeur est approximativement un entier à precision près
    return abs(val - round(val)) < precision

def is_valid_tuple(tuple_vals, precision):                  #Vérifie que toutes les actions du tuple sont dans [0, 10] et approximativement entières.
    return all(0 <= val <= 10 and is_approx_integer(val, precision) for val in tuple_vals)

def refine_search(a, b, c, d, e, range_min, range_max, precision, precision_lower_bound, precision_factor, results):                    #Raffinement récursif : si une combinaison de paramètres laser donne des actions entières, on explore son voisinage avec une précision croissante
    if precision < precision_lower_bound:
        result = adiabint(a, b, c, d ,e,1,precision)
        print(precision,result)
        print(f"ω={a:.4f}, pe={b:.4f}, σ={c:.4f}, tmax={d:.4f}, elx={e:.4f}")
        results.append((a, b, c, d, e, result))
        return

    for _ in range(1):  # Number of trials in the refined search
        a_new = random.uniform(max(a - precision, range_min), min(a + precision, range_max))
        b_new = random.uniform(max(b - precision, range_min), min(b + precision, range_max))
        c_new = random.uniform(max(c - precision, range_min), min(c + precision, range_max))
        d_new = random.uniform(max(d - precision, range_min), min(d + precision, range_max))
        e_new = random.uniform(max(e - precision, range_min), min(e + precision, range_max))
        
        result = adiabint(a_new, b_new, c_new, d_new, e_new,0,precision)
        if is_valid_tuple(result, precision):
            refine_search(a_new, b_new, c_new, d_new, e_new, range_min, range_max, precision / precision_factor, precision_lower_bound, precision_factor, results)

def find_valid_combinations_randomly(range_min, range_max, num_samples, precision, precision_lower_bound, precision_factor):                        # Boucle principale de recherche : tire aléatoirement des nombres et les teste
    valid_combinations = []
    for _ in range(num_samples):
        a = random.uniform(range_min, range_max)
        b = random.uniform(range_min, range_max)
        c = random.uniform(range_min, range_max)
        d = random.uniform(range_min, range_max)
        e = random.uniform(range_min, range_max)
        result = adiabint(a, b, c, d, e,0,precision)
        if is_valid_tuple(result, precision):
            refine_search(a, b, c, d, e, range_min, range_max, precision / precision_factor, precision_lower_bound, precision_factor, valid_combinations)
    return valid_combinations





def frange(start, stop, step):
    while start < stop:
        yield round(start, 2)
        start += step

# Example usage
t0=0.0
r0=1.0
m=1.0
pi=4*atan(1.0)
pr0=0.0
th0=pi/2
phi0=pi/2
l=0
pphi0=m
pth0=l
Y=[pr0,pth0,pphi0,r0,th0,phi0]
collected_data = []
cts=0

# Example usage
range_min, range_max = -10.0, 10.0
num_samples, initial_precision, precision_lower_bound, precision_factor = 10000, 0.15, 0.05, 2

valid_combinations = find_valid_combinations_randomly(range_min, range_max, num_samples, initial_precision, precision_lower_bound, precision_factor)
for combo in valid_combinations:
    print(f"Variables: {combo[:5]}, Result: {combo[5]}")





