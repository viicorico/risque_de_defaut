from numpy.random import rand
import matplotlib.pyplot as plt
import math

def loiExponentielle(y):
    U=rand()
    return -1*(math.log(1-U))/y

def tabLoiExponentielle(y,Nmc):
    X=[]
    for i in range(0,Nmc):
        X.append(loiExponentielle(y))
    return X


#variance empirique
def Eemp(X,Nmc):
    sum=0
    for i in range(0,Nmc):
        sum+=X[i]
    return sum/Nmc

def Vemp(X,Nmc):
    Emp=Eemp(X,Nmc)
    sum=0
    for i in range(0,Nmc):
        sum+=(Emp-X[i])**2
    return sum/Nmc

def F(X,a,b,Nx,Nmc):
    proba=[]
    x=[]
    for i in range(0,Nx):
        x.append(a+(b-a)*i/Nx)
        compteur=0
        
        for j in range(1,Nmc):
            if(X[j]<=x[i]):
                compteur+=1
        proba.append(compteur/Nmc)
    return (x,proba)

def f(X,a,b,Nx,Nmc):
    proba=[]
    x=[]
    for i in range(0,Nx):
        x.append(a+(b-a)*i/Nx)
        compteur=0
        
        for j in range(1,Nmc):
            if(X[j]<=x[i]+(b-a)/Nx and x[i]<X[j]):
                compteur+=1
        proba.append(compteur/(((b-a)/Nx)*Nmc))
    return (x,proba)

y=2
Nmc=1000
X=tabLoiExponentielle(y,Nmc)

print(X)
print(Eemp(X,Nmc))
print(Vemp(X,Nmc))


a=0
b=2
Nx=100

Y=F(X,a,b,Nx,Nmc)

xrepartition=Y[0]
yrepartition=Y[1]

Y=f(X,a,b,Nx,Nmc)

xdensite=Y[0]
ydensite=Y[1]

f = plt.figure()
f.add_subplot(1,2, 1)
plt.plot(xrepartition,yrepartition)
f.add_subplot(1,2, 2)
plt.plot(xdensite,ydensite)
