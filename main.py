#!/usr/bin/env python 
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import numpy as np 
from scipy import signal
from matplotlib import rcParams
rcParams['font-family']='StixGeneral'

def detect_shapes(acf):
	acf = np.array(acf)
	pik = signal.argrelmax(acf, 0, 100)
	shapes=sorted(pik[0], key = lambda x: acf[x], reverse = True)
	return shapes

def h(x): 
	m0 = -(8.0/7.0) 
	m1 = -(5.0/7.0)
	r=m1*x + (0.5)*(m0-m1) * (abs(x+1)-abs(x-1))
	if r>10.0:
		r=10.0
	if r<-10.0:
		r=-10.0
	return r 


def chua(fX,b):
	a = 7.0 
	g = 0.0 
	dt = 0.01 
	N = 20000
	x = [0.001+fX]
	y = [0.002] 
	z = [0.015] 
	for i in range (1, N): 
		x.append(x[i-1] + dt*a*(y[i-1]-x[i-1]-h(x[i-1]))) 
		y.append(y[i-1] + dt*(x[i-1] - y[i-1] + z[i-1])) 
		z.append(z[i-1] + dt*(-b*y[i-1]+g*z[i-1])) 
	return x,y

startX = 0.001
startB = 10.0
TopB = 10.0
BottomB = 10.0 
chaos = chua (0,startB)
mass = []
rr = []
delta=[0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050] 


for i in delta:
	mass.append(chua (i,startB))

		
for i in mass: 
	c = np.corrcoef(chaos[0][:2000], i[0][:2000]) 
	r = c[0,1]
	rr.append(r)


acf_chaos = np.correlate(chaos[0][:20000],chaos[0][:20000], mode="same")
acf_chaos = [x/max(acf_chaos) for x in acf_chaos]

while True:
	not_chaos1 = chua (0,TopB)
	acf1 = np.correlate(not_chaos1[0][:20000],not_chaos1[0][:20000], mode="same")
	acf1 = [x/max(acf1) for x in acf1]
	shapes1 = detect_shapes(acf1)
	if len(shapes1)<2 or acf1[shapes1[0]]-acf1[shapes1[1]]<=0.5:
		break
	TopB += 1
while True:
	not_chaos2 = chua (0,BottomB)
	acf2 = np.correlate(not_chaos2[0][:20000],not_chaos2[0][:20000], mode="same")
	acf2 = [x/max(acf2) for x in acf2]
	shapes2=detect_shapes(acf2)
	if len(shapes2)<2 or acf2[shapes2[0]]-acf2[shapes2[1]]<=0.5:
		break
	BottomB -= 1


print("Верхняя граница B: ", TopB)
print("Нижняя граница B: ", BottomB)
plt.subplot (2, 3, 1)
plt.title("Автокорреляция при B=8") 
plt.plot(range(0,20000),acf2)
plt.subplot (2, 3, 2)
plt.title("Автокорреляция при B=10")
plt.plot(range(0,20000),acf_chaos, "k")
plt.subplot (2, 3, 3)
plt.title("Автокорреляция при B=11")
plt.plot(range(0,20000),acf1)
plt.subplot (2, 3, 4)
plt.title("Функция")
plt.plot(chaos[0],chaos[1])
plt.subplot (2, 3, 5)
plt.title("Корреляция")
plt.plot(delta,rr) 
plt.show()
