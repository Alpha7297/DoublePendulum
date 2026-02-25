import taichi as ti
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time
ti.init(arch=ti.cuda)
g=9.8
N=1000
h=0.01
t1_fied=ti.field(dtype=ti.f32,shape=(2*N,2*N))
t2_fied=ti.field(dtype=ti.f32,shape=(2*N,2*N))
dt1_fied=ti.field(dtype=ti.f32,shape=(2*N,2*N))
dt2_fied=ti.field(dtype=ti.f32,shape=(2*N,2*N))
@ti.func
def clip(x):
    pi=ti.math.pi
    if(x>pi):
        x-=2*pi
    if(x<-pi):
        x+=2*pi
    return x
@ti.func
def acc(t1,t2,dt1,dt2):
    delta=t1-t2
    sin_delta=ti.sin(delta)
    cos_delta=ti.cos(delta)
    acc1=-dt2*dt2*sin_delta-2*g*ti.sin(t1)-dt1*dt1*sin_delta*cos_delta+g*ti.sin(t2)*cos_delta
    acc2=2*dt1*dt1*sin_delta-2*g*ti.sin(t2)+dt2*dt2*sin_delta*cos_delta+2*g*ti.sin(t1)*cos_delta
    denom=1+ti.sin(delta)**2
    return acc1/denom,acc2/denom
@ti.kernel
def initiate():
    pi=ti.math.pi
    for i,j in t1_fied:
        t1_fied[i,j]=pi*(i-N)/N
        t2_fied[i,j]=pi*(j-N)/N
        dt1_fied[i,j]=0
        dt2_fied[i,j]=0
@ti.kernel
def rk_step():
    for i,j in t1_fied:
        t1=t1_fied[i,j]
        t2=t2_fied[i,j]
        dt1=dt1_fied[i,j]
        dt2=dt2_fied[i,j]
        k1t1,k1t2=dt1,dt2
        k1dt1,k1dt2=acc(t1,t2,dt1,dt2)
        k2t1,k2t2=dt1+h/2*k1dt1,dt2+h/2*k1dt2
        k2dt1,k2dt2=acc(t1+h/2*k1t1,t2+h/2*k1t2,dt1+h/2*k1dt1,dt2+h/2*k1dt2)
        k3t1,k3t2=dt1+h/2*k2dt1,dt2+h/2*k2dt2
        k3dt1,k3dt2=acc(t1+h/2*k2t1,t2+h/2*k2t2,dt1+h/2*k2dt1,dt2+h/2*k2dt2)
        k4t1,k4t2=dt1+h*k3dt1,dt2+h*k3dt2
        k4dt1,k4dt2=acc(t1+h*k3t1,t2+h*k3t2,dt1+h*k3dt1,dt2+h*k3dt2)
        t1_fied[i,j]=clip(h/6.0*(k1t1+2.0*k2t1+2.0*k3t1+k4t1)+t1)
        t2_fied[i,j]=clip(h/6.0*(k1t2+2.0*k2t2+2.0*k3t2+k4t2)+t2)
        dt1_fied[i,j]=h/6.0*(k1dt1+2.0*k2dt1+2.0*k3dt1+k4dt1)+dt1
        dt2_fied[i,j]=h/6.0*(k1dt2+2.0*k2dt2+2.0*k3dt2+k4dt2)+dt2
initiate()
start_time=time.time()
for i in range(10000):
    rk_step()
    if(i%1000==0):
        print(f"iteratione {i} steps,cost time:{time.time()-start_time} s")