import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import taichi as ti
import time
ti.init(arch=ti.cuda)
g=9.8
maxframe=1000
N=1000
h=0.01
t1_field=ti.field(dtype=ti.f32,shape=(2*N,2*N))
t2_field=ti.field(dtype=ti.f32,shape=(2*N,2*N))
dt1_field=ti.field(dtype=ti.f32,shape=(2*N,2*N))
dt2_field=ti.field(dtype=ti.f32,shape=(2*N,2*N))
reflect_field=ti.field(dtype=ti.f32,shape=(2*N,2*N))

@ti.func
def clip(x):
    pi=ti.math.pi
    if x>pi:
        x-=2.0*pi
    if x<-pi:
        x+=2.0*pi
    return x

@ti.func
def accelerations(t1,t2,dt1,dt2):
    delta=t1-t2
    sin_delta=ti.sin(delta)
    cos_delta=ti.cos(delta)
    denom=1.0+sin_delta**2
    ddt1=(-dt2*dt2*sin_delta-2.0*g*ti.sin(t1)+
        g*ti.sin(t2)*cos_delta-dt1*dt1*sin_delta*cos_delta)/denom
    ddt2=(2.0*dt1*dt1*sin_delta-2.0*g*ti.sin(t2)+
        dt2*dt2*sin_delta*cos_delta+2.0*g*ti.sin(t1)*cos_delta)/denom
    
    return ddt1,ddt2

@ti.kernel
def initialize():
    for i,j in t1_field:
        t1_field[i,j]=ti.math.pi*(i-N)/N
        t2_field[i,j]=ti.math.pi*(j-N)/N
        dt1_field[i,j]=0.0
        dt2_field[i,j]=0.0

@ti.kernel
def compute_reflect():
    pi=ti.math.pi
    for i,j in reflect_field:
        t1_val=t1_field[i,j]
        t2_val=t2_field[i,j]
        reflect_field[i,j]=ti.abs((pi*t1_val+t2_val)/(pi*pi+pi))

@ti.kernel
def runge_kutta_step():
    h_val=h
    
    for i,j in t1_field:
        t1=t1_field[i,j]
        t2=t2_field[i,j]
        dt1=dt1_field[i,j]
        dt2=dt2_field[i,j]
        k1_t1=dt1
        k1_t2=dt2
        k1_dt1,k1_dt2=accelerations(t1,t2,dt1,dt2)
        t1_k2=t1+0.5*h_val*k1_t1
        t2_k2=t2+0.5*h_val*k1_t2
        dt1_k2=dt1+0.5*h_val*k1_dt1
        dt2_k2=dt2+0.5*h_val*k1_dt2
        k2_t1=dt1_k2
        k2_t2=dt2_k2
        k2_dt1,k2_dt2=accelerations(t1_k2,t2_k2,dt1_k2,dt2_k2)
        t1_k3=t1+0.5*h_val*k2_t1
        t2_k3=t2+0.5*h_val*k2_t2
        dt1_k3=dt1+0.5*h_val*k2_dt1
        dt2_k3=dt2+0.5*h_val*k2_dt2
        k3_t1=dt1_k3
        k3_t2=dt2_k3
        k3_dt1,k3_dt2=accelerations(t1_k3,t2_k3,dt1_k3,dt2_k3)
        t1_k4=t1+h_val*k3_t1
        t2_k4=t2+h_val*k3_t2
        dt1_k4=dt1+h_val*k3_dt1
        dt2_k4=dt2+h_val*k3_dt2
        k4_t1=dt1_k4
        k4_t2=dt2_k4
        k4_dt1,k4_dt2=accelerations(t1_k4,t2_k4,dt1_k4,dt2_k4)
        t1_new=t1+(h_val/6.0)*(k1_t1+2.0*k2_t1+2.0*k3_t1+k4_t1)
        t2_new=t2+(h_val/6.0)*(k1_t2+2.0*k2_t2+2.0*k3_t2+k4_t2)
        dt1_new=dt1+(h_val/6.0)*(k1_dt1+2.0*k2_dt1+2.0*k3_dt1+k4_dt1)
        dt2_new=dt2+(h_val/6.0)*(k1_dt2+2.0*k2_dt2+2.0*k3_dt2+k4_dt2)
        t1_field[i,j]=clip(t1_new)
        t2_field[i,j]=clip(t2_new)
        dt1_field[i,j]=dt1_new
        dt2_field[i,j]=dt2_new

initialize()
compute_reflect()
start_time=time.time()
for i in range(10000):
    runge_kutta_step()
    if i%1000==0:
        print(f"iteration {i} epochs,cost:{time.time()-start_time}s")
fig,ax=plt.subplots(figsize=(8,8))
ax.set_title("Double Pendulum")
ax.set_xlim(-np.pi,np.pi)
xticks=np.array([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
xtickslabel=["-π","-π/2","0","π/2","π"]
ax.set_xlabel("init_theta1")
ax.set_ylabel("init_theta1")
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabel)
ax.set_ylim(-np.pi,np.pi)
yticks=np.array([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
ytickslabel=["-π","-π/2","0","π/2","π"]
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabel)

im=ax.imshow(reflect_field.to_numpy(),cmap='inferno',
    extent=[-np.pi,np.pi,-np.pi,np.pi],
    vmin=0,vmax=1,origin='lower')
cbar=fig.colorbar(im,ax=ax)
cbar.set_label('Reflectivity')

text=ax.text(0.05,0.95,f'frame: 0',fontsize=12,
    transform=ax.transAxes,
    bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

def update(frame):
    runge_kutta_step()
    
    compute_reflect()
    
    im.set_array(reflect_field.to_numpy())
    
    text.set_text(f'frame: {frame+1}')
    
    return [im,text]

animation=ani.FuncAnimation(fig,update,frames=maxframe,
    interval=10,blit=True,repeat=True)

plt.show()