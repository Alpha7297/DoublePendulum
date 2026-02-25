import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import taichi
maxframe=1000

g=9.8
def clip(data):
    data=data-2*np.pi*(data>np.pi)
    data=data+2*np.pi*(data<-np.pi)
    return data
class theta:
    def __init__(self,t1,t2,dt1,dt2,h):
        self.t1=t1
        self.t2=t2
        self.dt1=dt1
        self.dt2=dt2
        self.h=h
    def f1(self):
        return self.dt1
    def f2(self):
        return self.dt2
    def g1(self):
        delta=self.t1-self.t2
        temp=1+np.sin(delta)**2
        return (-self.dt2*self.dt2*np.sin(delta)-2*g*np.sin(self.t1)+
                g*np.sin(self.t2)*np.cos(delta)-self.dt1*self.dt1*np.sin(delta)*np.cos(delta))/temp
    def g2(self):
        delta=self.t1-self.t2
        temp=1+np.sin(delta)**2
        return (2*self.dt1*self.dt1*np.sin(delta)-2*g*np.sin(self.t2)+
                self.dt2*self.dt2*np.sin(delta)*np.cos(delta)+2*g*np.sin(self.t1)*np.cos(delta))/temp
    def __add__(self,other):
        return theta(self.t1+other.t1,self.t2+other.t2,self.dt1+other.dt1,self.dt2+other.dt2,self.h)
    def __mul__(self,sc):
        return theta(self.t1*sc,self.t2*sc,self.dt1*sc,self.dt2*sc,self.h)  
    def __rmul__(self,sc):
        return self.__mul__(sc)     
    def step(self):
        kt1=self.f1()
        kt2=self.f2()
        kdt1=self.g1()
        kdt2=self.g2()
        k=theta(kt1,kt2,kdt1,kdt2,self.h)
        return k    
    def reflect(self):
        return np.abs((np.pi*self.t1+self.t2)/(np.pi**2+np.pi))
    def iterate(self):
        h=self.h
        k1=self.step()
        k2=(self+k1*(h/2.0)).step()
        k3=(self+k2*(h/2.0)).step()
        k4=(self+k3*h).step()
        k=self+(k1+2*k2+2*k3+k4)*(h/6.0)
        self.t1=clip(k.t1)
        self.t2=clip(k.t2)
        self.dt1=k.dt1
        self.dt2=k.dt2
N=100
h=0.01
t1=[]
t2=[]
for i in range(2*N):
    t1.append(np.pi*(i-N)/N)
    t2.append(np.pi*(i-N)/N)
tt1=np.array(t1)
tt2=np.array(t2)
t1,t2=np.meshgrid(tt1,tt2,indexing='ij')
dt1=np.zeros_like(t1)
dt2=np.zeros_like(t2)
p=theta(t1,t2,dt1,dt2,h)
fig,ax=plt.subplots(figsize=(8,8))
ax.set_title("double pendulum")
ax.set_xlim(-np.pi,np.pi)
xticks=np.array([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
xtickslabel=["-pi","-pi/2","0","pi/2","pi"]
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabel)
yticks=np.array([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
ytickslabel=["-pi","-pi/2","0","pi/2","pi"]
ax.set_ylim(-np.pi,np.pi)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabel)
im=ax.imshow(p.reflect(),cmap='inferno',extent=[-np.pi,np.pi,-np.pi,np.pi],vmin=0,vmax=1)
cbar=fig.colorbar(im,ax=ax)
cbar.set_label('Color Value')
text=ax.text(0.05,0.95,f'frame:0',fontsize=12,transform=ax.transAxes,bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
def update(frame):
    p.iterate()
    im.set_array(p.reflect())
    frame+=1
    text.set_text(f'frame:{frame}')
    return [im,text]
animation=ani.FuncAnimation(fig,update,frames=maxframe,interval=10,blit=True,repeat=True)
plt.show()