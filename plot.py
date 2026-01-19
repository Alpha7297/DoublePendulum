import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import os
maxn=200
maxframe=5000
data=np.zeros((maxframe,maxn,maxn))
def read():
    for i in range(maxn):
        filename=f"D:\\coding\\python\\physimulator\\doublependulum\\output\\data{i}.txt"
        with open(filename,'r') as file:
                for line_num,line in enumerate(file):
                    temp=line.strip().split()
                    for k in range(maxframe):
                        data[k][i][line_num]=1.0-float(temp[k])
read()
fig,ax=plt.subplots(figsize=(12,10))
im=ax.imshow(data[0],cmap='inferno',extent=[-np.pi,np.pi,-np.pi,np.pi],vmin=0,vmax=1)
cbar=fig.colorbar(im,ax=ax)
cbar.set_label('Color Value')
ax.set_xlabel('Init θ1')
ax.set_ylabel('Init θ2')
ax.set_xlim(-np.pi,np.pi)
ax.set_ylim(-np.pi,np.pi)
ax.set_title("Double Pendulum Chaos Map")
text=ax.text(0.05,0.95,f'frame:0',fontsize=12,transform=ax.transAxes,bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
def update(frame):
    im.set_array(data[frame])
    frame+=1
    text.set_text(f'frame:{frame}')
    return [im,text]
animation=ani.FuncAnimation(fig,update,frames=maxframe,interval=10,blit=True,repeat=True)
plt.show()