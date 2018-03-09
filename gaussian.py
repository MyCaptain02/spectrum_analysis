import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X,Y=np.meshgrid(np.linspace(-5,5,1000),np.linspace(-5,5,1000));
mu,sigma=0,1; #suppose that mux=muy=mu=0 and sigmax=sigmay=sigma
G=np.exp(-((X-mu)**2+(Y-mu)**2)/2.0*sigma**2)
# print(G)
fig=plt.figure();
ax=fig.add_subplot(111,projection='3d')
surf=ax.plot_surface(X,Y,G,color='red')
plt.show()