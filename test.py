

import numpy as np

def solver(I, V, q, f, dx, L, dt, T,user_action=None):
#"""Solve u_tt=u_x(q*u_x) + f on (0,L)x(0,T]."""
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1) # Mesh points in time
#    dx = dt*c/float(C)
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1) # Mesh points in space
    C2 = (dt/dx)**2 # Help variable in the scheme
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    if f is None or f == 0 :
        f = lambda x, t: np.exp(t)*(4*np.pi*np.pi - x)*np.sin(2*np.pi*x)
    if V is None or V == 0:
        V = lambda x: 0
        
    if q is None or q == 0:
        q = lambda x: 1+x
        
    u = np.zeros(Nx+1) # Solution array at new time level
    u_n = np.zeros(Nx+1) # Solution at 1 time level back
    u_nm1 = np.zeros(Nx+1) # Solution at 2 time levels back
    
    import time
    t0 = time.clock() # Measure CPU time
    # Load initial condition into u_n
    for i in range(0,Nx+1):
        u_n[i] = I(x[i])
    if user_action is not None:
        user_action(u_n, x, t, 0)
    # Special formula for first time step
    n = 0
#    for i in range(1, Nx):
#    u[i] = u_n[i] + dt*V(x[i]) + \
#    0.5*C2*(u_n[i-1] - 2*u_n[i] + u_n[i+1]) + \
#    0.5*dt**2*f(x[i], t[n])
#    
    for i in range(1, Nx):
        u[i] = 0.5*(2*u_n[i] + \
        C2*(u_n[i+1] -2* u_n[i]+u_n[i-1]) + \
        q(x[i])*dt*dt*u_n[i] + \
        dt*dt*f(x[i], t[n]))

    u[0] = 0; u[Nx] = 0
    if user_action is not None:
        user_action(u, x, t, 1)
#    
    # Switch variables before next step
    u_nm1[:] = u_n; u_n[:] = u
    for n in range(1, Nt):
    # Update all inner points at time t[n+1]
        for i in range(1, Nx):
            u[i] = - u_nm1[i] + 2*u_n[i] + \
            C2*(u_n[i+1] -2* u_n[i]+u_n[i-1]) + \
            q(x[i])*dt*dt*u_n[i] + \
            dt*dt*f(x[i], t[n])
            # Insert boundary conditions
        u[0] = 0; u[Nx] = 0
        if user_action is not None:
            if user_action(u, x, t, n+1):
                break
    # Switch variables before next step
        u_nm1[:] = u_n; u_n[:] = u
    
    cpu_time = time.clock() - t0
    return u, x, t, cpu_time

#def compute_error(u, x, t):
#    global error # must be global to be altered here
#    # (otherwise error is a local variable, different
#    # from error defined in the parent function)
#
#    error = max(error, np.abs(u - u_exact(x, t[-1])).max())
    
def convergence_rates(
    u_exact, # Python function for exact solution
    I, dx0, L, T, # physical parameters
    dt0, num_meshes): # numerical parameters
    """
    Half the time step and estimate convergence rates for
    for num_meshes simulations.
    """
    # First define an appropriate user action function
    global error
    error = 0 # error computed in the user action function
    def compute_error(u, x, t, n):
        global error
        if n == 0:
            error = 0
        else:
            error = max(error, np.abs(u - u_exact(x, t[n])).max())
    # Run finer and finer resolutions and compute true errors
    E = []
    h = [] # dt
    dt = dt0
    dx = dx0
    for i in range(num_meshes):
        u, x, t, cpu_time = solver(I, 0, 0, 0, dx, L, dt, T,user_action=compute_error) # error is computed in the final call to compute_error
        E.append(error)
        h.append(dt)
        dt /= 2 # halve the time step for next simulation
        dx /= 2

    r = [np.log(E[i]/E[i-1])/np.log(h[i]/h[i-1]) for i in range(1,num_meshes)]
    return u,x,E,h,r

I = lambda x: np.sin(2*np.pi*x)
dx = 0.1
L = 2
dt = 0.1
T = 10
#u, x, t, cpu_time = solver(I, 0, 0, 0, dx, L, dt, T)

u_exact = lambda x,t: np.exp(t)*np.sin(2*np.pi*x)
u,x ,E,h,r = convergence_rates(u_exact,I, dx, L,T,dt, 5)
import matplotlib.pyplot as plt
plt.plot(x, u,'r')
plt.plot(x,u_exact(x,T),'k')
plt.show()

