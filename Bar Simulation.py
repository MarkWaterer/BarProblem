import numpy as np
import matplotlib.pyplot as plt

class Configuration:
    # Domain
    X_MIN, X_MAX = -2.0, 2.0
    T_MIN, T_MAX = 0.0, 3.0
    
    # Grid Resolution
    NX = 81
    NT = 1500

    # Physics
    NU = 0.015     # Viscosity
    XI = 25.0      # Congestion Repulsion

    # Costs
    C_TERM = 2.0   # Terminal Cost
    C_RUN = 2.0    # Base Running Cost
    
    # Thirst Wave (Converging)
    ZETA = 10.0    # Wave Amplitude
    K = 3.0        # Wave Frequency

    # Solver
    MAX_ITER = 100
    ALPHA = 0.03   # Damping

cfg = Configuration()

# Grid Setup
dt = (cfg.T_MAX - cfg.T_MIN) / cfg.NT
dx = (cfg.X_MAX - cfg.X_MIN) / (cfg.NX - 1)
x_grid = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.NX)

def get_terminal_cost(x):
    return cfg.C_TERM * (x**2)

def get_initial_distribution(x):
    return (1 / (np.sqrt(2 * np.pi) * 0.3)) * np.exp(-0.5 * ((x - 1.0) / 0.3)**2)

def solve_hjb_backward(m_field, u_terminal):
    u = np.zeros((cfg.NT, cfg.NX))
    u[-1, :] = u_terminal
    
    x = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.NX)
    base_cost = cfg.C_RUN * (x**2)
    
    for n in range(cfg.NT - 2, -1, -1):
        u_next = u[n + 1, :]
        t = n * dt
        
        # Converging Wave: pi*t + k*|x| creates inward travel from both sides
        wave_arg = np.pi * t + cfg.K * np.abs(x)
        thirst_cost = cfg.ZETA * (x**2) * (np.cos(wave_arg)**2)
        
        # Gradients
        d_u_dx = np.zeros_like(u_next)
        d_u_dx[1:-1] = (u_next[2:] - u_next[:-2]) / (2 * dx)
        d_u_dx = np.clip(d_u_dx, -10.0, 10.0)
        
        d2_u_dx2 = np.zeros_like(u_next)
        d2_u_dx2[1:-1] = (u_next[2:] - 2 * u_next[1:-1] + u_next[:-2]) / (dx**2)
        
        # HJB Update
        hamiltonian = 0.5 * (d_u_dx**2)
        congestion = cfg.XI * m_field[n+1, :]
        change = -hamiltonian + cfg.NU * d2_u_dx2 + congestion + base_cost + thirst_cost
        
        u[n, :] = u_next + dt * change
        u[n, 0] = u[n, 1]
        u[n, -1] = u[n, -2] # Neumann BC
        
    return u

def solve_fp_forward(u_field, m_initial):
    m = np.zeros((cfg.NT, cfg.NX))
    m[0, :] = m_initial
    
    dx_val = (cfg.X_MAX - cfg.X_MIN) / (cfg.NX - 1)

    for n in range(0, cfg.NT - 1):
        m_curr = m[n, :]
        u_curr = u_field[n, :]
        
        # Velocity from HJB
        d_u_dx = np.zeros_like(u_curr)
        d_u_dx[1:-1] = (u_curr[2:] - u_curr[:-2]) / (2 * dx)
        
        cfl = 0.8 * (dx / dt)
        velocity = np.clip(-d_u_dx, -cfl, cfl)
        flux = m_curr * velocity
        
        # Advection (Upwind)
        d_flux_dx = np.zeros_like(m_curr)
        for i in range(1, cfg.NX - 1):
            if velocity[i] > 0:
                d_flux_dx[i] = (flux[i] - flux[i-1]) / dx
            else:
                d_flux_dx[i] = (flux[i+1] - flux[i]) / dx
                
        # Diffusion
        d2_m_dx2 = np.zeros_like(m_curr)
        d2_m_dx2[1:-1] = (m_curr[2:] - 2 * m_curr[1:-1] + m_curr[:-2]) / (dx**2)
        
        m[n + 1, :] = m_curr + dt * (-d_flux_dx + cfg.NU * d2_m_dx2)
        
        # Mass Conservation
        m[n + 1, :] = np.maximum(m[n + 1, :], 0)
        mass = np.sum(m[n + 1, :]) * dx_val
        if mass > 1e-9: m[n + 1, :] /= mass
            
    return m

# Main Simulation Loop
print("Running Converging Wave Simulation...")
m = np.zeros((cfg.NT, cfg.NX))
m_initial = get_initial_distribution(x_grid)
u_terminal = get_terminal_cost(x_grid)

for n in range(cfg.NT): m[n, :] = m_initial

for i in range(cfg.MAX_ITER):
    m_old = m.copy()
    u = solve_hjb_backward(m, u_terminal)
    m = (1 - cfg.ALPHA) * m_old + cfg.ALPHA * solve_fp_forward(u, m_initial)
    
    diff = np.max(np.abs(m - m_old))
    if i % 10 == 0: print(f"Iter {i}: Diff = {diff:.6f}")
    if diff < 1e-4: break

# Verification
print(f"Start Mass: {np.sum(m[0, :])*dx:.4f}")
print(f"End Mass:   {np.sum(m[-1, :])*dx:.4f}")

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
times = [0, int(cfg.NT*0.33), int(cfg.NT*0.66), cfg.NT-1]
colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
for i, idx in enumerate(times):
    plt.plot(x_grid, m[idx, :], color=colors[i], label=f"t={idx/cfg.NT:.2f}", linewidth=2)
plt.title("Density Evolution")
plt.xlabel("Position x")
plt.ylabel("Density m(x)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.imshow(m, aspect='auto', cmap='plasma', extent=[cfg.X_MIN, cfg.X_MAX, cfg.T_MAX, cfg.T_MIN])
plt.colorbar(label='Density')
plt.title("Space-Time Heatmap")
plt.xlabel("Position x")
plt.ylabel("Time t")
plt.grid(False)

plt.tight_layout()
plt.show()