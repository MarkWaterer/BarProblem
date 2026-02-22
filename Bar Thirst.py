import numpy as np
import matplotlib.pyplot as plt

class Configuration:
    X_MIN, X_MAX = -2.0, 2.0
    T_MIN, T_MAX = 0.0, 3.0
    
    NX = 81
    NT = 1500

    NU = 0.015
    XI = 25.0

    C_TERM = 2.0
    C_RUN = 1.0
    
    ZETA = 8.0
    K = 3.0

    MAX_ITER = 100
    ALPHA = 0.03

cfg = Configuration()

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
        
        wave_arg = np.pi * t + cfg.K * np.abs(x)
        thirst_cost = cfg.ZETA * (x**2) * (np.cos(wave_arg)**2)
        
        d_u_dx = np.zeros_like(u_next)
        d_u_dx[1:-1] = (u_next[2:] - u_next[:-2]) / (2 * dx)
        d_u_dx = np.clip(d_u_dx, -10.0, 10.0)
        
        d2_u_dx2 = np.zeros_like(u_next)
        d2_u_dx2[1:-1] = (u_next[2:] - 2 * u_next[1:-1] + u_next[:-2]) / (dx**2)
        
        hamiltonian = 0.5 * (d_u_dx**2)
        congestion = cfg.XI * m_field[n+1, :]
        
        change = -hamiltonian + cfg.NU * d2_u_dx2 + congestion + base_cost + thirst_cost
        
        u[n, :] = u_next + dt * change
        u[n, 0] = u[n, 1]
        u[n, -1] = u[n, -2]
        
    return u

def solve_fp_forward(u_field, m_initial):
    m = np.zeros((cfg.NT, cfg.NX))
    m[0, :] = m_initial
    
    dx_val = (cfg.X_MAX - cfg.X_MIN) / (cfg.NX - 1)

    for n in range(0, cfg.NT - 1):
        m_curr = m[n, :]
        u_curr = u_field[n, :]
        
        d_u_dx = np.zeros_like(u_curr)
        d_u_dx[1:-1] = (u_curr[2:] - u_curr[:-2]) / (2 * dx)
        
        cfl = 0.8 * (dx / dt)
        velocity = np.clip(-d_u_dx, -cfl, cfl)
        flux = m_curr * velocity
        
        d_flux_dx = np.zeros_like(m_curr)
        for i in range(1, cfg.NX - 1):
            if velocity[i] > 0:
                d_flux_dx[i] = (flux[i] - flux[i-1]) / dx
            else:
                d_flux_dx[i] = (flux[i+1] - flux[i]) / dx
                
        d2_m_dx2 = np.zeros_like(m_curr)
        d2_m_dx2[1:-1] = (m_curr[2:] - 2 * m_curr[1:-1] + m_curr[:-2]) / (dx**2)
        
        m[n + 1, :] = m_curr + dt * (-d_flux_dx + cfg.NU * d2_m_dx2)
        
        m[n + 1, :] = np.maximum(m[n + 1, :], 0)
        mass = np.sum(m[n + 1, :]) * dx_val
        if mass > 1e-9: m[n + 1, :] /= mass
            
    return m

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

print(f"Start Mass: {np.sum(m[0, :])*dx:.4f}")
print(f"End Mass:   {np.sum(m[-1, :])*dx:.4f}")

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
times = [0, int(cfg.NT*0.33), int(cfg.NT*0.66), cfg.NT-1]
colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
for i, idx in enumerate(times):
    plt.plot(x_grid, m[idx, :], color=colors[i], label=f"t={idx/cfg.NT:.2f}", linewidth=2)
plt.title("Crowd Density Evolution")
plt.xlabel("Position x")
plt.ylabel("Density m(x)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.imshow(m, aspect='auto', cmap='plasma', extent=[cfg.X_MIN, cfg.X_MAX, cfg.T_MAX, cfg.T_MIN])
plt.colorbar(label='Density')
plt.title("Crowd Density Heatmap")
plt.xlabel("Position x")
plt.ylabel("Time t")
plt.grid(False)

cost_grid = np.zeros((cfg.NT, cfg.NX))
x_vals = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.NX)
for n in range(cfg.NT):
    t_curr = n * dt
    wave_arg = np.pi * t_curr + cfg.K * np.abs(x_vals)
    cost_grid[n, :] = (cfg.C_RUN * x_vals**2) + (cfg.ZETA * (x_vals**2) * np.cos(wave_arg)**2)

plt.subplot(2, 1, 2)
plt.imshow(cost_grid, aspect='auto', cmap='magma', extent=[cfg.X_MIN, cfg.X_MAX, cfg.T_MAX, cfg.T_MIN])
cbar = plt.colorbar()
cbar.set_label('Cost', rotation=270, labelpad=15)
plt.title("Thirst Cost Landscape")
plt.xlabel("Position x")
plt.ylabel("Time t")

plt.tight_layout()
plt.show()