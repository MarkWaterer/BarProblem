import numpy as np
import matplotlib.pyplot as plt

class Configuration:
    X_MIN, X_MAX = -2.0, 2
    T_MIN, T_MAX = 0.0, 3.0

    NX = 81
    NT = 1500

    NU = 0.015
    XI = 25

    C_TERM = 2.0 
    C_RUN = 2 

    MAX_ITER = 100
    ALPHA = 0.03 

    GAMMA = 0.0001

cfg = Configuration()

dt = (cfg.T_MAX - cfg.T_MIN) / cfg.NT
dx = (cfg.X_MAX - cfg.X_MIN) / (cfg.NX - 1)
x_grid = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.NX)

def get_terminal_cost(x):
    return cfg.C_TERM * (x**2)

def get_initial_distribution(x):
    sigma = 0.3 
    mu = 1.0 
    dist = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    return dist

def solve_hjb_backward(m_field, u_terminal):
    u = np.zeros((cfg.NT, cfg.NX))
    u[-1, :] = u_terminal
    
    x_coords = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.NX)
    running_cost_grid = cfg.C_RUN * (x_coords**2)
    
    for n in range(cfg.NT - 2, -1, -1):
        u_next = u[n + 1, :]
        
        d_u_dx = np.zeros_like(u_next)
        d_u_dx[1:-1] = (u_next[2:] - u_next[:-2]) / (2 * dx)
        
        d2_u_dx2 = np.zeros_like(u_next)
        d2_u_dx2[1:-1] = (u_next[2:] - 2 * u_next[1:-1] + u_next[:-2]) / (dx**2)
        
        max_grad = 10.0
        d_u_dx = np.clip(d_u_dx, -max_grad, max_grad)
        hamiltonian = 0.5 * (d_u_dx**2)
        
        congestion = cfg.XI * m_field[n+1, :] - cfg.GAMMA * m_field[n+1,:]**-6
        
        change = -hamiltonian + cfg.NU * d2_u_dx2 + congestion + running_cost_grid
        
        u[n, :] = u_next + dt * change
        
        u[n, 0] = u[n, 1]
        u[n, -1] = u[n, -2]
        
    return u

def solve_fp_forward(u_field, m_initial):
    """
    Solves Fokker-Planck with Reflecting Boundary Conditions.
    Fixes the 'frozen peak' bug at the simulation edges.
    """
    m = np.zeros((cfg.NT, cfg.NX))
    m[0, :] = m_initial
    
    dx_val = (cfg.X_MAX - cfg.X_MIN) / (cfg.NX - 1)

    for n in range(0, cfg.NT - 1):
        m_curr = m[n, :]
        u_curr = u_field[n, :]
        
        # 1. Calculate Velocity
        d_u_dx = np.zeros_like(u_curr)
        d_u_dx[1:-1] = (u_curr[2:] - u_curr[:-2]) / (2 * dx_val)
        
        # Force 0 velocity at walls
        d_u_dx[0] = 0
        d_u_dx[-1] = 0
        
        cfl_limit = 0.8 * (dx_val / dt) 
        velocity = -d_u_dx
        velocity = np.clip(velocity, -cfl_limit, cfl_limit)
        
        # 2. Flux Calculation at Cell Faces
        flux = np.zeros(cfg.NX + 1)
        
        for i in range(1, cfg.NX):
            v_face = 0.5 * (velocity[i] + velocity[i-1])
            if v_face > 0:
                val = m_curr[i-1]
            else:
                val = m_curr[i]
            flux[i] = val * v_face

        # Reflecting BC: No flux across boundaries
        flux[0] = 0.0
        flux[-1] = 0.0
                
        # 3. Advection (Net Flux)
        advection_term = np.zeros_like(m_curr)
        for i in range(cfg.NX):
            advection_term[i] = -(flux[i+1] - flux[i]) / dx_val
            
        # 4. Diffusion with Mirror Boundaries
        d2_m_dx2 = np.zeros_like(m_curr)
        d2_m_dx2[1:-1] = (m_curr[2:] - 2 * m_curr[1:-1] + m_curr[:-2]) / (dx_val**2)
        d2_m_dx2[0] = 2 * (m_curr[1] - m_curr[0]) / (dx_val**2)
        d2_m_dx2[-1] = 2 * (m_curr[-2] - m_curr[-1]) / (dx_val**2)
        
        # 5. Update
        m[n + 1, :] = m_curr + dt * (advection_term + cfg.NU * d2_m_dx2)
        
        # Mass Conservation / Safety
        m[n + 1, :] = np.maximum(m[n + 1, :], 0)
        current_mass = np.sum(m[n + 1, :]) * dx_val
        if current_mass > 1e-9:
            m[n + 1, :] /= current_mass
            
    return m

print("Running simulation...")
m = np.zeros((cfg.NT, cfg.NX))
m_initial = get_initial_distribution(x_grid)
u_terminal = get_terminal_cost(x_grid)

# Initial Guess
for n in range(cfg.NT):
    m[n, :] = m_initial

# Iterative solving
for i in range(cfg.MAX_ITER):
    m_old = m.copy()
    
    u = solve_hjb_backward(m, u_terminal)
    m_calculated = solve_fp_forward(u, m_initial)
    
    m = (1 - cfg.ALPHA) * m_old + cfg.ALPHA * m_calculated
    
    diff = np.max(np.abs(m - m_old))
    if i % 10 == 0:
        print(f"Iteration {i}: Max Change = {diff:.6f}")
        
    if diff < 1e-4:
        print("Converged!")
        break

print("Checking Mass Conservation...")
start_mass = np.sum(m[0, :]) * dx
end_mass = np.sum(m[-1, :]) * dx
print(f"Start Mass: {start_mass:.4f}") 
print(f"End Mass:   {end_mass:.4f}")   

# Plotting
plt.style.use('default')
plt.figure(figsize=(10, 6))

time_indices = [0, int(cfg.NT*0.25), int(cfg.NT*0.5), int(cfg.NT*0.75), cfg.NT-1]
colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

for i, idx in enumerate(time_indices):
    t_label = f"t={idx/cfg.NT:.2f}"
    plt.plot(x_grid, m[idx, :], color=colors[i], label=t_label, linewidth=2)

plt.title("Crowd Density Evolution")
plt.xlabel("Position")
plt.ylabel("Density")
plt.axvline(0, color='red', linestyle='--', alpha=0.3, label="Bar")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Heat map
plt.figure(figsize=(10, 6))
plt.grid(False)
plt.imshow(m, aspect='auto', cmap='plasma', extent=[cfg.X_MIN, cfg.X_MAX, cfg.T_MAX, cfg.T_MIN])
cbar = plt.colorbar()
cbar.set_label('Crowd Density', rotation=270, labelpad=15)
plt.title("Crowd Density Heatmap (Space-Time)", fontsize=14)
plt.xlabel("Position (x) [Bar is at 0]", fontsize=12)
plt.ylabel("Time (t) [0=Start, 1=End]", fontsize=12)
plt.tight_layout()
plt.show()