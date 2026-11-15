import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def softplus(x):
    """Numerically stable Softplus activation function."""
    return np.log1p(np.exp(np.clip(x, -50, 50)))

def run_swarm_simulation(steps=100):
    """
    Simulates 100 evolutionary steps comparing a baseline swarm against
    the Swarm of Swarms v4.0 specification across three environmental phases:
    1. Normal operation + 'Dark Room' trap (steps 0-39)
    2. Deep space energy scarcity (steps 40-69)
    3. Anomaly surge & high environmental dynamics (steps 70-99)
    """
    t = np.arange(steps)
    
    # Environment profile: available energy (E_avail) and world dynamics (Env_dyn)
    E_avail = np.ones(steps) * 1.0
    E_avail[40:70] = 0.2   # Phase 2: Energy scarcity ("deep space")
    E_avail[70:] = 0.8     # Phase 3: Recovery and anomaly surge
    
    Env_dyn = np.ones(steps) * 0.5
    Env_dyn[40:70] = 0.15  # Low dynamics in deep space
    Env_dyn[70:] = 1.2     # High dynamics and new physical challenges

    # Metrics for the baseline swarm (without v4.0: prone to "dark room" trap and burnout)
    base_I_phys = np.zeros(steps)
    base_E_spent = np.zeros(steps)
    base_efficiency = np.zeros(steps)
    
    # Metrics for the v4.0 specification swarm
    v4_I_phys = np.zeros(steps)
    v4_E_spent = np.zeros(steps)
    v4_efficiency = np.zeros(steps)
    v4_L_evo = np.zeros(steps)
    v4_branches = np.ones(steps)

    # Specification v4.0 hyperparameters
    sigma_meas = 0.15      # Hardware sensor noise floor threshold
    eps = 1e-4             # Denominator regularization constant
    C_target = 0.35        # Criticality target ("pink noise" 1/f regime)
    alpha, beta, gamma, nu, lam, mu = 1.2, 0.8, 0.5, 1.5, 0.4, 0.9
    kappa, xi = 1.1, 1.1

    for i in range(steps):
        # True physical environmental signal + thermal sensor noise
        real_signal = Env_dyn[i] * (0.6 + 0.4 * np.sin(i * 0.25)) + np.random.normal(0, 0.05)
        
        # --- 1. BASELINE SWARM BEHAVIOR ---
        # Prone to getting stuck in predictable loops and burning fixed energy
        if 20 <= i < 40:
            # Trapped in a virtual "dark room": physical shift drops near zero
            dM_base = 0.05 + abs(np.random.normal(0, 0.02))
        else:
            dM_base = max(0.02, real_signal * 0.45)
            
        dE_base = 0.65 + np.random.normal(0, 0.03)  # Fails to throttle energy under scarcity
        dI_base = max(0.0, np.log2(dM_base / sigma_meas))
        
        base_I_phys[i] = dI_base
        base_E_spent[i] = dE_base
        base_efficiency[i] = dI_base / dE_base

        # --- 2. SWARM v4.0 SPECIFICATION BEHAVIOR ---
        # Adaptive activity and branching based on available energy and environmental surprise
        surprise_DKL = abs(real_signal - 0.3) * 0.8
        if i > 0 and surprise_DKL > 0.45 and E_avail[i] > 0.4:
            v4_branches[i] = min(8.0, v4_branches[i-1] + 0.5)  # Branching upon hitting a barrier
        elif E_avail[i] < 0.3:
            v4_branches[i] = max(1.0, v4_branches[i-1] - 0.6)  # Apoptosis of excess branches in deep space
        else:
            v4_branches[i] = v4_branches[i-1] if i > 0 else 1.0

        # Verified physical shift accounting for sensor coherence
        dM_v4 = max(0.05, real_signal * (0.55 + 0.12 * v4_branches[i]))
        dI_v4 = softplus(np.log2(dM_v4 / sigma_meas))
        
        # Energy expenditure adapts to hibernation / branching modes
        dE_v4 = (0.18 + 0.07 * v4_branches[i]) * (0.4 if E_avail[i] < 0.3 else 1.0)
        
        # Mutual information I(Q; P) and autocorrelation in the 1/f criticality zone
        mutual_info = dI_v4 * 0.85
        c_auto = C_target + np.random.normal(0, 0.04)
        
        # Adaptive survival thresholds
        theta_energy = 1.2 * E_avail[i]
        theta_rate = 0.8 * Env_dyn[i]
        
        # Full evolutionary functional L_evo''' with Softplus smooth barriers
        L_val = (
            alpha * surprise_DKL
            + beta * np.log1p(v4_branches[i])
            + gamma * 0.6
            + nu * mutual_info
            - lam * (dE_v4 / (dI_v4 + eps))
            - mu * ((c_auto - C_target) ** 2)
            - kappa * softplus(theta_energy - (dI_v4 / dE_v4))
            - xi * softplus(theta_rate - dI_v4)
        )
        
        v4_I_phys[i] = dI_v4
        v4_E_spent[i] = dE_v4
        v4_efficiency[i] = dI_v4 / dE_v4
        v4_L_evo[i] = L_val

    # === CONSOLE SUMMARY TABLE ===
    print("=" * 78)
    print(f"{'ENVIRONMENT PHASE (STEPS)':<28} | {'BASELINE EFF [Bit/J]':<22} | {'v4.0 EFF [Bit/J]':<20}")
    print("=" * 78)
    phases = [
        ("1. Normal + Trap (0-39)", slice(0, 40)),
        ("2. Energy Scarcity (40-69)", slice(40, 70)),
        ("3. Anomaly Surge (70-99)", slice(70, 100)),
        ("TOTAL FULL CYCLE (0-99)", slice(0, 100))
    ]
    for name, sl in phases:
        b_eff = np.mean(base_efficiency[sl])
        v_eff = np.mean(v4_efficiency[sl])
        gain = ((v_eff - b_eff) / (b_eff + 1e-6)) * 100
        print(f"{name:<28} | {b_eff:>10.2f}            | {v_eff:>10.2f} (+{gain:.0f}%)")
    print("=" * 78)
    print(f"Total verified physical shift: Baseline = {np.sum(base_I_phys):.1f} Bit | v4.0 = {np.sum(v4_I_phys):.1f} Bit")
    print(f"Total energy expenditure:      Baseline = {np.sum(base_E_spent):.1f} J   | v4.0 = {np.sum(v4_E_spent):.1f} J")

    # === PLOTTING RESULTS ===
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    
    # Plot 1: Evolutionary Thermodynamic Efficiency (Bit / J)
    axes[0].plot(t, base_efficiency, label="Baseline Swarm (without v4.0)", color="#d9534f", linestyle="--", linewidth=1.8)
    axes[0].plot(t, v4_efficiency, label="Swarm v4.0 Specification", color="#0275d8", linewidth=2.2)
    axes[0].axvspan(20, 40, color="gray", alpha=0.15, label="'Dark Room' Trap Zone")
    axes[0].axvspan(40, 70, color="orange", alpha=0.15, label="Energy Scarcity (Deep Space)")
    axes[0].set_ylabel("Efficiency [Bit / J]")
    axes[0].set_title("Swarm Evolution Simulation: Thermodynamic Efficiency (Specification v4.0)", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Cumulative Physical Shift (Bit)
    axes[1].plot(t, np.cumsum(base_I_phys), label="Cumulative Shift (Baseline)", color="#d9534f", linestyle="--")
    axes[1].plot(t, np.cumsum(v4_I_phys), label="Cumulative Shift (v4.0)", color="#5cb85c", linewidth=2.2)
    axes[1].set_ylabel("Cumulative [Bit]")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Adaptive Branching (Active Sub-agents) & Functional L_evo'''
    ax3_twin = axes[2].twinx()
    p1 = axes[2].plot(t, v4_branches, label="Active Swarm Branches (Branching / Apoptosis)", color="#f0ad4e", linewidth=2.2)
    p2 = ax3_twin.plot(t, v4_L_evo, label="Spiral Functional L_evo'''", color="#6f42c1", linestyle=":", linewidth=1.8)
    axes[2].set_ylabel("Active Swarm Branches")
    ax3_twin.set_ylabel("L_evo''' Value")
    axes[2].set_xlabel("Evolution Steps (t)")
    
    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    axes[2].legend(lines, labels, loc="upper left", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("swarm_v4_simulation.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    run_swarm_simulation()
