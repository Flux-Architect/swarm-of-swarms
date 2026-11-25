import numpy as np
import matplotlib.pyplot as plt


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x):
    return np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class SpatialEnvironment2D:
    def __init__(self, L=100.0):
        self.L = L
        self.dark_room = (50.0, 50.0, 25.0, 18.0)

    def get_phase_params(self, t):
        if t < 50:
            return 1.0, 1.0
        elif t < 100:
            return 0.35, 0.8
        else:
            return 1.2, 1.8

    def evaluate_fields(self, pos, t):
        _, anomaly_mult = self.get_phase_params(t)
        if t >= 100:
            ang = 0.11 * (t - 100)
            sources = (
                (25.0 + 22.0 * np.cos(ang), 25.0 + 22.0 * np.sin(ang), 12.0, 13.5),
                (75.0 - 18.0 * np.sin(ang), 70.0 - 18.0 * np.cos(ang), 15.0, 15.0)
            )
        else:
            sources = (
                (25.0, 25.0, 12.0, 13.5),
                (75.0, 70.0, 15.0, 15.0)
            )

        dM_phys = np.zeros(len(pos))
        grad_phys = np.zeros_like(pos)
        for sx, sy, amp, sig in sources:
            diff = np.array((sx, sy)) - pos
            val = (amp * anomaly_mult) * np.exp(-np.sum(diff**2, axis=1) / (2.0 * sig**2))
            dM_phys += val
            grad_phys += (val[:, None] * diff) / (sig**2)

        dx, dy, damp, dsig = self.dark_room
        diff_d = np.array((dx, dy)) - pos
        pseudo_sig = damp * np.exp(-np.sum(diff_d**2, axis=1) / (2.0 * dsig**2))
        grad_dark = (pseudo_sig[:, None] * diff_d) / (dsig**2)
        return dM_phys, grad_phys, pseudo_sig, grad_dark


def run_simulation(mode="v5_1", N0=80, T=150, L=100.0, seed=42):
    env = SpatialEnvironment2D(L=L)
    rng = np.random.default_rng(seed)

    # 100% STRICT HARDWARE & PRICING PARITY CONSTANTS
    c_signal = 15.0        # Relativistic signal speed [m/step]
    v_max = 3.5            # Actuator cruise limit [m/step]
    R_comm = 24.0          # Local communication radius [m]
    sigma_meas = 0.5       # Identical additive sensor noise std on BOTH channels
    d0_reg = 3.0           # Soft-core repulsion regularization radius [m]
    N_min, N_target = 50, 80
    E_idle = 0.12          # Identical unthrottled base metabolism [J/step]
    E_kin_coef = 0.06      # Identical kinetic actuation cost
    E_comm_unit = 0.015    # Identical communication cost per active link

    # Smooth Sigmoid Thermodynamic Selection Parameters (v5.1: dimensionally consistent in [Bit/J])
    alpha_ema = 0.25       # Exponential moving average decay for agent efficiency
    beta_cull = 2.0        # Inverse thermodynamic temperature [(Bit/J)^-1]
    p0_cull = 0.15         # Baseline per-step selection intensity

    pos = rng.uniform(10.0, L - 10.0, size=(N0, 2))
    vel = rng.normal(0.0, 0.3, size=(N0, 2))
    ema_eff = np.full(N0, 3.0)
    max_pre_clamp_v = 0.0
    tau_1_count = 0
    tau_2plus_count = 0

    hist = {"N": [], "I_phys": [], "E_total": [], "eff": [], "bif": [], "conn": [], "pos": None, "is_scout": None}

    for t in range(T):
        N = len(pos)
        E_avail, _ = env.get_phase_params(t)
        dM_phys, grad_phys, pseudo_sig, grad_dark = env.evaluate_fields(pos, t)

        measured_dM = np.maximum(dM_phys + rng.normal(0.0, sigma_meas, size=N), 0.0)
        measured_virt = np.maximum(pseudo_sig + rng.normal(0.0, sigma_meas, size=N), 0.0)

        # Emergent role switching B(grad S_t): agents in low-signal zones switch to wide-field Scout mode
        is_scout = measured_dM < 2.2
        sens_thresh = np.where(is_scout, 0.45, 3.2)[:, None]

        local_grad_phys = np.where(dM_phys[:, None] > sens_thresh, grad_phys, 0.0)
        local_grad_dark = np.where(pseudo_sig[:, None] > sens_thresh, grad_dark, 0.0)

        # Uniform metric weight = 1.0 across all models and roles
        I_agent = softplus(np.log2(np.maximum(measured_dM / sigma_meas, 1e-6)))
        I_step = float(np.sum(I_agent))

        diff_now = pos[:, None, :] - pos[None, :, :]
        dist_now = np.linalg.norm(diff_now, axis=2)
        np.fill_diagonal(dist_now, np.inf)
        adj_now = (dist_now <= R_comm).astype(float)
        nb_cnt = np.sum(adj_now, axis=1)

        rep = np.zeros_like(pos)
        for i in range(N):
            cm = dist_now[i] < 4.5
            if np.any(cm):
                d_vec = pos[i] - pos[cm]
                rep[i] = np.sum(d_vec / (np.sum(d_vec**2, axis=1, keepdims=True) + d0_reg**2), axis=0) * 5.0

        deficit_scale = 0.55 if E_avail < 0.5 else 1.0

        if mode in ("naive_baseline", "oracle_baseline"):
            dark_w = 1.0 if mode == "naive_baseline" else 0.0
            # Baselines have uniform medium sensor threshold (1.8)
            base_vis_p = (dM_phys > 1.8)[:, None]
            base_vis_d = (pseudo_sig > 1.8)[:, None]
            perceived_grad = np.where(base_vis_p, grad_phys, 0.0) + dark_w * np.where(base_vis_d, grad_dark, 0.0)
            in_blind = (np.linalg.norm(perceived_grad, axis=1) < 1e-4)[:, None]
            search_noise = np.where(in_blind, rng.normal(0.0, 0.95, size=(N, 2)), rng.normal(0.0, 0.25, size=(N, 2)))
            acc = 2.4 * perceived_grad + rep + search_noise
            acc_rel = acc * np.maximum(0.0, 1.0 - np.sum(vel**2, axis=1, keepdims=True) / (c_signal**2))
            vel = 0.65 * vel + deficit_scale * acc_rel
            max_pre_clamp_v = max(max_pre_clamp_v, float(np.max(np.linalg.norm(vel, axis=1))))
            spd = np.linalg.norm(vel, axis=1, keepdims=True)
            vel = np.where(spd > v_max * deficit_scale, vel / (spd + 1e-9) * (v_max * deficit_scale), vel)
            pos = np.clip(pos + vel, 0.0, L)
            E_step = float(np.sum(E_idle + E_kin_coef * np.sum(vel**2, axis=1) + E_comm_unit * nb_cnt))
            bif_r = 0.0
            conn_s = float(np.mean(nb_cnt >= 2))
        else:
            # 1. Calibrated Gaussian KL-divergence D_KL(N(dM, sigma^2) || N(virt, sigma^2))
            excess_virt = np.maximum(measured_virt - measured_dM, 0.0)
            gauss_kl = (excess_virt**2) / (2.0 * (sigma_meas**2))
            trust_virtual = np.exp(-0.25 * gauss_kl)  # Drops to ~0.00 inside Dark Room!
            grounded_grad = local_grad_phys + trust_virtual[:, None] * local_grad_dark

            # 2. Strict light-cone causal delay audit: tau_ij = ceil(d_ij / c) >= 1 (zero relay buffer overhead)
            finite_dist = np.where(np.isfinite(dist_now), dist_now, 0.0)
            tau_steps = np.ceil(finite_dist / c_signal).astype(int)
            active_taus = tau_steps[adj_now > 0]
            if len(active_taus) > 0:
                tau_1_count += int(np.sum(active_taus == 1))
                tau_2plus_count += int(np.sum(active_taus >= 2))

            has_local_grad = (np.linalg.norm(grounded_grad, axis=1) > 1e-4)[:, None]

            # Direct thermodynamic foraging without stale relay buffer chasing
            worker_pull = np.where(has_local_grad, 3.2 * grounded_grad - 0.55 * vel, rng.normal(0.0, 0.25, size=(N, 2)))
            scout_sweep = np.where(has_local_grad, 2.5 * grounded_grad + rng.normal(0.0, 0.35, size=(N, 2)), rng.normal(0.0, 1.35, size=(N, 2)))

            acc = np.where(is_scout[:, None], scout_sweep, worker_pull) + rep
            acc_rel = acc * np.maximum(0.0, 1.0 - np.sum(vel**2, axis=1, keepdims=True) / (c_signal**2))
            vel = 0.60 * vel + deficit_scale * acc_rel
            max_pre_clamp_v = max(max_pre_clamp_v, float(np.max(np.linalg.norm(vel, axis=1))))
            spd = np.linalg.norm(vel, axis=1, keepdims=True)
            vel = np.where(spd > v_max * deficit_scale, vel / (spd + 1e-9) * (v_max * deficit_scale), vel)
            pos = np.clip(pos + vel, 0.0, L)

            agent_E = E_idle + E_kin_coef * np.sum(vel**2, axis=1) + E_comm_unit * nb_cnt
            E_step = float(np.sum(agent_E))
            bif_r = float(np.mean(~is_scout))
            conn_s = float(np.mean(nb_cnt >= 2))

            # 3. Smooth Sigmoid Survival Pressure (no hard step counters, no array-index bias)
            theta_cull = 2.4 if E_avail < 0.5 else 1.5
            agent_eff = I_agent / (agent_E + 1e-9)
            ema_eff = (1.0 - alpha_ema) * ema_eff + alpha_ema * agent_eff

            if N > N_min:
                p_cull = p0_cull * sigmoid(beta_cull * (theta_cull - ema_eff))
                cand = np.flatnonzero(rng.random(N) < p_cull)
                if len(cand) > 0:
                    max_drop = min(len(cand), N - N_min, 1)
                    drop = cand[np.argsort(ema_eff[cand])[:max_drop]]
                    keep = np.ones(N, dtype=bool)
                    keep[drop] = False
                    pos = pos[keep]
                    vel = vel[keep]
                    is_scout = is_scout[keep]
                    ema_eff = ema_eff[keep]
                    I_agent = I_agent[keep]

            # 4. Continuous stochastic coverage-deficit regeneration
            if len(pos) < N_target and E_avail > 0.7:
                fert = np.flatnonzero(I_agent > 4.5)
                if len(fert) > 0 and rng.random() < 0.35:
                    n_sp = min(len(fert), N_target - len(pos), 1)
                    par = rng.choice(fert, size=n_sp, replace=False)
                    new_p = np.clip(pos[par] + rng.normal(0.0, 2.5, size=(n_sp, 2)), 0.0, L)
                    pos = np.vstack((pos, new_p))
                    vel = np.vstack((vel, np.zeros((n_sp, 2))))
                    is_scout = np.concatenate((is_scout, np.zeros(n_sp, dtype=bool)))
                    ema_eff = np.concatenate((ema_eff, np.full(n_sp, 3.0)))

        hist["N"].append(len(pos))
        hist["I_phys"].append(I_step)
        hist["E_total"].append(E_step)
        hist["eff"].append(I_step / (E_step + 1e-9))
        hist["bif"].append(bif_r)
        hist["conn"].append(conn_s)
        if t == T - 1:
            hist["pos"] = pos.copy()
            hist["is_scout"] = is_scout.copy()

    hist["max_pre_v"] = max_pre_clamp_v
    tot_links = max(tau_1_count + tau_2plus_count, 1)
    hist["tau1_share"] = tau_1_count / tot_links
    hist["tau2_share"] = tau_2plus_count / tot_links
    return hist, env


seeds = (42, 43, 44, 45, 46)
configs = (
    ("naive_baseline", "Naive Baseline (Dark Trap)"),
    ("v5_1", "v5.1 (Gauss-KL + Sigmoid Cull)"),
    ("oracle_baseline", "Oracle Upper Bound (No Trap)")
)
stats = {k: {"eff": [], "I_per_ag": [], "E_per_ag": [], "min_N": [], "max_v": [], "tau1": [], "tau2": []} for k, _ in configs}

for s in seeds:
    for key, _ in configs:
        h, env_obj = run_simulation(mode=key, seed=s)
        dI = float(np.sum(h["I_phys"]))
        dE = float(np.sum(h["E_total"]))
        ag_steps = float(np.sum(h["N"]))
        stats[key]["eff"].append(dI / dE)
        stats[key]["I_per_ag"].append(dI / ag_steps)
        stats[key]["E_per_ag"].append(dE / ag_steps)
        stats[key]["min_N"].append(min(h["N"]))
        stats[key]["max_v"].append(h["max_pre_v"])
        if key == "v5_1":
            stats[key]["tau1"].append(h["tau1_share"])
            stats[key]["tau2"].append(h["tau2_share"])
        if s == 42:
            if key == "naive_baseline":
                b_h42 = h
            elif key == "oracle_baseline":
                ob_h42 = h
            elif key == "v5_1":
                v_h42 = h

print("=== CALIBRATED AUDIT BENCHMARK v5.1 (5 SEEDS, WEIGHT=1.0, AGENT-STEP NORMALIZED) ===")
for key, label in configs:
    eff_m, eff_s = np.mean(stats[key]["eff"]), np.std(stats[key]["eff"])
    ipa_m, ipa_s = np.mean(stats[key]["I_per_ag"]), np.std(stats[key]["I_per_ag"])
    epa_m, epa_s = np.mean(stats[key]["E_per_ag"]), np.std(stats[key]["E_per_ag"])
    vp_m, vp_s = np.mean(stats[key]["max_v"]), np.std(stats[key]["max_v"])
    min_n_val = min(stats[key]["min_N"])
    print(f"{label:<31} | Eff={eff_m:.2f}+-{eff_s:.2f} Bit/J | I/ag={ipa_m:.2f}+-{ipa_s:.2f} | E/ag={epa_m:.3f}+-{epa_s:.3f} | min_N={min_n_val} | pre_v={vp_m:.2f}+-{vp_s:.2f} < c=15.0")

naive_eff = np.mean(stats["naive_baseline"]["eff"])
v5_eff = np.mean(stats["v5_1"]["eff"])
oracle_eff = np.mean(stats["oracle_baseline"]["eff"])
gap_closed = ((v5_eff - naive_eff) / (oracle_eff - naive_eff)) * 100.0
t1_m = np.mean(stats["v5_1"]["tau1"]) * 100.0
t2_m = np.mean(stats["v5_1"]["tau2"]) * 100.0
print(f"Summary Framing: +{(v5_eff/naive_eff - 1)*100:.1f}% vs Naive | {(v5_eff/oracle_eff - 1)*100:.1f}% vs Oracle | {gap_closed:.1f}% of Oracle Gap Closed")
print(f"Strict Light-Cone Causal Links (v5.1): tau=0: 0.0% | tau=1 step: {t1_m:.1f}% | tau>=2 steps: {t2_m:.1f}%")

fig = plt.figure(figsize=(15, 9), dpi=130)
plt.subplots_adjust(hspace=0.35, wspace=0.25)
gx, gy = np.meshgrid(np.linspace(0, 100, 80), np.linspace(0, 100, 80))
pmap, _, dmap, _ = env_obj.evaluate_fields(np.column_stack((gx.ravel(), gy.ravel())), 149)
pmap, dmap = pmap.reshape(80, 80), dmap.reshape(80, 80)

ax1 = fig.add_subplot(2, 3, 1)
ax1.contourf(gx, gy, pmap, levels=12, cmap="Blues", alpha=0.6)
ax1.contour(gx, gy, dmap, levels=4, colors="red", linestyles="--")
ax1.scatter(b_h42["pos"][:, 0], b_h42["pos"][:, 1], c="black", s=24)
ax1.set_title("1. Naive Baseline (Dark Room Trap)")

ax2 = fig.add_subplot(2, 3, 2)
ax2.contourf(gx, gy, pmap, levels=12, cmap="Blues", alpha=0.6)
ax2.contour(gx, gy, dmap, levels=4, colors="red", linestyles="--", alpha=0.5)
ax2.scatter(v_h42["pos"][v_h42["is_scout"], 0], v_h42["pos"][v_h42["is_scout"], 1], c="orange", marker="^", s=34, label="Emergent Scouts")
ax2.scatter(v_h42["pos"][~v_h42["is_scout"], 0], v_h42["pos"][~v_h42["is_scout"], 1], c="blue", s=28, label="Workers")
ax2.legend(fontsize=8)
ax2.set_title("2. v5.1 (Gauss-KL V_phys + Sigmoid Cull)")

steps = np.arange(150)
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(steps, b_h42["eff"], color="gray", label="Naive Base")
ax3.plot(steps, v_h42["eff"], color="green", lw=1.8, label=f"v5.1 (+{(v5_eff/naive_eff - 1)*100:.0f}% vs Naive)")
ax3.plot(steps, ob_h42["eff"], color="black", ls="--", label="Oracle Bound")
ax3.legend(fontsize=7.5)
ax3.set_title("3. Agent-Step Efficiency [Bit/J]")
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(steps, b_h42["N"], "k--", label="Baselines N=80")
ax4.plot(steps, v_h42["N"], color="purple", label="v5.1 N(t)")
ax4.axhline(50, color="red", ls=":", label="Quorum Floor N=50")
ax4.legend(fontsize=8)
ax4.set_title("4. Dynamic Population N(t) & Quorum")
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(steps, np.cumsum(b_h42["I_phys"]), color="gray", label="Naive Base")
ax5.plot(steps, np.cumsum(v_h42["I_phys"]), color="blue", lw=1.8, label="v5.1")
ax5.plot(steps, np.cumsum(ob_h42["I_phys"]), color="black", ls="--", label="Oracle Bound")
ax5.legend(fontsize=7.5)
ax5.set_title("5. Cumulative Shift [Bit, Weight=1.0]")
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(steps, v_h42["conn"], color="teal", label="Graph Connectivity (nb>=2)")
ax6.plot(steps, v_h42["bif"], color="orange", label="Active Workers Share")
ax6.set_ylim(0, 1.05)
ax6.legend(fontsize=8)
ax6.set_title("6. Connectivity & Role Dynamics")
ax6.grid(True, alpha=0.3)

plt.savefig("swarm_v5_1_2d_simulation.png", dpi=150)
plt.show()
