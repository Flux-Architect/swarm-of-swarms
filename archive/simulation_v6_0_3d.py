import csv
import json
import os
import warnings
import zipfile
import numpy as np
import matplotlib.pyplot as plt

# Suppress internal Matplotlib 3.10 mplot3d float-to-int deprecation noise during 3D scatter projection
warnings.filterwarnings("ignore", category=DeprecationWarning)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x):
    return np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class SpatialEnvironment3D:
    """
    3D volumetric environment [0, L]^3 with non-coplanar mobile physical anomalies
    and a central high-amplitude pseudo-signal attractor (Dark Room Trap).
    """

    def __init__(self, L=100.0):
        self.L = L
        # (x, y, z, amplitude, sigma)
        self.dark_room = (50.0, 50.0, 50.0, 25.0, 20.0)

    def get_phase_params(self, t):
        """
        Returns (phase_supply, anomaly_mult):
        phase_supply is a dimensionless environmental regime multiplier/flag
        (1.0 nominal, 0.35 lean/crisis phase, 1.2 dynamic surge phase), NOT stored energy.
        """
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
                (
                    25.0 + 22.0 * np.cos(ang),
                    25.0 + 22.0 * np.sin(ang),
                    30.0 + 18.0 * np.sin(0.8 * ang),
                    12.0,
                    15.5,
                ),
                (
                    75.0 - 18.0 * np.sin(ang),
                    70.0 - 18.0 * np.cos(ang),
                    70.0 - 18.0 * np.cos(0.8 * ang),
                    15.0,
                    17.0,
                ),
            )
        else:
            sources = (
                (25.0, 25.0, 30.0, 12.0, 15.5),
                (75.0, 70.0, 70.0, 15.0, 17.0),
            )

        dM_phys = np.zeros(len(pos))
        grad_phys = np.zeros_like(pos)
        for sx, sy, sz, amp, sig in sources:
            diff = np.array((sx, sy, sz)) - pos
            val = (amp * anomaly_mult) * np.exp(-np.sum(diff**2, axis=1) / (2.0 * sig**2))
            dM_phys += val
            grad_phys += (val[:, None] * diff) / (sig**2)

        dx, dy, dz, damp, dsig = self.dark_room
        diff_d = np.array((dx, dy, dz)) - pos
        pseudo_sig = damp * np.exp(-np.sum(diff_d**2, axis=1) / (2.0 * dsig**2))
        grad_dark = (pseudo_sig[:, None] * diff_d) / (dsig**2)
        return dM_phys, grad_phys, pseudo_sig, grad_dark


def run_simulation_3d(mode="v6_0", N0=80, T=150, L=100.0, seed=42):
    env = SpatialEnvironment3D(L=L)
    rng = np.random.default_rng(seed)

    # 100% STRICT HARDWARE, DENSITY & QUADRATIC LINK-COST PARITY CONSTANTS
    c_signal = 15.0        # Relativistic signal speed [m/step]
    v_max = 3.5            # Actuator cruise limit [m/step]
    R_ref_2d = 24.0        # Reference scale for Quadratic Link-Cost Model [m]
    # Volume-matched 3D communication radius preserving expected neighbor degree (~14.3)
    R_comm = L * ((3.0 * np.pi * (R_ref_2d**2)) / (4.0 * np.pi * (L**2))) ** (1.0 / 3.0)
    scale_3d = R_comm / R_ref_2d
    d_rep_cut = 4.5 * scale_3d
    d0_reg = 3.0 * scale_3d
    spawn_std = 2.5 * scale_3d

    sigma_meas = 0.5       # Identical additive sensor noise std on BOTH channels
    N_min, N_target = 50, 80
    E_idle = 0.12          # Identical unthrottled base metabolism [J/step]
    E_kin_coef = 0.06      # Identical kinetic actuation cost
    E_comm_unit = 0.015    # Base communication cost at reference scale R_ref_2d

    # Smooth Sigmoid Thermodynamic Selection Parameters (consistent in [Bit/J])
    alpha_ema = 0.25       # Exponential moving average decay for agent efficiency
    beta_cull = 2.0        # Inverse thermodynamic temperature [(Bit/J)^-1]
    p0_cull = 0.15         # Baseline per-step selection intensity

    pos = rng.uniform(10.0, L - 10.0, size=(N0, 3))
    vel = rng.normal(0.0, 0.3, size=(N0, 3))
    ema_eff = np.full(N0, 3.0)
    max_pre_clamp_v = 0.0
    tau_1_count = 0
    tau_2plus_count = 0
    boundary_hits = 0
    deg_all_steps = []
    deg_center_steps = []

    hist = {
        "t": [],
        "N": [],
        "I_phys": [],
        "E_total": [],
        "eff": [],
        "bif": [],
        "conn": [],
        "pos": None,
        "is_scout": None,
    }

    for t in range(T):
        N = len(pos)
        phase_supply, _ = env.get_phase_params(t)
        dM_phys, grad_phys, pseudo_sig, grad_dark = env.evaluate_fields(pos, t)

        measured_dM = np.maximum(dM_phys + rng.normal(0.0, sigma_meas, size=N), 0.0)
        if mode == "oracle_baseline":
            # Oracle Upper Bound has Dark Room virtual pseudo-signal strictly disabled (trust_virtual = 0.0)
            measured_virt = np.zeros(N)
        else:
            measured_virt = np.maximum(pseudo_sig + rng.normal(0.0, sigma_meas, size=N), 0.0)

        # Emergent role switching B(grad S_t): agents in low-signal zones switch to wide-field Scout mode
        is_scout = measured_dM < 2.2
        sens_thresh = np.where(is_scout, 0.45, 3.2)[:, None]

        local_grad_phys = np.where(dM_phys[:, None] > sens_thresh, grad_phys, 0.0)
        local_grad_dark = np.where(pseudo_sig[:, None] > sens_thresh, grad_dark, 0.0)

        # Uniform metric weight = 1.0 across all models and roles
        I_agent = softplus(np.log2(np.maximum(measured_dM / sigma_meas, 1e-6)))
        I_step = float(np.sum(I_agent))

        # Vectorized 3D pairwise distances & Quadratic Link-Cost Model: E_comm(r) = E_unit * (1 + r^2 / R_ref^2)
        diff_now = pos[:, None, :] - pos[None, :, :]
        dist_sq = np.sum(diff_now**2, axis=2)
        dist_now = np.sqrt(dist_sq)
        np.fill_diagonal(dist_now, np.inf)

        adj_mask = dist_now <= R_comm
        adj_now = adj_mask.astype(float)
        nb_cnt = np.sum(adj_now, axis=1)

        deg_all_steps.append(float(np.mean(nb_cnt)))
        in_center = np.all((pos >= 20.0) & (pos <= 80.0), axis=1)
        if np.any(in_center):
            deg_center_steps.append(float(np.mean(nb_cnt[in_center])))

        # Strict 3D light-cone causal delay audit metric: tau_ij = ceil(d_ij / c) >= 1
        finite_dist = np.where(np.isfinite(dist_now), dist_now, 0.0)
        tau_steps = np.ceil(finite_dist / c_signal).astype(int)
        active_taus = tau_steps[adj_mask]
        if len(active_taus) > 0:
            tau_1_count += int(np.sum(active_taus == 1))
            tau_2plus_count += int(np.sum(active_taus >= 2))

        # Quadratic Link-Cost Model: E_comm(r) = E_comm_unit * (1 + r^2 / R_ref_2d^2), r in (0, R_comm]
        finite_dist_sq = np.where(adj_mask, dist_sq, 0.0)
        link_cost_matrix = np.where(adj_mask, E_comm_unit * (1.0 + finite_dist_sq / (R_ref_2d**2)), 0.0)
        comm_energy_agent = np.sum(link_cost_matrix, axis=1)

        # Fully vectorized 3D soft-core repulsion
        rep_mask = dist_now < d_rep_cut
        inv_denom = np.where(rep_mask, 1.0 / (dist_sq + d0_reg**2), 0.0)
        rep = np.sum(diff_now * inv_denom[:, :, None], axis=1) * (5.0 * scale_3d)

        deficit_scale = 0.55 if phase_supply < 0.5 else 1.0
        # Unified causal & thermodynamic foraging pipeline across ALL 3 modes (100% pipeline parity)
        if mode == "naive_baseline":
            # Naive baseline blindly trusts virtual Dark Room gradient (trust_virtual = 1.0)
            trust_virtual = np.ones(N)
            grounded_grad = local_grad_phys + local_grad_dark
        elif mode == "oracle_baseline":
            # Oracle Upper Bound: Dark Room pseudo-signal is strictly disabled (trust_virtual = 0.0)
            trust_virtual = np.zeros(N)
            grounded_grad = local_grad_phys
        else:
            # v6.0: One-Sided Gaussian Trust Penalty = max(virt - dM, 0)^2 / (2 * sigma^2) (not symmetric D_KL)
            excess_virt = np.maximum(measured_virt - measured_dM, 0.0)
            one_sided_gauss_pen = (excess_virt**2) / (2.0 * (sigma_meas**2))
            trust_virtual = np.exp(-0.25 * one_sided_gauss_pen)  # Drops to ~0.00 inside Dark Room!
            grounded_grad = local_grad_phys + trust_virtual[:, None] * local_grad_dark

        has_local_grad = (np.linalg.norm(grounded_grad, axis=1) > 1e-4)[:, None]

        # Direct 3D thermodynamic foraging without stale relay buffer chasing
        worker_pull = np.where(has_local_grad, 3.2 * grounded_grad - 0.55 * vel, rng.normal(0.0, 0.25, size=(N, 3)))
        scout_sweep = np.where(
            has_local_grad,
            2.5 * grounded_grad + rng.normal(0.0, 0.35, size=(N, 3)),
            rng.normal(0.0, 1.35, size=(N, 3)),
        )

        acc = np.where(is_scout[:, None], scout_sweep, worker_pull) + rep
        acc_rel = acc * np.maximum(0.0, 1.0 - np.sum(vel**2, axis=1, keepdims=True) / (c_signal**2))
        vel = 0.60 * vel + deficit_scale * acc_rel
        max_pre_clamp_v = max(max_pre_clamp_v, float(np.max(np.linalg.norm(vel, axis=1))))
        spd = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = np.where(spd > v_max * deficit_scale, vel / (spd + 1e-9) * (v_max * deficit_scale), vel)
        pos = np.clip(pos + vel, 0.0, L)

        # Boundary contact check: fraction of steps where >=1 agent touches 0.0 or L.0 after clip
        if np.any((pos <= 0.0) | (pos >= L)):
            boundary_hits += 1

        agent_E = E_idle + E_kin_coef * np.sum(vel**2, axis=1) + comm_energy_agent
        E_step = float(np.sum(agent_E))
        bif_r = float(np.mean(~is_scout))
        conn_s = float(np.mean(nb_cnt >= 2))

        # Smooth Sigmoid Survival Pressure (identical across all 3 modes; max_drop=1 prevents cascade collapse)
        theta_cull = 2.4 if phase_supply < 0.5 else 1.5
        agent_eff = I_agent / (agent_E + 1e-9)
        ema_eff = (1.0 - alpha_ema) * ema_eff + alpha_ema * agent_eff

        if N > N_min:
            p_cull = p0_cull * sigmoid(beta_cull * (theta_cull - ema_eff))
            cand = np.flatnonzero(rng.random(N) < p_cull)
            if len(cand) > 0:
                # Intentional smooth single-agent step-wise culling (max_drop = 1 per step)
                max_drop = min(len(cand), N - N_min, 1)
                drop = cand[np.argsort(ema_eff[cand])[:max_drop]]
                keep = np.ones(N, dtype=bool)
                keep[drop] = False
                pos = pos[keep]
                vel = vel[keep]
                is_scout = is_scout[keep]
                ema_eff = ema_eff[keep]
                I_agent = I_agent[keep]

        # Continuous stochastic 3D coverage-deficit regeneration (identical across all 3 modes)
        if len(pos) < N_target and phase_supply > 0.7:
            fert = np.flatnonzero(I_agent > 4.5)
            if len(fert) > 0 and rng.random() < 0.35:
                n_sp = min(len(fert), N_target - len(pos), 1)
                par = rng.choice(fert, size=n_sp, replace=False)
                new_p = np.clip(pos[par] + rng.normal(0.0, spawn_std, size=(n_sp, 3)), 0.0, L)
                pos = np.vstack((pos, new_p))
                vel = np.vstack((vel, np.zeros((n_sp, 3))))
                is_scout = np.concatenate((is_scout, np.zeros(n_sp, dtype=bool)))
                ema_eff = np.concatenate((ema_eff, np.full(n_sp, 3.0)))

        hist["t"].append(int(t))
        hist["N"].append(int(len(pos)))
        hist["I_phys"].append(float(I_step))
        hist["E_total"].append(float(E_step))
        hist["eff"].append(float(I_step / (E_step + 1e-9)))
        hist["bif"].append(float(bif_r))
        hist["conn"].append(float(conn_s))
        if t == T - 1:
            hist["pos"] = pos.copy()
            hist["is_scout"] = is_scout.copy()

    hist["max_pre_v"] = float(max_pre_clamp_v)
    hist["R_comm_3d"] = float(R_comm)
    hist["R_ref_2d"] = float(R_ref_2d)
    hist["L"] = float(L)
    hist["T"] = int(T)
    hist["tau1_count"] = int(tau_1_count)
    hist["tau2_count"] = int(tau_2plus_count)
    tot_links = max(tau_1_count + tau_2plus_count, 1)
    hist["tau1_share"] = float(tau_1_count / tot_links)
    hist["tau2_share"] = float(tau_2plus_count / tot_links)
    hist["boundary_share"] = float(boundary_hits / T)
    hist["mean_deg_all"] = float(np.mean(deg_all_steps)) if deg_all_steps else 0.0
    hist["mean_deg_center"] = float(np.mean(deg_center_steps)) if deg_center_steps else 0.0
    return hist, env


import os
import zipfile

seeds = (42, 43, 44, 45, 46)
configs = (
    ("naive_baseline", "3D Naive Baseline (Dark Trap)"),
    ("v6_0", "3D v6.0 (One-Sided Gauss + Cull)"),
    ("oracle_baseline", "3D Oracle Upper Bound (No Trap)"),
)
stats = {
    k: {
        "eff": [],
        "I_per_ag": [],
        "E_per_ag": [],
        "min_N": [],
        "max_v": [],
        "tau1": [],
        "tau2": [],
        "bound": [],
        "deg_all": [],
        "deg_center": [],
    }
    for k, _ in configs
}

out_dir = "audit_benchmark_v6.0_3d"
os.makedirs(out_dir, exist_ok=True)
csv_rows = []

print("=== CALIBRATED 3D AUDIT BENCHMARK v6.0 (5 SEEDS, R_comm=35.09m, QUADRATIC LINK-COST MODEL) ===")
header = (
    f"{'Seed':<4} | {'Mode':<15} | {'Eff':<6} | {'I/ag':<5} | {'E/ag':<5} | {'minN':<4} | "
    f"{'pre_v':<5} | {'c=15':<4} | {'tau1%':<5} | {'tau2%':<5} | {'bnd%':<5} | {'degAll':<6} | {'degCtr':<6}"
)
print(header)
print("-" * len(header))

for s in seeds:
    for key, _ in configs:
        h, env_obj = run_simulation_3d(mode=key, seed=s)
        dI = float(np.sum(h["I_phys"]))
        dE = float(np.sum(h["E_total"]))
        ag_steps = float(np.sum(h["N"]))
        eff_val = dI / dE
        ipa_val = dI / ag_steps
        epa_val = dE / ag_steps
        min_n_val = int(min(h["N"]))
        final_n_val = int(h["N"][-1])

        stats[key]["eff"].append(eff_val)
        stats[key]["I_per_ag"].append(ipa_val)
        stats[key]["E_per_ag"].append(epa_val)
        stats[key]["min_N"].append(min_n_val)
        stats[key]["max_v"].append(h["max_pre_v"])
        stats[key]["tau1"].append(h["tau1_share"] * 100.0)
        stats[key]["tau2"].append(h["tau2_share"] * 100.0)
        stats[key]["bound"].append(h["boundary_share"] * 100.0)
        stats[key]["deg_all"].append(h["mean_deg_all"])
        stats[key]["deg_center"].append(h["mean_deg_center"])

        c_ok = "OK" if h["max_pre_v"] < 15.0 else "FAIL"
        print(
            f"{s:<4} | {key:<15} | {eff_val:<6.2f} | {ipa_val:<5.2f} | {epa_val:<5.3f} | {min_n_val:<4} | "
            f"{h['max_pre_v']:<5.2f} | {c_ok:<4} | {h['tau1_share']*100:<5.1f} | {h['tau2_share']*100:<5.1f} | "
            f"{h['boundary_share']*100:<5.1f} | {h['mean_deg_all']:<6.2f} | {h['mean_deg_center']:<6.2f}"
        )

        # Export individual JSON audit artifact per (seed, mode)
        audit_payload = {
            "audit_id": "v6.0_3d_control_run",
            "L": h["L"],
            "T": h["T"],
            "seed": int(s),
            "mode": key,
            "R_comm": round(h["R_comm_3d"], 4),
            "R_ref": h["R_ref_2d"],
            "arrays": {
                "t": h["t"],
                "N": h["N"],
                "I_phys": [round(x, 6) for x in h["I_phys"]],
                "E_total": [round(x, 6) for x in h["E_total"]],
            },
            "per_step_checks": {
                "max_pre_v": round(h["max_pre_v"], 6),
                "light_cone_tau0_share": 0.0,
                "light_cone_tau1_count": h["tau1_count"],
                "light_cone_tau2plus_count": h["tau2_count"],
            },
            "enriched_metrics": {
                "mean_deg_all": round(h["mean_deg_all"], 4),
                "mean_deg_center": round(h["mean_deg_center"], 4),
                "boundary_share": round(h["boundary_share"], 4),
                "min_N_over_T": min_n_val,
                "final_N": final_n_val,
            },
        }
        if key == "oracle_baseline":
            audit_payload["oracle_config"] = {
                "dark_room_disabled": True,
                "trust_virtual_zero": True,
            }

        json_path = os.path.join(out_dir, f"audit_seed{s}_mode_{key}.json")
        with open(json_path, "w", encoding="utf-8", newline="\n") as jf:
            json.dump(audit_payload, jf, indent=2)

        csv_rows.append(
            {
                "seed": int(s),
                "mode": key,
                "sum_I": round(dI, 4),
                "sum_E": round(dE, 4),
                "eff": round(eff_val, 4),
                "I_per_ag": round(ipa_val, 4),
                "E_per_ag": round(epa_val, 4),
                "min_N": min_n_val,
                "max_pre_v": round(h["max_pre_v"], 4),
                "tau1_share": round(h["tau1_share"] * 100.0, 2),
                "tau2_share": round(h["tau2_share"] * 100.0, 2),
                "boundary_share": round(h["boundary_share"] * 100.0, 2),
                "mean_deg_all": round(h["mean_deg_all"], 2),
                "mean_deg_center": round(h["mean_deg_center"], 2),
                "final_N": final_n_val,
            }
        )

        if s == 42:
            if key == "naive_baseline":
                b_h42 = h
            elif key == "oracle_baseline":
                ob_h42 = h
            elif key == "oracle_baseline":
                ob_h42 = h
            elif key == "v6_0":
                v_h42 = h

# Write unified summary.csv
csv_path = os.path.join(out_dir, "summary.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as cf:
    writer = csv.DictWriter(cf, fieldnames=list(csv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(csv_rows)

# Bundle all 15 JSON + summary.csv into a single archive for easy download in JupyterLite
zip_filename = "audit_benchmark_v6.0_3d.zip"
with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(os.listdir(out_dir)):
        zf.write(os.path.join(out_dir, fname), arcname=os.path.join(out_dir, fname))

print("\n=== AGGREGATED 5-SEED SUMMARY (SD OVER 5 SEEDS) ===")
for key, label in configs:
    eff_m, eff_s = np.mean(stats[key]["eff"]), np.std(stats[key]["eff"])
    ipa_m, ipa_s = np.mean(stats[key]["I_per_ag"]), np.std(stats[key]["I_per_ag"])
    epa_m, epa_s = np.mean(stats[key]["E_per_ag"]), np.std(stats[key]["E_per_ag"])
    vp_m, vp_s = np.mean(stats[key]["max_v"]), np.std(stats[key]["max_v"])
    min_n_val = min(stats[key]["min_N"])
    print(
        f"{label:<33} | Eff={eff_m:.2f}+-{eff_s:.2f} Bit/J | "
        f"I/ag={ipa_m:.2f}+-{ipa_s:.2f} | E/ag={epa_m:.3f}+-{epa_s:.3f} | "
        f"min_N={min_n_val} | pre_v={vp_m:.2f}+-{vp_s:.2f} < c=15.0"
    )

naive_eff = float(np.mean(stats["naive_baseline"]["eff"]))
v6_eff = float(np.mean(stats["v6_0"]["eff"]))
oracle_eff = float(np.mean(stats["oracle_baseline"]["eff"]))
gap_closed = ((v6_eff - naive_eff) / (oracle_eff - naive_eff)) * 100.0
constraint_status = "PASS" if (v6_eff <= oracle_eff and gap_closed <= 100.0) else "FAIL"

print(
    f"Summary: +{(v6_eff / naive_eff - 1) * 100:.1f}% vs Naive | "
    f"{(v6_eff / oracle_eff - 1) * 100:.1f}% vs Oracle | {gap_closed:.1f}% of Oracle Gap Closed"
)
print(f"Constraint: Eff(v6.0) <= Eff(Oracle)  ->  [{constraint_status}]")
print(f"Exported 15 JSON + summary.csv into '{out_dir}/' and '{zip_filename}'")

# 6-Panel 3D + Telemetry Figure (JupyterLite / Agg safe)
fig = plt.figure(figsize=(16, 9.5), dpi=130)
plt.subplots_adjust(hspace=0.35, wspace=0.28)

ang_end = 0.11 * (149 - 100)
src1_end = (25.0 + 22.0 * np.cos(ang_end), 25.0 + 22.0 * np.sin(ang_end), 30.0 + 18.0 * np.sin(0.8 * ang_end))
src2_end = (75.0 - 18.0 * np.sin(ang_end), 70.0 - 18.0 * np.cos(ang_end), 70.0 - 18.0 * np.cos(0.8 * ang_end))

# Panel 1: 3D Naive Baseline trapped in volumetric Dark Room
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
ax1.scatter(b_h42["pos"][:, 0], b_h42["pos"][:, 1], b_h42["pos"][:, 2], c="black", s=22, alpha=0.8, label="Naive Agents")
ax1.scatter([50.0], [50.0], [50.0], c="red", marker="D", s=110, label="Dark Trap (50,50,50)")
ax1.scatter([src1_end[0], src2_end[0]], [src1_end[1], src2_end[1]], [src1_end[2], src2_end[2]], c="dodgerblue", marker="*", s=130, label="True 3D Sources")
ax1.set_xlim(0, 100); ax1.set_ylim(0, 100); ax1.set_zlim(0, 100)
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
ax1.legend(fontsize=7, loc="upper left")
ax1.set_title("1. 3D Naive Baseline (Dark Trap)")

# Panel 2: 3D v6.0 Self-Organized Swarm tracking non-coplanar sources
ax2 = fig.add_subplot(2, 3, 2, projection="3d")
sc_m = v_h42["is_scout"]
ax2.scatter(v_h42["pos"][sc_m, 0], v_h42["pos"][sc_m, 1], v_h42["pos"][sc_m, 2], c="orange", marker="^", s=32, label="Emergent Scouts")
ax2.scatter(v_h42["pos"][~sc_m, 0], v_h42["pos"][~sc_m, 1], v_h42["pos"][~sc_m, 2], c="blue", s=26, alpha=0.85, label="Workers")
ax2.scatter([50.0], [50.0], [50.0], c="red", marker="D", s=90, alpha=0.5, label="Dark Trap")
ax2.scatter([src1_end[0], src2_end[0]], [src1_end[1], src2_end[1]], [src1_end[2], src2_end[2]], c="deepskyblue", marker="*", s=130, label="True 3D Sources")
ax2.set_xlim(0, 100); ax2.set_ylim(0, 100); ax2.set_zlim(0, 100)
ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
ax2.legend(fontsize=7, loc="upper left")
ax2.set_title("2. 3D v6.0 (One-Sided Gauss + Cull)")

# Panel 3: Agent-Step Efficiency [Bit/J]
steps = np.arange(150)
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(steps, b_h42["eff"], color="gray", label="3D Naive Base")
ax3.plot(steps, v_h42["eff"], color="green", lw=1.8, label=f"3D v6.0 (+{(v6_eff / naive_eff - 1) * 100:.0f}% vs Naive)")
ax3.plot(steps, ob_h42["eff"], color="black", ls="--", label="3D Oracle Bound")
ax3.legend(fontsize=7.5)
ax3.set_title("3. 3D Agent-Step Efficiency [Bit/J]")
ax3.grid(True, alpha=0.3)

# Panel 4: Dynamic Population N(t) & Quorum Floor
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(steps, b_h42["N"], color="gray", ls="-.", label="3D Naive N(t)")
ax4.plot(steps, ob_h42["N"], "k--", label="3D Oracle N(t)")
ax4.plot(steps, v_h42["N"], color="purple", label="3D v6.0 N(t)")
ax4.axhline(50, color="red", ls=":", label="Quorum Floor N=50")
ax4.legend(fontsize=7.5)
ax4.set_title("4. Dynamic Population N(t) & Quorum")
ax4.grid(True, alpha=0.3)

# Panel 5: Cumulative Information Shift [Bit]
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(steps, np.cumsum(b_h42["I_phys"]), color="gray", label="3D Naive Base")
ax5.plot(steps, np.cumsum(v_h42["I_phys"]), color="blue", lw=1.8, label="3D v6.0")
ax5.plot(steps, np.cumsum(ob_h42["I_phys"]), color="black", ls="--", label="3D Oracle Bound")
ax5.legend(fontsize=7.5)
ax5.set_title("5. Cumulative 3D Shift [Bit, Weight=1.0]")
ax5.grid(True, alpha=0.3)

# Panel 6: 3D Graph Connectivity & Role Dynamics
ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(steps, v_h42["conn"], color="teal", label="3D Connectivity (nb>=2)")
ax6.plot(steps, v_h42["bif"], color="orange", label="Active Workers Share")
ax6.set_ylim(0, 1.05)
ax6.legend(fontsize=8)
ax6.set_title("6. 3D Connectivity & Role Dynamics")
ax6.grid(True, alpha=0.3)

plt.savefig("swarm_v6_0_3d_simulation.png", dpi=150)
plt.show()
