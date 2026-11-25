import argparse
import csv
import io
import json
import os
import platform
import sys
import time
import zipfile
import numpy as np


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def binary_entropy(p):
    p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(p_c * np.log(p_c) + (1.0 - p_c) * np.log(1.0 - p_c))


def build_action_basis(dim):
    """Isotropic unit action directions: 8 in 2D, 14 in 3D."""
    if dim == 2:
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        return np.stack((np.cos(angles), np.sin(angles)), axis=1)
    axes = np.vstack((np.eye(3), -np.eye(3)))
    diags = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=float
    ) / np.sqrt(3.0)
    return np.vstack((axes, diags))


class NonStationaryEnvironment:
    """
    Unified d-dimensional (d in {2, 3}) non-stationary environment:
    - Depleting & regenerating real Gaussian sources with capacity C_k(t) in [0.05, 1.0]
    - Static correlated decoy at domain center leaking eta_leak * D into the physical field
    - Three-phase thermodynamic & spatial regime (nominal, energy winter, turbulent drift)
    """
    def __init__(self, dim=2, L=100.0, seed=42):
        self.dim = dim
        self.L = L
        self.rng = np.random.default_rng(seed)
        if dim == 2:
            self.src_pos = np.array([[22.0, 25.0], [78.0, 72.0]], dtype=float)
            self.src_amp = np.array([15.0, 17.0], dtype=float)
            self.src_sig = np.array([13.0, 14.5], dtype=float)
            self.decoy_pos = np.array([50.0, 50.0], dtype=float)
        else:
            self.src_pos = np.array([[22.0, 25.0, 24.0], [78.0, 72.0, 76.0]], dtype=float)
            self.src_amp = np.array([17.0, 19.0], dtype=float)
            self.src_sig = np.array([15.0, 16.5], dtype=float)
            self.decoy_pos = np.array([50.0, 50.0, 50.0], dtype=float)

        self.capacity = np.array([1.0, 1.0], dtype=float)
        self.decoy_amp = 26.0
        self.decoy_sig = 17.0
        self.eta_leak = 0.45

    def get_phase(self, t):
        if t < 50:
            return 1.0, 1.0      # E_avail, signal_mult
        elif t < 100:
            return 0.35, 0.85    # Energy winter
        else:
            return 1.15, 1.45    # Turbulent relocation

    def step_dynamics(self, agent_pos, t):
        for k in range(len(self.src_pos)):
            d2 = np.sum((agent_pos - self.src_pos[k])**2, axis=1)
            uptake = np.sum(np.exp(-d2 / (2.0 * self.src_sig[k]**2)))
            self.capacity[k] = np.clip(
                self.capacity[k] - 0.0021 * uptake + 0.060 * (1.0 - self.capacity[k]),
                0.05, 1.0
            )
            if t >= 100 and (t % 16 == 0):
                new_loc = self.rng.uniform(16.0, self.L - 16.0, size=self.dim)
                while np.linalg.norm(new_loc - self.decoy_pos) < 24.0:
                    new_loc = self.rng.uniform(16.0, self.L - 16.0, size=self.dim)
                self.src_pos[k] = new_loc
            elif self.capacity[k] < 0.15:
                jump = self.rng.uniform(16.0, self.L - 16.0, size=self.dim)
                self.src_pos[k] = 0.65 * self.src_pos[k] + 0.35 * jump

    def evaluate(self, pos, t):
        _, mult = self.get_phase(t)
        I_true = np.zeros(len(pos))
        grad_true = np.zeros_like(pos)
        for k in range(len(self.src_pos)):
            diff = self.src_pos[k] - pos
            eff_amp = self.src_amp[k] * self.capacity[k] * mult
            val = eff_amp * np.exp(-np.sum(diff**2, axis=1) / (2.0 * self.src_sig[k]**2))
            I_true += val
            grad_true += (val[:, None] * diff) / (self.src_sig[k]**2)

        diff_d = self.decoy_pos - pos
        D_val = self.decoy_amp * np.exp(-np.sum(diff_d**2, axis=1) / (2.0 * self.decoy_sig**2))
        grad_D = (D_val[:, None] * diff_d) / (self.decoy_sig**2)

        phys_field = I_true + self.eta_leak * D_val
        grad_phys_field = grad_true + self.eta_leak * grad_D
        return I_true, phys_field, grad_phys_field


def run_simulation_v7(
    dim=2,
    mode="efe_full",
    seed=42,
    N0=80,
    T=150,
    L=100.0,
    use_relay=True,
    cull_regen=True,
    epist_weight=1.2,
    inference_enabled=True,
    baseline_kl=False
):
    env = NonStationaryEnvironment(dim=dim, L=L, seed=seed)
    rng = np.random.default_rng(seed + 1000 * dim)

    # Identical physical constants across 2D and 3D
    c_sig = 15.0
    v_max = 3.5
    R_comm = 32.0
    sigma_m = 0.45
    sigma_g = 0.08
    r_trap = 17.0
    E_idle = 0.12
    E_kin = 0.06
    E_comm = 0.015
    N_min, N_target = 50, 80

    pos = rng.uniform(10.0, L - 10.0, size=(N0, dim))
    vel = rng.normal(0.0, 0.3, size=(N0, dim))
    dwell_true = np.zeros(N0)
    dwell_obs = np.zeros(N0)
    stagnation = np.zeros(N0)
    belief_src = np.full(N0, 0.5)
    prev_y_phys = np.zeros(N0)
    prev_pos = pos.copy()

    max_delay = int(np.ceil(L * np.sqrt(dim) / c_sig)) + 2
    pos_buf = [pos.copy() for _ in range(max_delay)]
    val_buf = [np.zeros(N0) for _ in range(max_delay)]
    valid_buf = [np.ones(N0, dtype=bool) for _ in range(max_delay)]

    action_dirs = build_action_basis(dim)
    n_dirs = len(action_dirs)
    dir_norm = n_dirs / 8.0
    softmax_beta = 3.0 * (np.log(n_dirs) / np.log(8.0))

    tot_I, tot_E = 0.0, 0.0
    trap_steps, ag_steps = 0, 0
    births, deaths = 0, 0
    max_v_invariant = 0.0

    for t in range(T):
        N = len(pos)
        E_avail, _ = env.get_phase(t)
        I_true, phys_field, grad_phys = env.evaluate(pos, t)

        # Ground-truth evaluation metrics ONLY (never exposed to agent controllers)
        # Strict noise-floor gating (Issue #6): sub-noise field yields 0.0 Bit
        spd_now = np.linalg.norm(vel, axis=1)
        dwell_true = np.where((spd_now < 0.6) & (I_true > 1.5), dwell_true + 1.0, np.maximum(0.0, dwell_true - 1.5))
        novelty_true = np.exp(-0.08 * dwell_true)
        raw_I_true = softplus(np.log2(np.maximum(I_true / sigma_m, 1e-6))) * novelty_true
        I_true_agent = np.where(I_true > sigma_m, raw_I_true, 0.0)

        # Unified noisy single-channel physical sensor protocol across ALL agent modes (Zero Data Leakage)
        y_phys = np.maximum(phys_field + rng.normal(0.0, sigma_m, size=N), 0.0)
        g_obs = grad_phys + rng.normal(0.0, sigma_g, size=(N, dim))

        # Observable novelty & perceived signal (strictly from y_phys with noise-floor gate)
        dwell_obs = np.where((spd_now < 0.6) & (y_phys > 2.2), dwell_obs + 1.0, np.maximum(0.0, dwell_obs - 1.5))
        novelty_obs = np.exp(-0.08 * dwell_obs)
        raw_I_obs = softplus(np.log2(np.maximum(y_phys / sigma_m, 1e-6))) * novelty_obs
        I_obs_agent = np.where(y_phys > sigma_m, raw_I_obs, 0.0)

        # Spatial interaction graph & safe-distance repulsion
        diff_mat = pos[:, None, :] - pos[None, :, :]
        dist_mat = np.linalg.norm(diff_mat, axis=2)
        np.fill_diagonal(dist_mat, np.inf)
        adj = (dist_mat <= R_comm).astype(float)
        nb_cnt = np.sum(adj, axis=1)

        rep = np.zeros_like(pos)
        close_mask = dist_mat < 4.4
        for i in range(N):
            if np.any(close_mask[i]):
                dv = pos[i] - pos[close_mask[i]]
                rep[i] = np.sum(dv / (np.sum(dv**2, axis=1, keepdims=True) + 9.0), axis=0) * 4.5

        deficit_scale = 0.55 if E_avail < 0.5 else 1.0

        if mode == "oracle":
            if N > N_min and E_avail < 0.5 and (t % 2 == 0):
                keep = np.ones(N, dtype=bool)
                keep[-2:] = False
                pos, vel, dwell_true, dwell_obs, stagnation, belief_src, prev_y_phys, prev_pos, I_obs_agent, I_true_agent = (
                    pos[keep], vel[keep], dwell_true[keep], dwell_obs[keep],
                    stagnation[keep], belief_src[keep], prev_y_phys[keep], prev_pos[keep],
                    I_obs_agent[keep], I_true_agent[keep]
                )
                deaths += 2
                N = len(pos)
                diff_mat = pos[:, None, :] - pos[None, :, :]
                dist_mat = np.linalg.norm(diff_mat, axis=2)
                np.fill_diagonal(dist_mat, np.inf)
                adj = (dist_mat <= R_comm).astype(float)
                nb_cnt = np.sum(adj, axis=1)
                rep = np.zeros_like(pos)

            c0, c1 = float(env.capacity.item(0)), float(env.capacity.item(1))
            d0 = np.linalg.norm(pos - env.src_pos[:1], axis=1) / (c0 + 0.15)
            d1 = np.linalg.norm(pos - env.src_pos[1:2], axis=1) / (c1 + 0.15)
            src_choice = (d1 < d0).astype(int)
            targets = env.src_pos[src_choice]
            to_target = targets - pos
            dist_t = np.linalg.norm(to_target, axis=1, keepdims=True)
            dir_t = to_target / (dist_t + 1e-6)
            if dim == 2:
                tangent = np.stack((-dir_t[:, 1], dir_t[:, 0]), axis=1)
            else:
                ref_up = np.array([[0.0, 0.0, 1.0]])
                tangent = np.cross(dir_t, ref_up)
                t_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
                tangent = np.where(t_norm > 1e-4, tangent / (t_norm + 1e-6), np.array([[1.0, 0.0, 0.0]]))
            radial_err = dist_t - 11.5
            desired_v = np.where(dist_t > 14.0, 2.0 * dir_t, 0.25 * np.tanh(radial_err / 3.0) * dir_t + 0.78 * tangent)
            from_decoy = pos - env.decoy_pos
            d_dec = np.linalg.norm(from_decoy, axis=1, keepdims=True)
            avoid = np.where(d_dec < 22.0, 5.0 * (from_decoy / (d_dec + 1e-6)), 0.0)
            acc = 1.6 * (desired_v - vel) + avoid + 0.2 * rep

        elif mode == "naive":
            vis = (y_phys > 2.6)[:, None]
            perc_grad = np.where(vis, g_obs, 0.0)
            if baseline_kl:
                dy_obs = y_phys - prev_y_phys
                dy_pred = np.sum(g_obs * (pos - prev_pos), axis=1)
                kl_innov = ((dy_obs - dy_pred)**2) / (2.0 * (sigma_m**2))
                trust_1d = np.exp(-0.25 * kl_innov)
                perc_grad = trust_1d[:, None] * perc_grad

            blind = (np.linalg.norm(perc_grad, axis=1) < 1e-4)[:, None]
            noise = np.where(blind, rng.normal(0.0, 0.95, size=(N, dim)), rng.normal(0.0, 0.25, size=(N, dim)))
            acc = 2.4 * perc_grad + rep + noise

        elif mode == "efe_full":
            if inference_enabled:
                dy_phys = y_phys - prev_y_phys
                dy_move = np.sum(g_obs * (pos - prev_pos), axis=1)
                resid_dy = dy_phys - dy_move
                active_crowd = (nb_cnt >= 2) & (y_phys > 2.6)
                mu_1 = -0.0018 * nb_cnt * y_phys
                mu_0 = np.where(dwell_obs > 3.0, 0.12, 0.0)
                var_obs = 2.0 * (sigma_m**2)
                ll_1 = -((resid_dy - mu_1)**2) / (2.0 * var_obs)
                ll_0 = -((resid_dy - mu_0)**2) / (2.0 * var_obs)
                llr = np.where(active_crowd, ll_1 - ll_0 + 0.06, 0.06)

                b_clip = np.clip(belief_src, 1e-4, 1.0 - 1e-4)
                logit_b = np.log(b_clip / (1.0 - b_clip))
                belief_src = sigmoid(0.90 * logit_b + llr)
            else:
                belief_src = np.full(N, 0.5)

            pos_buf.pop(0)
            pos_buf.append(pos.copy())
            val_buf.pop(0)
            val_buf.append((belief_src * y_phys).copy())
            valid_buf.pop(0)
            valid_buf.append(np.ones(N, dtype=bool))

            tau_mat = np.minimum(np.ceil(np.where(np.isfinite(dist_mat), dist_mat, 0.0) / c_sig).astype(int), max_delay - 1)
            relay_vec = np.zeros_like(pos)
            relay_conf = np.zeros(N)
            if use_relay:
                for i in range(N):
                    nbs = np.flatnonzero(adj[i] > 0)
                    if len(nbs) > 0:
                        d_vals = []
                        d_pos = []
                        for j in nbs:
                            slot = -1 - int(tau_mat.item(i, j))
                            if valid_buf[slot][j]:
                                d_vals.append(val_buf[slot][j])
                                d_pos.append(pos_buf[slot][j])
                        if len(d_vals) > 0:
                            d_vals_arr = np.array(d_vals)
                            d_pos_arr = np.array(d_pos)
                            best_v = float(np.max(d_vals_arr))
                            if best_v > 2.2:
                                w = (d_vals_arr**3) / (np.sum(d_vals_arr**3) + 1e-6)
                                target = np.sum(d_pos_arr * w[:, None], axis=0)
                                vec = target - pos[i]
                                relay_vec[i] = vec / (np.linalg.norm(vec) + 1e-6)
                                relay_conf[i] = np.tanh(best_v / 6.0)

            local_sense = (y_phys > 4.2).astype(float)[:, None]
            eff_grad = local_sense * belief_src[:, None] * g_obs * novelty_obs[:, None]
            need_relay = ((y_phys <= 4.2) & (nb_cnt <= 8)).astype(float)
            prag_align = (eff_grad @ action_dirs.T) + 1.65 * ((need_relay * relay_conf)[:, None] * (relay_vec @ action_dirs.T))

            nb_dir = np.zeros((N, n_dirs))
            for i in range(N):
                nbs = np.flatnonzero(dist_mat[i] < 22.0)
                if len(nbs) > 0:
                    u_nb = (pos[nbs] - pos[i]) / (dist_mat[i, nbs][:, None] + 1e-6)
                    nb_dir[i] = np.sum(np.maximum(0.0, u_nb @ action_dirs.T), axis=0) / dir_norm

            H_prior = binary_entropy(belief_src)[:, None]
            uncert_factor = np.clip(1.0 - 0.22 * nb_dir, 0.15, 1.0)
            info_gain = H_prior * uncert_factor * 3.2
            energy_cost = 0.15 * np.sum((vel[:, None, :] + action_dirs[None, :, :])**2, axis=2)

            G = -1.6 * prag_align - epist_weight * info_gain + energy_cost
            logits_pi = -softmax_beta * (G - np.min(G, axis=1, keepdims=True))
            probs_pi = np.exp(logits_pi) / np.sum(np.exp(logits_pi), axis=1, keepdims=True)
            cum_p = np.cumsum(probs_pi, axis=1)
            u_rand = rng.random((N, 1))
            chosen_idx = np.argmax(u_rand <= cum_p, axis=1)
            chosen_dir = action_dirs[chosen_idx]

            acc = 2.8 * chosen_dir + rep - 0.35 * vel

        # Synchronized per-step accounting (Issue #9 bookkeeping consistency)
        d_trap = np.linalg.norm(pos - env.decoy_pos, axis=1)
        trap_steps += int(np.sum(d_trap < r_trap))
        ag_steps += len(pos)
        tot_I += float(np.sum(I_true_agent))

        # Kinematic integration with elastic damped walls & per-step invariant assertions
        prev_pos = pos.copy()
        prev_y_phys = y_phys.copy()

        vel = 0.62 * vel + deficit_scale * acc
        spd = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = np.where(spd > v_max * deficit_scale, vel / (spd + 1e-9) * (v_max * deficit_scale), vel)
        max_v_invariant = max(max_v_invariant, float(np.max(np.linalg.norm(vel, axis=1))))

        next_pos = pos + vel
        hit_wall = (next_pos < 0.0) | (next_pos > L)
        vel = np.where(hit_wall, -0.5 * vel, vel)
        pos = np.clip(pos + vel, 0.0, L)

        if not np.all((pos >= 0.0) & (pos <= L)):
            raise RuntimeError(f"Domain invariant violated at t={t}")
        if float(np.max(np.linalg.norm(vel, axis=1))) > v_max + 1e-6:
            raise RuntimeError(f"Speed invariant violated at t={t}")

        comm_cost = 0.0 if mode == 'oracle' else (E_comm * nb_cnt)
        agent_E = E_idle + E_kin * np.sum(vel**2, axis=1) + comm_cost
        tot_E += float(np.sum(agent_E))

        # Evolutionary culling & reproduction driven strictly by observable sensor efficiency
        if mode == "efe_full" and cull_regen:
            theta_c = 2.1 if E_avail < 0.5 else 1.2
            eff_obs = I_obs_agent / (agent_E + 1e-9)
            stagnation = np.where(eff_obs < theta_c, stagnation + 1, 0)
            if N > N_min and (t % 4 == 0):
                cand = np.flatnonzero(stagnation >= 6)
                if len(cand) > 0:
                    drop = cand[:min(len(cand), N - N_min, 2)]
                    deaths += len(drop)
                    keep = np.ones(N, dtype=bool)
                    keep[drop] = False
                    pos, vel, dwell_true, dwell_obs, stagnation, belief_src, prev_y_phys, prev_pos, I_obs_agent = (
                        pos[keep], vel[keep], dwell_true[keep], dwell_obs[keep],
                        stagnation[keep], belief_src[keep], prev_y_phys[keep], prev_pos[keep], I_obs_agent[keep]
                    )
                    for b in range(max_delay):
                        pos_buf[b] = pos_buf[b][keep]
                        val_buf[b] = val_buf[b][keep]
                        valid_buf[b] = valid_buf[b][keep]
            if len(pos) < N_target and E_avail > 0.7 and (t % 4 == 0):
                fert = np.flatnonzero(I_obs_agent > 3.8)
                if len(fert) > 0:
                    n_sp = min(len(fert), N_target - len(pos), 2)
                    births += n_sp
                    par = rng.choice(fert, size=n_sp, replace=False)
                    new_p = np.clip(pos[par] + rng.normal(0.0, 2.5, size=(n_sp, dim)), 0.0, L)
                    pos = np.vstack((pos, new_p))
                    vel = np.vstack((vel, np.zeros((n_sp, dim))))
                    dwell_true = np.concatenate((dwell_true, np.zeros(n_sp)))
                    dwell_obs = np.concatenate((dwell_obs, np.zeros(n_sp)))
                    stagnation = np.concatenate((stagnation, np.zeros(n_sp)))
                    belief_src = np.concatenate((belief_src, np.full(n_sp, 0.55)))
                    prev_y_phys = np.concatenate((prev_y_phys, np.zeros(n_sp)))
                    prev_pos = np.vstack((prev_pos, new_p))
                    for b in range(max_delay):
                        pos_buf[b] = np.vstack((pos_buf[b], new_p))
                        val_buf[b] = np.concatenate((val_buf[b], np.zeros(n_sp)))
                        valid_buf[b] = np.concatenate((valid_buf[b], np.zeros(n_sp, dtype=bool)))

        env.step_dynamics(pos, t)

    return {
        "eff": tot_I / (tot_E + 1e-9),
        "i_per_ag": tot_I / max(ag_steps, 1),
        "trap_pct": 100.0 * trap_steps / max(ag_steps, 1),
        "births": births,
        "deaths": deaths,
        "max_v": max_v_invariant
    }


def summarize_suite(dim, split, label, runs, csv_rows=None):
    effs = [r["eff"] for r in runs]
    ipas = [r["i_per_ag"] for r in runs]
    trps = [r["trap_pct"] for r in runs]
    b_avg = float(np.mean([r["births"] for r in runs]))
    d_avg = float(np.mean([r["deaths"] for r in runs]))
    mean_eff = float(np.mean(effs))
    sd_eff = float(np.std(effs, ddof=1)) if len(effs) > 1 else 0.0
    se_eff = sd_eff / np.sqrt(max(len(effs), 1))
    print(
        f"[{dim}D | {split:<9}] {label:<36} | {mean_eff:5.2f} ± {sd_eff:4.2f} (SE={se_eff:.3f}) | "
        f"I/ag={np.mean(ipas):4.2f} | Trap(r<17)={np.mean(trps):5.1f}% | B/D={b_avg:4.1f}/{d_avg:4.1f}"
    )
    if csv_rows is not None:
        csv_rows.append({
            "dim": f"{dim}D",
            "seed_split": split,
            "n_seeds": len(runs),
            "configuration": label,
            "task_eff_mean": round(mean_eff, 4),
            "task_eff_sd": round(sd_eff, 4),
            "task_eff_se": round(se_eff, 4),
            "info_per_agent_step": round(float(np.mean(ipas)), 4),
            "trap_r17_pct": round(float(np.mean(trps)), 2),
            "avg_births": round(b_avg, 1),
            "avg_deaths": round(d_avg, 1)
        })
    return mean_eff


def main():
    parser = argparse.ArgumentParser(description="Swarm of Swarms v7.1.0 Unified Zero-Leakage 2D/3D Benchmark")
    parser.add_argument("--out", default="audit_benchmark_v7.1_unified.zip", help="Output ZIP archive path")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output ZIP archive")
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing audit archive '{args.out}'. Pass --overwrite or specify --out."
        )

    t0 = time.time()
    cal_seeds = list(range(42, 47))
    val_seeds = list(range(100, 120))
    all_csv = []

    configs = [
        ("naive", "1. Naive baseline (honest 1-ch)", dict(mode="naive")),
        ("base_kl", "2. Naive + 1D KL innovation filter", dict(mode="naive", baseline_kl=True)),
        ("no_relay", "3. EFE ablation: NO causal relay", dict(mode="efe_full", use_relay=False)),
        ("no_cull", "4. EFE ablation: NO cull/regen", dict(mode="efe_full", cull_regen=False)),
        ("no_epist", "5. EFE ablation: NO epistemic term", dict(mode="efe_full", epist_weight=0.0)),
        ("no_inf", "6. EFE ablation: Bayesian inf OFF", dict(mode="efe_full", inference_enabled=False)),
        ("efe_full", "7. Active Inference EFE Full (v7.1)", dict(mode="efe_full")),
        ("oracle", "8. Centralized Information Oracle", dict(mode="oracle")),
    ]

    for dim in (2, 3):
        print(f"\n=== {dim}D FULL ABLATION MATRIX (seeds=42..46, n=5) ===")
        R = {}
        for key, label, kw in configs:
            R[key] = summarize_suite(dim, "42..46", label, [run_simulation_v7(dim=dim, seed=s, **kw) for s in cal_seeds], all_csv)
        gain = 100.0 * (R["efe_full"] - R["base_kl"]) / max(R["base_kl"], 1e-6)
        gap = 100.0 * (R["efe_full"] - R["naive"]) / max(R["oracle"] - R["naive"], 1e-6)
        print(f"[{dim}D] Summary: EFE Full vs Naive+KL = {gain:+.1f}% | Oracle gap closed = {gap:.1f}%")

    for dim in (2, 3):
        print(f"\n=== {dim}D OUT-OF-SAMPLE VALIDATION (seeds=100..119, n=20) ===")
        R20 = {}
        for key, label, kw in configs:
            R20[key] = summarize_suite(dim, "100..119", label, [run_simulation_v7(dim=dim, seed=s, **kw) for s in val_seeds], all_csv)
        gain20 = 100.0 * (R20["efe_full"] - R20["base_kl"]) / max(R20["base_kl"], 1e-6)
        print(f"[{dim}D | 20-seed] Summary: EFE Full vs Naive+KL = {gain20:+.1f}%")

    env_meta = {
        "version": "v7.1.0",
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "total_runs": 400,
        "noise_floor_gating": "I_true > sigma_m (sigma_m=0.45)"
    }

    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(next(iter(all_csv)).keys()))
        writer.writeheader()
        writer.writerows(all_csv)
        zf.writestr("summary_unified_2d_3d.csv", buf.getvalue())
        zf.writestr("environment_metadata.json", json.dumps(env_meta, indent=2))

    print(f"\nCompleted full zero-leakage 2D+3D verification in {time.time() - t0:.2f}s -> {args.out}")


if __name__ == "__main__":
    main()
