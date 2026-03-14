import streamlit as st
import numpy as np
from numba import njit, prange
import plotly.express as px
import plotly.graph_objects as go
import pyvista as pv
import time
import shutil
from pathlib import Path
import io

# ----------------------------------------------------------------------
# Numba‑accelerated multiphase‑field update (N+2 order parameters)
# ----------------------------------------------------------------------
@njit(parallel=True)
def update_step_multiphase(eta, M, epsilon, gamma0, beta,
                           theta_k, theta_sub, mask_substrate,
                           dt, dx, N_total, nx, ny):
    """
    One explicit Euler step for N+2 order parameters with Ση=1 constraint.

    Parameters
    ----------
    eta : ndarray (N_total, ny, nx)
        Order parameters: [substrate, IMC_1 ... IMC_N, solder]
    M : ndarray (N_total,)
        Mobilities for each phase
    epsilon : float
        Base gradient energy coefficient
    gamma0, beta : float
        Anisotropy parameters: γ(θ) = γ0*(1 + β*cos(6*(θ_k - θ_sub)))
    theta_k : ndarray (N_total,)
        Preferred orientation for each variant (radians)
    theta_sub : ndarray (ny, nx)
        Substrate crystallographic orientation field
    mask_substrate : ndarray (ny, nx), bool
        True where substrate anisotropy applies
    dt, dx : float
        Time step and grid spacing
    N_total : int
        Total number of order parameters (N_IMC + 2)
    nx, ny : int
        Grid dimensions
    """
    # Laplacians for all fields (5‑point stencil)
    lap = np.zeros((N_total, ny, nx))
    for k in prange(N_total):
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                lap[k, i, j] = (
                    eta[k, i-1, j] + eta[k, i+1, j] +
                    eta[k, i, j-1] + eta[k, i, j+1] -
                    4.0 * eta[k, i, j]
                ) / (dx * dx)

    # Chemical potentials μₖ = δF/δηₖ
    mu = np.zeros((N_total, ny, nx))

    for k in prange(N_total):
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                eta_k = eta[k, i, j]

                # 1) Double‑well derivative: d/dη[¼η²(1-η)²] = ½η(1-η)(1-2η)
                dw_deriv = 0.5 * eta_k * (1.0 - eta_k) * (1.0 - 2.0 * eta_k)

                # 2) Pairwise gradient energy contribution
                grad_term = 0.0
                for l in range(N_total):
                    if l == k:
                        continue
                    # Use same epsilon for all pairs (could be made pair‑specific)
                    grad_term += epsilon * (lap[k, i, j] - lap[l, i, j])

                # 3) Anisotropic interfacial energy (substrate region only)
                aniso_term = 0.0
                if mask_substrate[i, j]:
                    # Apply anisotropy only to substrate–IMC interfaces
                    if k == 0 or (1 <= k <= N_total - 2):  # substrate or IMC variant
                        theta_diff = theta_k[k] - theta_sub[i, j]
                        gamma_theta = gamma0 * (1.0 + beta * np.cos(6.0 * theta_diff))
                        # Derivative of the anisotropic term w.r.t. ηₖ
                        aniso_term = gamma_theta * eta_k * (1.0 - eta_k) * (1.0 - 2.0 * eta_k)

                mu[k, i, j] = dw_deriv - grad_term + aniso_term

    # Lagrange multiplier λ enforcing Ση = 1 (weighted by mobility)
    lambda_num = np.zeros((ny, nx))
    lambda_den = np.zeros((ny, nx))
    for k in prange(N_total):
        for i in range(ny):
            for j in range(nx):
                lambda_num[i, j] += M[k] * mu[k, i, j]
                lambda_den[i, j] += M[k]
    lambda_den = np.maximum(lambda_den, 1e-12)  # avoid division by zero
    lambda_lagrange = lambda_num / lambda_den

    # Allen–Cahn update with constraint
    for k in prange(N_total):
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                deta = -M[k] * (mu[k, i, j] - lambda_lagrange[i, j]) * dt
                eta[k, i, j] += deta
                # Clip for stability
                if eta[k, i, j] < 0.0:
                    eta[k, i, j] = 0.0
                elif eta[k, i, j] > 1.0:
                    eta[k, i, j] = 1.0

    # Renormalize to strictly enforce Ση = 1
    for i in range(ny):
        for j in range(nx):
            total = 0.0
            for k in range(N_total):
                total += eta[k, i, j]
            if total > 1e-12:
                for k in range(N_total):
                    eta[k, i, j] /= total
            else:
                # Fallback: equal fractions (should never happen)
                for k in range(N_total):
                    eta[k, i, j] = 1.0 / N_total

    return eta


# ----------------------------------------------------------------------
# Initialisation for N+2 order parameters
# ----------------------------------------------------------------------
def initialize_multiphase(N_imc, nx, ny, substrate_rows,
                          seed_density=0.05, seed=42):
    """
    Create initial eta array with:
        η[0]          : substrate (bottom region)
        η[1:N_imc+1]  : IMC grain variants (small seeds near interface)
        η[N_imc+1]    : solder (top region)
    """
    np.random.seed(seed)
    N_total = N_imc + 2
    eta = np.zeros((N_total, ny, nx))

    # Substrate phase (bottom)
    eta[0, :substrate_rows, :] = 1.0

    # Solder phase (top, will be reduced by seeds)
    eta[-1, substrate_rows:, :] = 1.0

    # IMC grain seeds near the substrate/solder interface
    num_seeds = int(seed_density * nx * substrate_rows)
    for _ in range(num_seeds):
        x = np.random.randint(0, nx)
        y = np.random.randint(max(0, substrate_rows - 3),
                              min(ny, substrate_rows + 3))
        k_imc = np.random.randint(1, N_imc + 1)   # IMC variant index

        # Place a Gaussian bump (radius ~2 grid points)
        for i in range(max(0, y - 2), min(ny, y + 3)):
            for j in range(max(0, x - 2), min(nx, x + 3)):
                dist = (i - y) ** 2 + (j - x) ** 2
                if dist < 4.0:
                    bump = 0.6 * np.exp(-dist / 2.0)
                    eta[k_imc, i, j] += bump
                    # Conserve total = 1 by subtracting from existing phases
                    if i < substrate_rows:
                        eta[0, i, j] = max(0.0, eta[0, i, j] - bump)
                    else:
                        eta[-1, i, j] = max(0.0, eta[-1, i, j] - bump)

    # Final normalization to ensure Ση = 1 everywhere
    for i in range(ny):
        for j in range(nx):
            total = np.sum(eta[:, i, j])
            if total > 1e-12:
                eta[:, i, j] /= total
            else:
                eta[:, i, j] = 1.0 / N_total

    return np.clip(eta, 0.0, 1.0)


# ----------------------------------------------------------------------
# Parameter setup for the two substrate types
# ----------------------------------------------------------------------
def setup_parameters(N_imc, substrate_type, nx, ny, substrate_rows):
    """Return a dictionary of physics parameters."""
    N_total = N_imc + 2

    epsilon = 2.0      # gradient energy coefficient
    gamma0 = 1.0       # reference interfacial energy
    M_base = 1.0       # base mobility

    # Variant orientations: random within ±30°, substrate orientation fixed at 0
    theta_k = np.random.uniform(-np.pi / 6, np.pi / 6, N_total)
    theta_k[0] = 0.0

    theta_sub = np.zeros((ny, nx))
    mask_substrate = np.zeros((ny, nx), dtype=bool)
    mask_substrate[:substrate_rows, :] = True

    if substrate_type == "EP Ni":
        beta = 0.05
        # Random substrate orientation per column
        for j in range(nx):
            theta_sub[:substrate_rows, j] = np.random.uniform(-np.pi / 6, np.pi / 6)
        mobilities = np.ones(N_total) * M_base

    elif substrate_type == "nt-Ni-13Co":
        beta = 0.30
        theta_sub[:substrate_rows, :] = 0.0          # uniform substrate
        theta_k[1] = 0.0                              # favour the first IMC variant
        mobilities = np.ones(N_total) * M_base
        mobilities[1] *= 1.8                           # 80% mobility boost for variant 1
    else:
        raise ValueError(f"Unknown substrate: {substrate_type}")

    return {
        'N_total': N_total,
        'epsilon': epsilon,
        'gamma0': gamma0,
        'beta': beta,
        'theta_k': theta_k,
        'theta_sub': theta_sub,
        'mask_substrate': mask_substrate,
        'M': mobilities
    }


# ----------------------------------------------------------------------
# Free energy computation (optional diagnostic)
# ----------------------------------------------------------------------
@njit
def total_free_energy(eta, epsilon, gamma0, beta, theta_k, theta_sub,
                      mask_substrate, dx, N_total, nx, ny):
    """Compute total free energy for monitoring."""
    F = 0.0
    # double‑well
    for k in range(N_total):
        for i in range(ny):
            for j in range(nx):
                e = eta[k, i, j]
                F += 0.25 * e * e * (1.0 - e) * (1.0 - e) * dx * dx

    # pairwise gradient energy (approximate with sum over k<l)
    for k in range(N_total):
        for l in range(k + 1, N_total):
            diff = eta[k] - eta[l]
            # crude gradient via differences (not fully accurate but fast)
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    gx = (diff[i, j + 1] - diff[i, j - 1]) / (2.0 * dx)
                    gy = (diff[i + 1, j] - diff[i - 1, j]) / (2.0 * dx)
                    F += 0.5 * epsilon * (gx * gx + gy * gy) * dx * dx

    # anisotropic term in substrate region
    for k in range(N_total):
        if k == 0 or (1 <= k <= N_total - 2):   # substrate or IMC
            for i in range(ny):
                for j in range(nx):
                    if mask_substrate[i, j]:
                        e = eta[k, i, j]
                        theta_diff = theta_k[k] - theta_sub[i, j]
                        gamma_theta = gamma0 * (1.0 + beta * np.cos(6.0 * theta_diff))
                        F += gamma_theta * e * e * (1.0 - e) * (1.0 - e) * dx * dx
    return F


# ----------------------------------------------------------------------
# Post‑processing and output helpers
# ----------------------------------------------------------------------
def dominant_phase(eta):
    """Return array of dominating phase index (0 … N_total-1)."""
    return np.argmax(eta, axis=0)

def imc_density(eta, N_imc):
    """Sum of IMC order parameters (indices 1..N_imc)."""
    return np.sum(eta[1:N_imc+1], axis=0)

def save_vts(eta, step, t, output_dir, dx_um, nx, ny, N_total):
    """Save all fields as a VTS file for Paraview."""
    x = np.arange(nx) * dx_um
    y = np.arange(ny) * dx_um
    z = np.array([0.0])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (nx, ny, 1)

    for k in range(N_total):
        grid.point_data[f"eta_{k}"] = eta[k].flatten(order='F')

    filename = output_dir / f"imc_t{t:.3f}_step{step:06d}.vts"
    grid.save(filename)
    return filename


# ----------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Multiphase‑Field Model for IMC Grain Growth (N+2 Order Parameters)")
st.markdown(r"""
This app implements the **Steinbach multiphase‑field formalism** with $N+2$ order parameters:
- $\eta_0$: substrate (EP Ni or (111) nt‑Ni‑13Co)
- $\eta_1 \dots \eta_N$: $N$ IMC grain variants
- $\eta_{N+1}$: top solder phase

The constraint $\sum_{k=0}^{N+1} \eta_k = 1$ is enforced via a Lagrange multiplier + renormalisation.
""")

# ----------------------------------------------------------------------
# Sidebar – parameters
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Domain & Grid")
    nx = st.slider("Grid points (x)", 100, 500, 300, step=10)
    ny = st.slider("Grid points (y)", 50, 200, 100, step=10)
    dx_um = st.number_input("Grid spacing (µm)", 0.1, 5.0, 1.0, step=0.1)
    substrate_rows = st.slider("Substrate thickness (grid rows)", 3, 20, 5)

    st.header("Physical Parameters")
    substrate = st.selectbox("Substrate type", ["EP Ni", "nt-Ni-13Co"])
    N_imc = st.slider("Number of IMC variants", 1, 20, 5)
    total_steps = st.slider("Number of time steps", 1000, 50000, 5000, step=500)
    dt = st.number_input("Time step dt (µs)", 1e-4, 1e-1, 0.001, format="%.4f")
    epsilon = st.number_input("Gradient coefficient ε", 0.1, 10.0, 2.0)
    gamma0 = st.number_input("Reference interfacial energy γ₀", 0.1, 10.0, 1.0)
    M_base = st.number_input("Base mobility M₀", 0.1, 10.0, 1.0)

    st.header("Output")
    output_interval = st.slider("Output every N steps", 10, 1000, 100, step=10)
    colormap = st.selectbox("Colormap", px.colors.named_colorscales(), index=20)  # turbo

    st.header("Checkpoints")
    checkpoint_interval = st.slider("Checkpoint every N steps", 100, 2000, 500, step=100)

    # Temporary output directory
    output_dir = Path("imc_outputs")
    output_dir.mkdir(exist_ok=True)

    if st.button("Clear output files"):
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(exist_ok=True)
        st.success("Output directory cleared.")

# ----------------------------------------------------------------------
# Initialisation and session state management
# ----------------------------------------------------------------------
if "seed" not in st.session_state:
    st.session_state.seed = 42

if "sim_state" not in st.session_state:
    st.session_state.sim_state = {
        "eta": None,
        "params": None,
        "step": 0,
        "running": False,
        "outputs": [],          # list of (step, t, vts_path, free_energy)
        "free_energies": []     # list of (t, F)
    }

# ----------------------------------------------------------------------
# Geometry preview (always shown)
# ----------------------------------------------------------------------
st.subheader("Initial Geometry Preview")
# Generate preview using current parameters
preview_eta = initialize_multiphase(N_imc, nx, ny, substrate_rows, seed=st.session_state.seed)
phase_preview = dominant_phase(preview_eta)
density_preview = imc_density(preview_eta, N_imc)

fig_preview = px.imshow(
    phase_preview,
    color_continuous_scale=colormap,
    title="Dominant phase (preview)",
    labels=dict(x="x (grid points)", y="y (grid points)", color="phase"),
    origin="lower", aspect="auto"
)
st.plotly_chart(fig_preview, use_container_width=True)

# Button to refresh random seeds
col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Generate new random seeds"):
        st.session_state.seed += 1
        st.rerun()
with col2:
    if st.button("▶ Run Simulation") and not st.session_state.sim_state["running"]:
        # Initialise simulation state
        params = setup_parameters(N_imc, substrate, nx, ny, substrate_rows)
        # Override epsilon and gamma0 with user values
        params['epsilon'] = epsilon
        params['gamma0'] = gamma0
        params['M'] = params['M'] * (M_base / 1.0)  # scale base mobility

        eta_init = initialize_multiphase(N_imc, nx, ny, substrate_rows,
                                         seed=st.session_state.seed)

        st.session_state.sim_state = {
            "eta": eta_init,
            "params": params,
            "step": 0,
            "running": True,
            "outputs": [],
            "free_energies": []
        }
        st.rerun()

# ----------------------------------------------------------------------
# Simulation loop (runs while sim_state["running"] is True)
# ----------------------------------------------------------------------
sim = st.session_state.sim_state
if sim["running"] and sim["eta"] is not None:
    # Extract state
    eta = sim["eta"]
    params = sim["params"]
    current_step = sim["step"]
    N_total = params['N_total']

    # Determine batch size (how many steps to run before refreshing UI)
    batch_size = 50
    stop_requested = False

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Run a batch
    for _ in range(batch_size):
        if current_step >= total_steps:
            sim["running"] = False
            st.success("Simulation completed.")
            break

        if not sim["running"]:
            stop_requested = True
            break

        # Perform one update
        eta = update_step_multiphase(
            eta, params['M'], params['epsilon'], params['gamma0'], params['beta'],
            params['theta_k'], params['theta_sub'], params['mask_substrate'],
            dt, dx_um, N_total, nx, ny
        )
        current_step += 1

        # Compute free energy at output steps
        if current_step % output_interval == 0 or current_step == total_steps:
            F = total_free_energy(eta, params['epsilon'], params['gamma0'], params['beta'],
                                  params['theta_k'], params['theta_sub'],
                                  params['mask_substrate'], dx_um, N_total, nx, ny)
            sim["free_energies"].append((current_step * dt, F))

        # Save checkpoint
        if current_step % checkpoint_interval == 0:
            ckpt_file = output_dir / f"checkpoint_{current_step:06d}.npz"
            np.savez(ckpt_file, eta=eta, step=current_step)

        # Save VTS and store metadata at output steps
        if current_step % output_interval == 0 or current_step == total_steps:
            t = current_step * dt
            vts_path = save_vts(eta, current_step, t, output_dir, dx_um, nx, ny, N_total)
            sim["outputs"].append((current_step, t, vts_path))

        # Update progress
        progress_bar.progress(min(current_step / total_steps, 1.0))
        status_text.text(f"Step {current_step}/{total_steps} – time {current_step*dt:.3f} µs")

    # Update state
    sim["eta"] = eta
    sim["step"] = current_step
    sim["running"] = sim["running"] and (current_step < total_steps) and not stop_requested

    # Rerun to refresh UI (next batch)
    st.rerun()

# ----------------------------------------------------------------------
# Display results
# ----------------------------------------------------------------------
if sim["eta"] is not None:
    st.header("Current State")
    eta = sim["eta"]
    current_step = sim["step"]
    t = current_step * dt

    phase = dominant_phase(eta)
    density = imc_density(eta, N_imc)

    col1, col2 = st.columns(2)
    with col1:
        fig_phase = px.imshow(
            phase,
            color_continuous_scale=colormap,
            title=f"Dominant phase at step {current_step} (t={t:.3f} µs)",
            labels=dict(x="x (grid points)", y="y (grid points)", color="phase"),
            origin="lower", aspect="auto"
        )
        st.plotly_chart(fig_phase, use_container_width=True)

    with col2:
        fig_dens = px.imshow(
            density,
            color_continuous_scale="hot",
            title=f"IMC density (Σ IMC η) at step {current_step}",
            labels=dict(x="x (grid points)", y="y (grid points)", color="Ση"),
            origin="lower", aspect="auto",
            zmin=0, zmax=1
        )
        st.plotly_chart(fig_dens, use_container_width=True)

    # Free energy plot
    if sim["free_energies"]:
        st.subheader("Total Free Energy Evolution")
        times, Fs = zip(*sim["free_energies"])
        fig_F = px.line(x=times, y=Fs, labels=dict(x="Time (µs)", y="Free energy"))
        st.plotly_chart(fig_F, use_container_width=True)

    # Download buttons for outputs
    st.subheader("Output Files")
    for step_out, t_out, vts_path in sim["outputs"]:
        with st.expander(f"Step {step_out} (t={t_out:.3f} µs)"):
            if vts_path.exists():
                with open(vts_path, "rb") as f:
                    st.download_button(
                        label="Download VTS (Paraview)",
                        data=f,
                        file_name=vts_path.name,
                        mime="application/octet-stream"
                    )
            # Optionally, also provide PNG of phase map
            fig_temp = px.imshow(
                dominant_phase(eta),  # Not the snapshot, but we don't have it stored
                color_continuous_scale=colormap,
                title=f"Phase map at t={t_out:.3f} µs"
            )
            img_bytes = fig_temp.to_image(format="png")
            st.download_button(
                label="Download phase map PNG",
                data=img_bytes,
                file_name=f"phase_t{t_out:.3f}.png",
                mime="image/png"
            )

# Stop button (always visible)
if sim["running"]:
    if st.button("⏹ Stop Simulation"):
        sim["running"] = False
        st.rerun()

# ----------------------------------------------------------------------
# Instructions
# ----------------------------------------------------------------------
st.markdown("""
---
## Instructions
1. Adjust domain size, grid spacing, and physical parameters in the sidebar.
2. Choose substrate type (EP Ni or nt-Ni-13Co). The anisotropy strength and mobility boost are set automatically.
3. Click **Generate new random seeds** to change the initial distribution of IMC nuclei.
4. Click **Run Simulation** to start. The simulation runs in batches, updating the plots every 50 steps.
5. Use the **Stop Simulation** button to pause.
6. After completion (or during a pause), explore the output files in the expandable sections.
7. Checkpoints are saved every N steps in the `imc_outputs` folder. You can download them later.

### File formats
- **VTS**: Structured grid for Paraview (contains all order parameters).
- **PNG**: Phase map image (generated on‑the‑fly for download).
- **NPZ**: NumPy checkpoint (contains `eta` array and step number).

### Physical interpretation
- **EP Ni** (weak anisotropy): all IMC variants grow with similar probability → random phase map.
- **nt-Ni-13Co** (strong anisotropy + mobility boost for variant 1): favoured variant quickly dominates → uniform phase map.

The constraint $\sum \eta_k = 1$ is enforced, so the solder and substrate phases naturally appear where IMC grains are absent.
""")
