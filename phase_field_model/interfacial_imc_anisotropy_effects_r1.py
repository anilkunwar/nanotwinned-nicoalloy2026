import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

# ----------------------------------------------------------------------
# Numba‑accelerated simulation core
# ----------------------------------------------------------------------
@njit
def simulate(eta, M, theta_k, theta_sub, anisotropy_strength, gamma0, epsilon,
             dt, mask, N, nx, ny, dx, nsteps):
    """
    eta      : (N, ny, nx) current order parameters
    M        : (N,) mobility per variant
    theta_k  : (N,) orientation of each variant (radians)
    theta_sub: (ny, nx) substrate orientation (only used where mask is True)
    anisotropy_strength : scalar (0.05 for EP Ni, 0.3 for nt-Ni-13Co)
    gamma0   : reference interfacial energy
    epsilon  : gradient energy coefficient
    dt       : time step
    mask     : (ny, nx) boolean, True in substrate region
    N, nx, ny, dx : grid parameters
    nsteps   : number of time steps
    """
    for _ in range(nsteps):
        # ---- compute Laplacians for all fields ----
        lap = np.zeros((N, ny, nx))
        for k in range(N):
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    lap[k, i, j] = (
                        eta[k, i-1, j] + eta[k, i+1, j] +
                        eta[k, i, j-1] + eta[k, i, j+1] -
                        4.0 * eta[k, i, j]
                    ) / (dx * dx)

        # ---- sum of Laplacians ----
        sum_lap = np.zeros((ny, nx))
        for k in range(N):
            for i in range(ny):
                for j in range(nx):
                    sum_lap[i, j] += lap[k, i, j]

        # ---- update each order parameter ----
        for k in range(N):
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    eta_ij = eta[k, i, j]

                    # double‑well derivative
                    dw = 0.5 * eta_ij * (1.0 - eta_ij) * (1.0 - 2.0 * eta_ij)

                    # gradient contribution
                    grad = -epsilon * (N * lap[k, i, j] - sum_lap[i, j])

                    chem = dw + grad

                    # anisotropic term only in substrate region
                    if mask[i, j]:
                        B = gamma0 * (1.0 + anisotropy_strength *
                                      np.cos(6.0 * (theta_k[k] - theta_sub[i, j])))
                        aniso = 2.0 * B * eta_ij * (1.0 - eta_ij) * (1.0 - 2.0 * eta_ij)
                        chem += aniso

                    # Allen–Cahn
                    deta = -M[k] * chem * dt
                    eta[k, i, j] += deta

                    # keep within [0,1]
                    if eta[k, i, j] < 0.0:
                        eta[k, i, j] = 0.0
                    elif eta[k, i, j] > 1.0:
                        eta[k, i, j] = 1.0

    return eta


# ----------------------------------------------------------------------
# Initialisation of order parameters (small seeds near substrate)
# ----------------------------------------------------------------------
def initialize_eta(N, nx, ny, substrate_thickness, seed_density=0.1):
    eta = np.zeros((N, ny, nx))
    np.random.seed(42)  # reproducibility
    num_seeds = int(seed_density * nx * substrate_thickness)
    for _ in range(num_seeds):
        x = np.random.randint(0, nx)
        y = np.random.randint(0, substrate_thickness)
        k = np.random.randint(0, N)
        # place a small Gaussian bump
        for i in range(max(0, y - 2), min(ny, y + 3)):
            for j in range(max(0, x - 2), min(nx, x + 3)):
                dist = (i - y) ** 2 + (j - x) ** 2
                if dist < 4.0:
                    eta[k, i, j] += 0.5 * np.exp(-dist / 2.0)
    # clip to 1
    eta = np.clip(eta, 0.0, 1.0)
    return eta


# ----------------------------------------------------------------------
# Streamlit user interface
# ----------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Anisotropic Multi‑Phase‑Field Model for IMC Grain Growth")
st.markdown("""
This app simulates the growth of \(N\) interfacial IMC grains on two different substrates:
**EP Ni** (random orientation, weak anisotropy) and **(111) nt-Ni-13Co** (strong six‑fold anisotropy
with a favoured {010} variant). The model uses only order parameters (no concentration fields)
and incorporates the effect of Co via substrate‑dependent anisotropy and mobility.
""")

with st.sidebar:
    st.header("Simulation Parameters")
    substrate = st.selectbox("Substrate type", ["EP Ni", "nt-Ni-13Co"])
    N = st.slider("Number of variants", 1, 20, 5)
    nx = st.slider("Grid size (x)", 50, 200, 100)
    ny = st.slider("Grid size (y)", 50, 200, 100)
    nsteps = st.slider("Number of time steps", 100, 5000, 1000)

    st.subheader("Physical constants")
    dx = 1.0                      # grid spacing (arbitrary units)
    dt = st.slider("Time step dt", 0.001, 0.1, 0.01, format="%.3f")
    epsilon = st.slider("Gradient coefficient ε", 0.1, 5.0, 1.0)
    gamma0 = st.slider("Reference interfacial energy γ₀", 0.1, 5.0, 1.0)
    M0 = st.slider("Base mobility M₀", 0.1, 5.0, 1.0)

    # Substrate‑dependent parameters (from the theory)
    if substrate == "EP Ni":
        anisotropy_strength = 0.05
        mobility_factor = 1.0          # all variants equal
        st.info("EP Ni: weak anisotropy, all variants equal mobility")
    else:
        anisotropy_strength = 0.3
        mobility_factor = 1.8           # favoured variant gets 80% boost
        st.info("nt-Ni-13Co: strong anisotropy, favoured {010} variant (index 0) has higher mobility")

    st.write(f"**Anisotropy strength**: {anisotropy_strength}")
    st.write(f"**Mobility factor for favoured variant**: {mobility_factor}")

    run = st.button("Run Simulation")

# ----------------------------------------------------------------------
# Main simulation block
# ----------------------------------------------------------------------
if run:
    start_time = time.time()

    # ---- 1. Variant orientations ----
    # Random orientations within ±30° (six‑fold symmetry)
    theta_k = np.random.uniform(-np.pi / 6, np.pi / 6, N)
    if substrate == "nt-Ni-13Co":
        theta_k[0] = 0.0   # favoured variant

    # ---- 2. Substrate orientation field ----
    substrate_thickness = 5   # rows from bottom
    theta_sub = np.zeros((ny, nx))
    mask = np.zeros((ny, nx), dtype=bool)
    if substrate == "EP Ni":
        # random orientation per column in the substrate region
        for j in range(nx):
            theta_sub[:substrate_thickness, j] = np.random.uniform(-np.pi / 6, np.pi / 6)
    else:
        theta_sub[:substrate_thickness, :] = 0.0
    mask[:substrate_thickness, :] = True

    # ---- 3. Mobility array ----
    M = np.ones(N) * M0
    if substrate == "nt-Ni-13Co":
        M[0] *= mobility_factor

    # ---- 4. Initialise order parameters ----
    eta = initialize_eta(N, nx, ny, substrate_thickness)

    # ---- 5. Run the simulation ----
    with st.spinner("Simulating grain evolution ..."):
        eta_final = simulate(eta, M, theta_k, theta_sub, anisotropy_strength,
                             gamma0, epsilon, dt, mask, N, nx, ny, dx, nsteps)

    elapsed = time.time() - start_time
    st.success(f"Simulation finished in {elapsed:.2f} seconds")

    # ---- 6. Visualisation ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Orientation map: which variant dominates at each pixel
    orientation = np.argmax(eta_final, axis=0)
    max_eta = np.max(eta_final, axis=0)
    orientation[max_eta < 0.1] = -1   # mark empty regions

    im0 = axes[0].imshow(orientation, cmap='tab10', origin='lower', aspect='auto',
                         interpolation='nearest')
    axes[0].set_title("Grain orientation (variant index)")
    plt.colorbar(im0, ax=axes[0], ticks=range(N), label='variant')

    # Grain density (sum of all order parameters)
    sum_eta = np.sum(eta_final, axis=0)
    im1 = axes[1].imshow(sum_eta, cmap='hot', origin='lower', aspect='auto',
                         interpolation='bilinear')
    axes[1].set_title("Grain density")
    plt.colorbar(im1, ax=axes[1], label='Σ η_k')

    for ax in axes:
        ax.set_xlabel('x (grid points)')
        ax.set_ylabel('y (grid points)')

    st.pyplot(fig)

    st.markdown("""
    **Interpretation**  
    - The orientation map shows which crystallographic variant (0 … N-1) is present.
    - For **EP Ni** all variants grow with similar probability → random colours near the interface.
    - For **nt-Ni-13Co** the favoured variant (index 0, orientation 0°) quickly outcompetes others → large blue regions.
    - The density map highlights where IMC grains have formed (values close to 1 inside grains).
    """)
