#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
NB6 = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=int)

def make_generator_mask(k_keep: int):
    # 3x3x3에서 중심에서 먼 순으로 k개 True(고체) 유지 → pore 네트워크 확보
    coords=[]
    for z in range(3):
        for y in range(3):
            for x in range(3):
                m = abs(x-1)+abs(y-1)+abs(z-1)
                coords.append(((z,y,x), m))
    coords.sort(key=lambda t: t[1], reverse=True)
    gen = np.zeros((3,3,3), dtype=bool)
    for i in range(k_keep):
        (z,y,x),_ = coords[i]; gen[z,y,x]=True
    return gen  # True=solid

def build_menger(level=3, k_keep=20):
    solid = np.ones((1,1,1), dtype=bool)
    gen = make_generator_mask(k_keep)
    for _ in range(level):
        solid = np.kron(solid, gen)
    return solid  # True=solid, False=pore

def build_uniform(level=3):
    n = 3**level
    solid = np.ones((n,n,n), dtype=bool)
    # 십자형 대공 채널을 뚫어서 균일 pore망 생성(간단화)
    w = max(2, n//12)  # 채널 폭
    solid[:, :, n//2-w:n//2+w] = False
    solid[:, n//2-w:n//2+w, :] = False
    solid[n//2-w:n//2+w, :, :] = False
    return solid

def masks_from_solid(solid: np.ndarray):
    pore = ~solid
    Z,Y,X = solid.shape
    pad = np.pad(solid, 1, constant_values=False)
    surf = np.zeros_like(solid, dtype=bool)
    for dz,dy,dx in NB6:
        neigh = pad[1+dz:1+dz+Z, 1+dy:1+dy+Y, 1+dx:1+dx+X]
        surf |= (solid & (~neigh))  # 고체이면서 이웃에 공극 존재
    bpore = np.zeros_like(pore, dtype=bool)
    bpore[0,:,:]|=pore[0,:,:];  bpore[-1,:,:]|=pore[-1,:,:]
    bpore[:,0,:]|=pore[:,0,:];  bpore[:,-1,:]|=pore[:,-1,:]
    bpore[:,:,0]|=pore[:,:,0];  bpore[:,:,-1]|=pore[:,:,-1]
    return pore, surf, bpore




# In[2]:


def init_fields(solid: np.ndarray, n_atm=50):
    pore, surf, bpore = masks_from_solid(solid)
    A = np.zeros_like(solid, dtype=np.int32)  # reactant
    P = np.zeros_like(solid, dtype=np.int32)  # product
    A[bpore] = n_atm  # 외부 대기층
    return pore, surf, bpore, A, P

def mcs_step_numba(solid, pore, bpore, A, P, Ndif=1, Pr=0.1):
    Z,Y,X = solid.shape
    intrinsic = 0
    apparent  = 0
    idx = np.array(np.where(pore)).T
    rng.shuffle(idx)

    for z,y,x in idx:
        occ = A[z,y,x] + P[z,y,x]
        if occ <= 0: 
            continue
        is_A = (rng.integers(occ) < A[z,y,x])
        dz,dy,dx = NB6[rng.integers(6)]

        # 최대 Ndif 직진 (pore만 통과, 고체 만나면 정지)
        zn,yn,xn = z,y,x
        steps=0
        while steps < Ndif:
            z2,y2,x2 = zn+dz, yn+dy, xn+dx
            if not (0<=z2<Z and 0<=y2<Y and 0<=x2<X): break
            if solid[z2,y2,x2]: break
            zn,yn,xn = z2,y2,x2
            steps += 1

        # 이동 수용확률 P_d (목적지 총점유가 낮을수록 유리)
        n_src = A[z,y,x] + P[z,y,x]
        n_dst = A[zn,yn,xn] + P[zn,yn,xn]
        Pd = np.exp(-(n_dst - n_src))
        Pd = 1.0 if Pd>1 else Pd
        if rng.random() < Pd:
            if is_A: A[z,y,x]-=1; A[zn,yn,xn]+=1
            else:    P[z,y,x]-=1; P[zn,yn,xn]+=1
            z,y,x = zn,yn,xn

        # surface 인접 시 reaction(A만)
        if is_A and A[z,y,x]>0:
            touched = False
            for dz2,dy2,dx2 in NB6:
                zz,yy,xx = z+dz2, y+dy2, x+dx2
                if 0<=zz<Z and 0<=yy<Y and 0<=xx<X and solid[zz,yy,xx]:
                    touched=True; break
            if touched and rng.random()<Pr:
                A[z,y,x]-=1; P[z,y,x]+=1
                intrinsic += 1

        # boundary pore의 P는 외부로 방출(측정=apparent rate)
        if bpore[z,y,x] and P[z,y,x]>0:
            apparent += P[z,y,x]
            P[z,y,x] = 0

    return intrinsic, apparent


# In[3]:


def run_sim_numba(solid, Ndif=1, Pr=0.1, mcs_total=1200, mcs_burn=600, n_atm=50):
    pore, surf, bpore, A, P = init_fields(solid, n_atm=n_atm)
    intr_hist=[]; app_hist=[]
    for _ in range(mcs_total):
        intr, app = mcs_step_numba(solid, pore, bpore, A, P, Ndif=Ndif, Pr=Pr)
        intr_hist.append(intr); app_hist.append(app)
    intr_ss = np.mean(intr_hist[mcs_burn:])
    app_ss  = np.mean(app_hist[mcs_burn:])
    return intr_ss, app_ss, np.array(intr_hist), np.array(app_hist)

def sweep_Pr_curves(struct="fractal", level=3, k_keep=20, Ndif=1, Pr_list=None):
    if Pr_list is None:
        Pr_list = np.geomspace(5e-3, 1.0, 8)
    rates=[]
    for Pr in Pr_list:
        solid = build_menger(level, k_keep) if struct=="fractal" else build_uniform(level)
        intr, app, *_ = run_sim_numba(solid, Ndif=Ndif, Pr=Pr, mcs_total=1200, mcs_burn=600, n_atm=50)
        rates.append(app)
    return np.array(Pr_list), np.array(rates)

def sweep_D_opt(level=3, k_values=(12,14,16,18,20,22,24), Ndif=10, Pr=0.1):
    Ds=[]; rates=[]
    for k in k_values:
        solid = build_menger(level, k)
        intr, app, *_ = run_sim_numba(solid, Ndif=Ndif, Pr=Pr, mcs_total=1000, mcs_burn=500, n_atm=50)
        Ds.append(np.log(k)/np.log(3)); rates.append(app)
    return np.array(Ds), np.array(rates)


# In[4]:


# ==========================================================
# STEP 4: Optimal fractal dimension search
# - Sweep different k_keep values (mapping to Df)
# - Identify the Df that maximizes apparent rate
# ==========================================================
# === Step 4: Monte Carlo 이동/reaction 커널 (Guo & Keil 2003 규칙) ===

# 전 셀에서 이미 있다면 재정의 안 해도 됨
rng = np.random.default_rng(0)
NB6 = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=int)



# --- (선택) 빠른 동작 test ---
# 만약 solid가 아직 없다면, 아래 example 한 줄을 잠깐 풀어서 structure를 만든 뒤 test 가능:
# def build_uniform(level=3):
#     n=3**level; solid=np.ones((n,n,n),bool); w=max(2,n//12)
#     solid[:,:,n//2-w:n//2+w]=False; solid[:,n//2-w:n//2+w,:]=False; solid[n//2-w:n//2+w,:,:]=False; return solid
# solid = build_uniform(level=3)
# intr_ss, app_ss, intr_hist, app_hist = run_sim_numba(solid, Ndif=10, Pr=0.1, mcs_total=400, mcs_burn=200, n_atm=50)
# print("steady intrinsic:", intr_ss, "steady apparent:", app_ss)


# In[5]:


# ==========================================================
# STEP 2a: Mean Square Displacement (MSD) analysis
# - Compare diffusion in uniform vs fractal structures
# - MSD is useful to characterize transport regimes
# ==========================================================
# === Step 2a: MSD (Mean Squared Displacement) for ghost walkers ===
# ==========================================================
# STEP 4: Optimal fractal dimension search
# - Sweep different k_keep values (mapping to Df)
# - Identify the Df that maximizes apparent rate
# ==========================================================
# 위치: Step 4 (numba 커널) 바로 아래에 붙이세요.

from numba import njit


# NB6가 이미 위에서 정의돼 있다면 재정의 불필요합니다.
# NB6 = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.int32)

@njit
def _msd_walkers_numba(pore, coords, steps=400, n_walkers=300, Ndif=1, seed=0):
    """
    pore: bool[Z,Y,X] (True=공극)
    coords: int32[Npore,3] (공극 좌표 목록)
    steps: 시간 스텝 수
    n_walkers: 고스트 워커 수 (농도/반응에 영향 X)
    Ndif: 한 번에 직진 이동 가능한 최대 칸 수 (Guo의 Ndif 규칙)
    seed: 난수 시드
    return: msd[steps] (각 t에서의 평균 r^2)
    """
    np.random.seed(seed)
    Z, Y, X = pore.shape
    ncoords = coords.shape[0]

    # 시작/현재 위치
    starts = np.empty((n_walkers, 3), np.int32)
    pos    = np.empty((n_walkers, 3), np.int32)
    for i in range(n_walkers):
        j = np.random.randint(ncoords)
        starts[i,0] = coords[j,0]; starts[i,1] = coords[j,1]; starts[i,2] = coords[j,2]
        pos[i,0]    = coords[j,0]; pos[i,1]    = coords[j,1]; pos[i,2]    = coords[j,2]

    msd = np.zeros(steps, dtype=np.float64)

    for t in range(steps):
        sum_r2 = 0.0
        for i in range(n_walkers):
            dz,dy,dx = NB6[np.random.randint(6)]
            z = pos[i,0]; y = pos[i,1]; x = pos[i,2]
            zn,yn,xn = z,y,x

            # 최대 Ndif 직진 (solid이면 중단)
            k = 0
            while k < Ndif:
                z2 = zn + dz; y2 = yn + dy; x2 = xn + dx
                if not (0 <= z2 < Z and 0 <= y2 < Y and 0 <= x2 < X):
                    break
                if not pore[z2, y2, x2]:  # solid이면 이동 불가
                    break
                zn,yn,xn = z2,y2,x2
                k += 1

            # 위치 업데이트
            pos[i,0] = zn; pos[i,1] = yn; pos[i,2] = xn

            # r^2 누적
            dz0 = pos[i,0] - starts[i,0]
            dy0 = pos[i,1] - starts[i,1]
            dx0 = pos[i,2] - starts[i,2]
            sum_r2 += dz0*dz0 + dy0*dy0 + dx0*dx0

        msd[t] = sum_r2 / n_walkers

    return msd

def compute_msd(solid, steps=400, n_walkers=300, Ndif=1, seed=0):
    """
    외부 래퍼: solid(=True면 고체)에서 공극을 추출해 MSD 계산
    """
    pore, _, _ = masks_from_solid(solid)  # 기존 함수 재사용
    coords = np.argwhere(pore).astype(np.int32)
    msd = _msd_walkers_numba(pore, coords, steps=steps, n_walkers=n_walkers, Ndif=Ndif, seed=seed)
    t = np.arange(1, steps+1, dtype=np.float64)
    return t, msd

def plot_msd_compare(level=3, k_keep=18, steps=400, n_walkers=300, Ndif=10, seed=0):
    """
    동일 조건에서 Uniform vs Fractal MSD 비교 (log–log)
    """
    solid_fractal = build_menger(level=level, k_keep=k_keep)
    solid_uniform = build_uniform(level=level)

    t, msd_u = compute_msd(solid_uniform, steps=steps, n_walkers=n_walkers, Ndif=Ndif, seed=seed)
    _, msd_f = compute_msd(solid_fractal, steps=steps, n_walkers=n_walkers, Ndif=Ndif, seed=seed+1)

    plt.figure(figsize=(6,4))
    plt.loglog(t, msd_u, label="Uniform")
    plt.loglog(t, msd_f, label="Fractal")
    plt.xlabel("t (MC steps)")
    plt.ylabel("MSD = ⟨r²(t)⟩")
    plt.title(f"MSD vs t (Ndif={Ndif}, level={level}, k_keep={k_keep})")
    plt.legend(); plt.grid(True, which="both", ls=":")
    plt.show()

    # 기울기(알파) 대략 추정: 후반부 선형구간에서
    lo = steps//3; hi = steps-1
    s_u = np.polyfit(np.log(t[lo:hi]), np.log(msd_u[lo:hi]), 1)[0]
    s_f = np.polyfit(np.log(t[lo:hi]), np.log(msd_f[lo:hi]), 1)[0]
    print(f"slope α (Uniform) ≈ {s_u:.2f},   slope α (Fractal) ≈ {s_f:.2f}")


# 1) 기본 MSD 비교 (Knudsen 레짐 권장: Ndif=10)
plot_msd_compare(level=3, k_keep=18, steps=400, n_walkers=300, Ndif=10)

# 2) Df 바꾸면서 체감 (예: 보다 거친 fractal)
plot_msd_compare(level=3, k_keep=22, steps=400, n_walkers=300, Ndif=10)



# In[ ]:


# ==========================================================
# STEP 2b: Residence Time Distribution (RTD) analysis
# - Probe how long walkers spend inside pores before exiting
# - Reveals transport efficiency differences
# ==========================================================
# === Step 2b: Residence Time Distribution (RTD) ===
import numpy as np
import matplotlib.pyplot as plt

def simulate_RTD(solid, Ndif=10, Pr=0.1, n_particles=500, max_steps=2000, seed=0):
    """
    여러 입자를 경계에서 주입 → 반응 후 생성물 P로 바뀌어 탈출할 때까지의
    '체류 시간'을 기록.
    """
    rng = np.random.default_rng(seed)
    pore, surf, bpore, A, P = init_fields(solid, n_atm=0)  # 대기층은 직접 주입
    times = []

    for _ in range(n_particles):
        # initial 위치: boundary pore 하나 무작위
        bz, by, bx = np.array(np.argwhere(bpore))[rng.integers(bpore.sum())]
        z, y, x = bz, by, bx
        t = 0
        alive = True

        while alive and t < max_steps:
            t += 1
            # 무작위 이동 방향
            dz, dy, dx = NB6[rng.integers(6)]
            z2, y2, x2 = z+dz, y+dy, x+dx

            if not (0 <= z2 < solid.shape[0] and 0 <= y2 < solid.shape[1] and 0 <= x2 < solid.shape[2]):
                # boundary 바깥으로 탈출 → residence time 기록
                times.append(t)
                alive = False
                break

            if solid[z2,y2,x2]:
                # 벽 충돌이면 그대로
                continue

            # 이동
            z, y, x = z2, y2, x2

            # surface 접촉 시 reaction(확률 Pr)
            touched = False
            for dz2, dy2, dx2 in NB6:
                zz, yy, xx = z+dz2, y+dy2, x+dx2
                if 0 <= zz < solid.shape[0] and 0 <= yy < solid.shape[1] and 0 <= xx < solid.shape[2]:
                    if solid[zz,yy,xx]:
                        touched = True
                        break
            if touched and rng.random() < Pr:
                # reaction물 → 생성물로 바뀌었다고 가정 → 즉시 탈출
                times.append(t)
                alive = False
                break

    return np.array(times)

def plot_RTD_compare(level=3, k_keep=18, Ndif=10, Pr=0.1,
                     n_particles=500, max_steps=2000):
    """
    Uniform vs Fractal 구조의 RTD 비교
    """
    solid_uni = build_uniform(level)
    solid_frac = build_menger(level, k_keep=k_keep)

    t_uni = simulate_RTD(solid_uni, Ndif=Ndif, Pr=Pr,
                         n_particles=n_particles, max_steps=max_steps)
    t_frac = simulate_RTD(solid_frac, Ndif=Ndif, Pr=Pr,
                          n_particles=n_particles, max_steps=max_steps)

    plt.figure(figsize=(6,4))
    bins = np.linspace(0, max_steps, 40)
    plt.hist(t_uni, bins=bins, density=True, alpha=0.5, label="Uniform")
    plt.hist(t_frac, bins=bins, density=True, alpha=0.5, label=f"Fractal (k_keep={k_keep})")
    plt.xlabel("Residence time (steps)")
    plt.ylabel("Probability density")
    plt.title("RTD comparison")
    plt.legend()
    plt.show()

    print(f"평균 체류시간 — Uniform: {np.mean(t_uni):.1f}, Fractal: {np.mean(t_frac):.1f}")


# ==========================================================
# STEP 2b: Residence Time Distribution (RTD) analysis
# - Probe how long walkers spend inside pores before exiting
# - Reveals transport efficiency differences
# ==========================================================
# Step 2b test
plot_RTD_compare(level=3, k_keep=18, Ndif=10, Pr=0.1,
                 n_particles=500, max_steps=2000)


# In[ ]:


# ==========================================================
# STEP 3: Concentration profiles (A vs P)
# - Visualize distribution of reactant (A) and product (P)
# - Compare uniform and fractal geometries slice-by-slice
# ==========================================================
# === Step 3: Concentration profile visualization (A vs P) ===
import matplotlib.pyplot as plt

def plot_concentration_profile(solid, Ndif=10, Pr=0.1,
                               mcs_total=800, mcs_burn=400, n_atm=50,
                               slice_axis=0, slice_index=None):
    """
    Run simulation and visualize final reactant (A) and product (P) distributions
    on a 2D slice of the 3D catalyst.
    """
    # Run simulation
    pore, surf, bpore, A, P = init_fields(solid, n_atm=n_atm)
    for _ in range(mcs_total):
        mcs_step_numba(solid, pore, bpore, A, P, Ndif=Ndif, Pr=Pr)

    # Select slice index
    if slice_index is None:
        slice_index = solid.shape[slice_axis] // 2  # take middle slice

    if slice_axis == 0:
        A_slice, P_slice = A[slice_index,:,:], P[slice_index,:,:]
    elif slice_axis == 1:
        A_slice, P_slice = A[:,slice_index,:], P[:,slice_index,:]
    else:
        A_slice, P_slice = A[:,:,slice_index], P[:,:,slice_index]

    # Plot
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    im0 = axes[0].imshow(A_slice, origin="lower", cmap="Blues")
    axes[0].set_title("Reactant A (slice)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(P_slice, origin="lower", cmap="Reds")
    axes[1].set_title("Product P (slice)")
    plt.colorbar(im1, ax=axes[1])

    plt.suptitle("Concentration profile in catalyst slice")
    plt.show()

# Uniform vs Fractal 농도 profile 비교
solid_uni = build_uniform(level=3)
solid_frac = build_menger(level=3, k_keep=18)

plot_concentration_profile(solid_uni, Ndif=10, Pr=0.1,
                           mcs_total=800, n_atm=50, slice_axis=0)

plot_concentration_profile(solid_frac, Ndif=10, Pr=0.1,
                           mcs_total=800, n_atm=50, slice_axis=0)


# In[ ]:


# ==========================================================
# STEP 4: Optimal fractal dimension search
# - Sweep different k_keep values (mapping to Df)
# - Identify the Df that maximizes apparent rate
# ==========================================================
# === Step 4: Df sweep – find optimal fractal dimension ===
import numpy as np
import matplotlib.pyplot as plt

def df_from_kkeep(k_keep, level=3):
    """
    Estimate fractal dimension for Menger-like structure.
    Df ≈ log(k_keep)/log(3) (single generation basis).
    """
    return np.log(k_keep) / np.log(3)

def sweep_Df(level=3, k_keep_list=(12,14,16,18,20,22,24),
             Ndif=10, Pr=0.08, 
             mcs_total=1000, mcs_burn=500, n_atm=50):
    Ds, rates = [], []
    for kk in k_keep_list:
        solid = build_menger(level, k_keep=kk)
        intr, app, *_ = run_sim_numba(solid, Ndif=Ndif, Pr=Pr,
                                      mcs_total=mcs_total, mcs_burn=mcs_burn,
                                      n_atm=n_atm)
        Ds.append(df_from_kkeep(kk, level=level))
        rates.append(app)
    return np.array(Ds), np.array(rates)

def plot_optimal_D(level=3, k_keep_list=(12,14,16,18,20,22,24),
                   Ndif=10, Pr=0.08,
                   mcs_total=1000, mcs_burn=500, n_atm=50):
    Ds, rates = sweep_Df(level, k_keep_list,
                         Ndif=Ndif, Pr=Pr,
                         mcs_total=mcs_total, mcs_burn=mcs_burn,
                         n_atm=n_atm)
    plt.figure(figsize=(5,4))
    plt.plot(Ds, rates, "o-", label=f"Ndif={Ndif}, Pr={Pr}")
    plt.xlabel("Fractal dimension $D_f$")
    plt.ylabel("Apparent rate (arb. units)")
    plt.title("Search for optimal $D_f$")
    plt.grid(True)
    plt.legend()
    plt.show()

    idx = np.argmax(rates)
    print(f"Optimal Df ≈ {Ds[idx]:.2f}, rate = {rates[idx]:.3f}")

# ==========================================================
# STEP 4: Optimal fractal dimension search
# - Sweep different k_keep values (mapping to Df)
# - Identify the Df that maximizes apparent rate
# ==========================================================
# Step 4 test
plot_optimal_D(level=3,
               k_keep_list=[12,14,16,18,20,22,24],
               Ndif=10, Pr=0.08,
               mcs_total=1200, mcs_burn=600, n_atm=80)


# In[ ]:


# ==========================================================
# STEP 5: Sweep over reaction probability (Pr)
# - Analyze how probability of reaction at surface impacts rate
# - Interplay between diffusion and reactivity
# ==========================================================
# === Step 5: Effect of reaction probability (Pr) on optimal Df ===
def plot_Pr_sweep(level=3, k_keep_list=(12,14,16,18,20,22,24),
                  Ndif=10, Pr_list=(0.03,0.05,0.08,0.12,0.2),
                  mcs_total=800, mcs_burn=400, n_atm=50):
    plt.figure(figsize=(6,4))
    for Pr in Pr_list:
        Ds, rates = sweep_Df(level=level, k_keep_list=k_keep_list,
                             Ndif=Ndif, Pr=Pr,
                             mcs_total=mcs_total, mcs_burn=mcs_burn,
                             n_atm=n_atm)
        plt.plot(Ds, rates, "o-", label=f"Pr={Pr}")
    plt.xlabel("Fractal dimension $D_f$")
    plt.ylabel("Apparent rate (arb. units)")
    plt.title(f"Pr sweep (Ndif={Ndif})")
    plt.grid(True)
    plt.legend()
    plt.show()


# In[ ]:


# ==========================================================
# STEP 5: Sweep over reaction probability (Pr)
# - Analyze how probability of reaction at surface impacts rate
# - Interplay between diffusion and reactivity
# ==========================================================
# Step 5 run
plot_Pr_sweep(level=3,
              k_keep_list=[12,14,16,18,20,22,24],
              Ndif=10,
              Pr_list=[0.03,0.05,0.08,0.12,0.2],
              mcs_total=1000, mcs_burn=500, n_atm=80)


# In[ ]:


# ==========================================================
# STEP 6: Thiele modulus analytical model
# - Connect fractal dimension Df to Knudsen diffusivity
# - Compute effectiveness factor η(phi)
# ==========================================================
# === Step 6: Monte Carlo vs Thiele model comparison ===
def thiele_knudsen_diffusivity(Df, r0=1.0):
    # 단순 모델: D_K ~ (3 - Df)
    return np.maximum(r0*(3.0 - Df), 1e-6)

def thiele_modulus(Df, k=1.0, L=1.0):
    DK = thiele_knudsen_diffusivity(Df)
    return L * np.sqrt(k / DK)

def thiele_eta(phi):
    if np.isscalar(phi):
        if phi < 1e-8: return 1.0
        return (3.0/phi)*(1.0/np.tanh(phi) - 1.0/phi)
    else:
        phi = np.asarray(phi, dtype=float)
        out = np.empty_like(phi)
        small = phi < 1e-8
        out[small] = 1.0
        ps = phi[~small]
        out[~small] = (3.0/ps)*(1.0/np.tanh(ps) - 1.0/ps)
        return out

def compare_MC_vs_Thiele(level=3, k_keep_list=(12,14,16,18,20,22,24),
                         Ndif=10, Pr=0.08,
                         mcs_total=800, mcs_burn=400, n_atm=50):
    # --- Monte Carlo ---
    Ds, rates = sweep_Df(level, k_keep_list, Ndif, Pr,
                         mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)

    # --- Thiele model ---
    Df_vals = np.linspace(min(Ds), max(Ds), 100)
    phi = thiele_modulus(Df_vals, k=1.0, L=1.0)
    eta = thiele_eta(phi)

    # --- Plot ---
    plt.figure(figsize=(6,4))
    plt.plot(Ds, rates/np.max(rates), "o-", label="Monte Carlo (normalized)")
    plt.plot(Df_vals, eta/np.max(eta), "-", label="Thiele model (normalized)")
    plt.xlabel("Fractal dimension $D_f$")
    plt.ylabel("Relative performance")
    plt.title("MC vs Thiele model")
    plt.grid(True)
    plt.legend()
    plt.show()


# In[ ]:


# ==========================================================
# STEP 6: Thiele modulus analytical model
# - Connect fractal dimension Df to Knudsen diffusivity
# - Compute effectiveness factor η(phi)
# ==========================================================
# Step 6 run
compare_MC_vs_Thiele(level=3,
                     k_keep_list=[12,14,16,18,20,22,24],
                     Ndif=10, Pr=0.08,
                     mcs_total=1000, mcs_burn=500, n_atm=80)


# In[ ]:


# ==========================================================
# STEP 7–12: Heatmap and normalized analysis
# - Sweep across Df and Pr
# - Normalize by pore volume to remove trivial scaling
# - Generate contour maps to locate optimal region
# ==========================================================
# === Step 7: Reaction probability (Pr) sweep ===
def sweep_Pr_vs_D(level=3, k_keep_list=(12,14,16,18,20,22,24),
                  Ndif=10, Pr_list=(0.03,0.05,0.08,0.12,0.2),
                  mcs_total=800, mcs_burn=400, n_atm=50):
    """
    여러 Pr 조건에서 Df vs rate 곡선 비교
    """
    plt.figure(figsize=(7,5))
    for Pr in Pr_list:
        Ds, rates = sweep_Df(level, k_keep_list, Ndif, Pr,
                             mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)
        rates_norm = rates / np.max(rates)  # 조건별 정규화
        plt.plot(Ds, rates_norm, "o-", label=f"Pr={Pr}")
    plt.xlabel("Fractal dimension $D_f$")
    plt.ylabel("Normalized apparent rate")
    plt.title(f"Effect of Pr on optimal Df (Ndif={Ndif})")
    plt.grid(True)
    plt.legend()
    plt.show()

# ==========================================================
# STEP 7–12: Heatmap and normalized analysis
# - Sweep across Df and Pr
# - Normalize by pore volume to remove trivial scaling
# - Generate contour maps to locate optimal region
# ==========================================================
# Step 7 run
sweep_Pr_vs_D(level=3,
              k_keep_list=[12,14,16,18,20,22,24],
              Ndif=10,
              Pr_list=[0.03,0.05,0.08,0.12,0.2],
              mcs_total=1000, mcs_burn=500, n_atm=80)


# In[ ]:


# === Step 8: Transient behavior (rate vs time) ===
def plot_transient_rates(level=3, k_keep=18,
                         Ndif=10, Pr=0.08,
                         mcs_total=1500, n_atm=50):
    """
    Track intrinsic & apparent reaction rates vs time steps.
    Compare uniform vs fractal structures.
    """
    solid_uni = build_uniform(level)
    solid_frac = build_menger(level, k_keep=k_keep)

    # Uniform run
    _, _, intr_u, app_u = run_sim_numba(solid_uni, Ndif=Ndif, Pr=Pr,
                                        mcs_total=mcs_total, n_atm=n_atm)
    # Fractal run
    _, _, intr_f, app_f = run_sim_numba(solid_frac, Ndif=Ndif, Pr=Pr,
                                        mcs_total=mcs_total, n_atm=n_atm)

    t = np.arange(mcs_total)

    plt.figure(figsize=(7,5))
    plt.plot(t, app_u, "-", alpha=0.7, label="Uniform (apparent)")
    plt.plot(t, app_f, "-", alpha=0.7, label=f"Fractal k_keep={k_keep} (apparent)")
    plt.xlabel("Monte Carlo steps")
    plt.ylabel("Apparent rate per step")
    plt.title(f"Transient apparent rate (Ndif={Ndif}, Pr={Pr})")
    plt.legend(); plt.grid(True)
    plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(t, intr_u, "-", alpha=0.7, label="Uniform (intrinsic)")
    plt.plot(t, intr_f, "-", alpha=0.7, label=f"Fractal k_keep={k_keep} (intrinsic)")
    plt.xlabel("Monte Carlo steps")
    plt.ylabel("Intrinsic reaction events per step")
    plt.title(f"Transient intrinsic rate (Ndif={Ndif}, Pr={Pr})")
    plt.legend(); plt.grid(True)
    plt.show()

# Step 8 run
plot_transient_rates(level=3, k_keep=18,
                     Ndif=10, Pr=0.08,
                     mcs_total=1500, n_atm=50)


# In[ ]:


# === Step 9: Normalize rates by pore volume ===
def normalized_rate(solid, Ndif=10, Pr=0.08,
                    mcs_total=1200, mcs_burn=600, n_atm=50):
    """
    Run simulation and return apparent rate normalized by pore volume.
    """
    pore = (~solid).sum()   # total pore voxel count
    intr_ss, app_ss, _, _ = run_sim_numba(solid, Ndif=Ndif, Pr=Pr,
                                          mcs_total=mcs_total,
                                          mcs_burn=mcs_burn, n_atm=n_atm)
    return intr_ss/pore, app_ss/pore

def sweep_normalized(level=3, k_keep_list=(12,14,16,18,20,22,24),
                     Ndif=10, Pr=0.08,
                     mcs_total=1200, mcs_burn=600, n_atm=50):
    """
    여러 fractal dimension 후보에 대해 normalized apparent rate 계산
    """
    Ds, rates = [], []
    for kk in k_keep_list:
        solid = build_menger(level, k_keep=kk)
        intr_n, app_n = normalized_rate(solid, Ndif=Ndif, Pr=Pr,
                                        mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)
        Df = np.log(kk)/np.log(3)   # fractal dimension
        Ds.append(Df); rates.append(app_n)
    return np.array(Ds), np.array(rates)

def plot_normalized_optimal(level=3, k_keep_list=(12,14,16,18,20,22,24),
                            Ndif=10, Pr=0.08,
                            mcs_total=1200, mcs_burn=600, n_atm=50):
    Ds, rates = sweep_normalized(level, k_keep_list=k_keep_list,
                                 Ndif=Ndif, Pr=Pr,
                                 mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)
    plt.figure(figsize=(6,4))
    plt.plot(Ds, rates, "o-", label="Normalized apparent rate")
    plt.axvline(2.6, ls="--", color="r", alpha=0.6, label="~Reported optimum")
    plt.xlabel("Fractal dimension $D_f$")
    plt.ylabel("Normalized apparent rate (per pore voxel)")
    plt.title("Optimal fractal dimension (normalized by pore volume)")
    plt.legend(); plt.grid(True); plt.show()

    idx = np.argmax(rates)
    print(f"▶ 최적 Df ≈ {Ds[idx]:.2f}, normalized rate = {rates[idx]:.4f}")


# Step 9 run
plot_normalized_optimal(level=3,
                        k_keep_list=[12,14,16,18,20,22,24],
                        Ndif=10, Pr=0.08,
                        mcs_total=1200, mcs_burn=600, n_atm=80)


# In[ ]:


# ==========================================================
# STEP 1: Imports and lattice definitions
# - Import core libraries (numpy, matplotlib)
# - Define 6-connected neighbors for cubic lattice
# ==========================================================
# === Step 10: 2D heatmap of (Df, Pr) space ===
def scan_Df_Pr(level=3, k_keep_list=(12,14,16,18,20,22,24),
               Pr_list=(0.03,0.05,0.08,0.12,0.2),
               Ndif=10, mcs_total=800, mcs_burn=400, n_atm=50):
    """
    Sweep over (Df, Pr) and return normalized apparent rates.
    """
    Ds = [np.log(k)/np.log(3) for k in k_keep_list]
    Prs = list(Pr_list)
    results = np.zeros((len(Prs), len(Ds)))

    for i, Pr in enumerate(Prs):
        for j, kk in enumerate(k_keep_list):
            solid = build_menger(level, k_keep=kk)
            _, app_n = normalized_rate(solid, Ndif=Ndif, Pr=Pr,
                                       mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)
            results[i,j] = app_n
    return np.array(Ds), np.array(Prs), results

def plot_heatmap(level=3, k_keep_list=(12,14,16,18,20,22,24),
                 Pr_list=(0.03,0.05,0.08,0.12,0.2),
                 Ndif=10, mcs_total=800, mcs_burn=400, n_atm=50):
    Ds, Prs, R = scan_Df_Pr(level, k_keep_list, Pr_list,
                             Ndif=Ndif, mcs_total=mcs_total,
                             mcs_burn=mcs_burn, n_atm=n_atm)
    plt.figure(figsize=(7,5))
    im = plt.imshow(R, origin="lower", aspect="auto",
                    extent=[min(Ds), max(Ds), min(Prs), max(Prs)],
                    cmap="viridis")
    plt.colorbar(im, label="Normalized apparent rate")
    plt.xlabel("Fractal dimension $D_f$")
    plt.ylabel("Reaction probability Pr")
    plt.title("Heatmap of performance vs (Df, Pr)")
    plt.show()


plot_heatmap(level=3,
             k_keep_list=[12,14,16,18,20,22,24],
             Pr_list=[0.03,0.05,0.08,0.12,0.2],
             Ndif=10, mcs_total=800, mcs_burn=400, n_atm=80)



# In[ ]:


# ==========================================================
# STEP 1: Imports and lattice definitions
# - Import core libraries (numpy, matplotlib)
# - Define 6-connected neighbors for cubic lattice
# ==========================================================
# === Step 11: Publication-style contour plot with optimum marker ===
from scipy.ndimage import gaussian_filter

def plot_contour_with_optimum(level=3, 
                              k_keep_list=(12,14,16,18,20,22,24),
                              Pr_list=(0.03,0.05,0.08,0.12,0.2),
                              Ndif=10, mcs_total=800, mcs_burn=400, n_atm=80):
    Ds, Prs, R = scan_Df_Pr(level, k_keep_list, Pr_list,
                             Ndif=Ndif, mcs_total=mcs_total,
                             mcs_burn=mcs_burn, n_atm=n_atm)

    # Smoothing for prettier contours
    R_smooth = gaussian_filter(R, sigma=1.0)

    # Find optimum
    idx = np.unravel_index(np.argmax(R_smooth), R_smooth.shape)
    Pr_opt = Prs[idx[0]]
    Df_opt = Ds[idx[1]]
    R_opt  = R_smooth[idx]

    # Contour plot
    plt.figure(figsize=(7,5))
    cs = plt.contourf(Ds, Prs, R_smooth, levels=15, cmap="viridis")
    plt.colorbar(cs, label="Normalized apparent rate")

    # Overlay contours
    plt.contour(Ds, Prs, R_smooth, levels=6, colors="k", linewidths=0.5)

    # Optimum marker
    plt.plot(Df_opt, Pr_opt, "ro", markersize=8, label=f"Optimum Df={Df_opt:.2f}, Pr={Pr_opt:.2f}")

    plt.xlabel("Fractal dimension $D_f$")
    plt.ylabel("Reaction probability Pr")
    plt.title("Contour map of performance vs (Df, Pr)")
    plt.legend()
    plt.grid(False)
    plt.show()

    print(f"▶ Optimum at Df={Df_opt:.2f}, Pr={Pr_opt:.2f}, Rate={R_opt:.3f}")

plot_contour_with_optimum(level=3,
                          k_keep_list=[12,14,16,18,20,22,24],
                          Pr_list=[0.03,0.05,0.08,0.12,0.2],
                          Ndif=10, mcs_total=1000, mcs_burn=500, n_atm=80)


# In[ ]:


# ==========================================================
# STEP 1: Imports and lattice definitions
# - Import core libraries (numpy, matplotlib)
# - Define 6-connected neighbors for cubic lattice
# ==========================================================
# === Step 12: Combined figure (Df vs rate + Df–Pr contour) ===
from matplotlib.gridspec import GridSpec

def plot_combined_fig(level=3,
                      k_keep_list=(12,14,16,18,20,22,24),
                      Pr_list=(0.03,0.05,0.08,0.12,0.2),
                      Ndif=10, mcs_total=1000, mcs_burn=500, n_atm=80):
    # --- 1) Line plot: Df vs Rate at a chosen Pr ---
    chosen_Pr = 0.08  # 중간 값 하나 선택
    Ds, rates = sweep_Df(level=level, k_keep_list=k_keep_list,
                         Ndif=Ndif, Pr=chosen_Pr,
                         mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)

    # --- 2) Contour data: Df–Pr heatmap ---
    Ds2, Prs, R = scan_Df_Pr(level=level, k_keep_list=k_keep_list, Pr_list=Pr_list,
                             Ndif=Ndif, mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)
    R_smooth = gaussian_filter(R, sigma=1.0)

    # Find optimum
    idx = np.unravel_index(np.argmax(R_smooth), R_smooth.shape)
    Pr_opt = Prs[idx[0]]
    Df_opt = Ds2[idx[1]]
    R_opt  = R_smooth[idx]

    # --- Combined plot layout ---
    fig = plt.figure(figsize=(12,5))
    gs = GridSpec(1, 2, width_ratios=[1,1.2], figure=fig)

    # (a) Left: line plot
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(Ds, rates, "o-", label=f"Pr={chosen_Pr}")
    ax1.axvline(Df_opt, color="r", linestyle="--", label=f"Optimum Df={Df_opt:.2f}")
    ax1.set_xlabel("Fractal dimension $D_f$")
    ax1.set_ylabel("Normalized rate")
    ax1.set_title("(a) Df vs Rate")
    ax1.legend()
    ax1.grid(True)

    # (b) Right: contour map
    ax2 = fig.add_subplot(gs[0,1])
    cs = ax2.contourf(Ds2, Prs, R_smooth, levels=15, cmap="viridis")
    fig.colorbar(cs, ax=ax2, label="Normalized apparent rate")
    ax2.contour(Ds2, Prs, R_smooth, levels=6, colors="k", linewidths=0.5)
    ax2.plot(Df_opt, Pr_opt, "ro", markersize=8, label="Optimum")
    ax2.set_xlabel("Fractal dimension $D_f$")
    ax2.set_ylabel("Reaction probability Pr")
    ax2.set_title("(b) Contour map")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"▶ Optimum at Df={Df_opt:.2f}, Pr={Pr_opt:.2f}, Rate={R_opt:.3f}")

plot_combined_fig(level=3,
                  k_keep_list=[12,14,16,18,20,22,24],
                  Pr_list=[0.03,0.05,0.08,0.12,0.2],
                  Ndif=10, mcs_total=1000, mcs_burn=500, n_atm=80)


# In[ ]:


# ==========================================================
# STEP 1: Imports and lattice definitions
# - Import core libraries (numpy, matplotlib)
# - Define 6-connected neighbors for cubic lattice
# ==========================================================
# === Step 13: Figure with caption ===
def plot_combined_with_caption(level=3,
                               k_keep_list=(12,14,16,18,20,22,24),
                               Pr_list=(0.03,0.05,0.08,0.12,0.2),
                               Ndif=10, mcs_total=1000, mcs_burn=500, n_atm=80):
    # --- 데이터 준비 ---
    chosen_Pr = 0.08
    Ds, rates = sweep_Df(level=level, k_keep_list=k_keep_list,
                         Ndif=Ndif, Pr=chosen_Pr,
                         mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)

    Ds2, Prs, R = scan_Df_Pr(level=level, k_keep_list=k_keep_list, Pr_list=Pr_list,
                             Ndif=Ndif, mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)
    R_smooth = gaussian_filter(R, sigma=1.0)
    idx = np.unravel_index(np.argmax(R_smooth), R_smooth.shape)
    Pr_opt, Df_opt, R_opt = Prs[idx[0]], Ds2[idx[1]], R_smooth[idx]

    # --- 그림 배치 ---
    fig, axes = plt.subplots(1,2, figsize=(12,5))

    # (a) Line plot
    axes[0].plot(Ds, rates, "o-", label=f"Pr={chosen_Pr}")
    axes[0].axvline(Df_opt, color="r", linestyle="--", label=f"Optimum Df={Df_opt:.2f}")
    axes[0].set_xlabel("Fractal dimension $D_f$")
    axes[0].set_ylabel("Normalized rate")
    axes[0].set_title("(a) Df vs Rate")
    axes[0].legend()
    axes[0].grid(True)

    # (b) Contour map
    cs = axes[1].contourf(Ds2, Prs, R_smooth, levels=15, cmap="viridis")
    fig.colorbar(cs, ax=axes[1], label="Normalized apparent rate")
    axes[1].contour(Ds2, Prs, R_smooth, levels=6, colors="k", linewidths=0.5)
    axes[1].plot(Df_opt, Pr_opt, "ro", markersize=8, label="Optimum")
    axes[1].set_xlabel("Fractal dimension $D_f$")
    axes[1].set_ylabel("Reaction probability Pr")
    axes[1].set_title("(b) Contour map")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # --- 논문 스타일 캡션 ---
    caption = (
        "Figure X. (a) Normalized reaction rate as a function of fractal dimension $D_f$ "
        f"at fixed probability Pr={chosen_Pr}. The maximum occurs near $D_f={Df_opt:.2f}$. "
        "(b) Contour map of normalized apparent reaction rate as a function of "
        "fractal dimension $D_f$ and reaction probability Pr. "
        f"The global optimum is located at $D_f={Df_opt:.2f}$ and Pr={Pr_opt:.2f}, "
        f"with normalized rate ≈ {R_opt:.3f}. "
        "These results illustrate the existence of an optimal fractal dimension "
        "where the trade-off between increased surface area and diffusion/side-reaction limitations "
        "is balanced."
    )
    print(caption)

plot_combined_with_caption(level=3,
                           k_keep_list=[12,14,16,18,20,22,24],
                           Pr_list=[0.03,0.05,0.08,0.12,0.2],
                           Ndif=10, mcs_total=1000, mcs_burn=500, n_atm=80)


# In[ ]:


# ==========================================================
# STEP 1: Imports and lattice definitions
# - Import core libraries (numpy, matplotlib)
# - Define 6-connected neighbors for cubic lattice
# ==========================================================
# === Step 14: Save figure to PNG/PDF ===
def save_combined_figure(level=3,
                         k_keep_list=(12,14,16,18,20,22,24),
                         Pr_list=(0.03,0.05,0.08,0.12,0.2),
                         Ndif=10, mcs_total=1000, mcs_burn=500, n_atm=80,
                         filename="figure_optimum"):
    # --- 데이터 준비 ---
    chosen_Pr = 0.08
    Ds, rates = sweep_Df(level=level, k_keep_list=k_keep_list,
                         Ndif=Ndif, Pr=chosen_Pr,
                         mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)

    Ds2, Prs, R = scan_Df_Pr(level=level, k_keep_list=k_keep_list, Pr_list=Pr_list,
                             Ndif=Ndif, mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)
    R_smooth = gaussian_filter(R, sigma=1.0)
    idx = np.unravel_index(np.argmax(R_smooth), R_smooth.shape)
    Pr_opt, Df_opt, R_opt = Prs[idx[0]], Ds2[idx[1]], R_smooth[idx]

    # --- 그림 그리기 ---
    fig, axes = plt.subplots(1,2, figsize=(12,5))

    # (a) Line plot
    axes[0].plot(Ds, rates, "o-", label=f"Pr={chosen_Pr}")
    axes[0].axvline(Df_opt, color="r", linestyle="--", label=f"Optimum Df={Df_opt:.2f}")
    axes[0].set_xlabel("Fractal dimension $D_f$")
    axes[0].set_ylabel("Normalized rate")
    axes[0].set_title("(a) Df vs Rate")
    axes[0].legend()
    axes[0].grid(True)

    # (b) Contour plot
    cs = axes[1].contourf(Ds2, Prs, R_smooth, levels=15, cmap="viridis")
    fig.colorbar(cs, ax=axes[1], label="Normalized apparent rate")
    axes[1].contour(Ds2, Prs, R_smooth, levels=6, colors="k", linewidths=0.5)
    axes[1].plot(Df_opt, Pr_opt, "ro", markersize=8, label="Optimum")
    axes[1].set_xlabel("Fractal dimension $D_f$")
    axes[1].set_ylabel("Reaction probability Pr")
    axes[1].set_title("(b) Contour map")
    axes[1].legend()

    plt.tight_layout()

    # --- 저장 ---
    png_file = f"{filename}.png"
    pdf_file = f"{filename}.pdf"
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"✅ Figures saved as {png_file} and {pdf_file}")

save_combined_figure(level=3,
                     k_keep_list=[12,14,16,18,20,22,24],
                     Pr_list=[0.03,0.05,0.08,0.12,0.2],
                     Ndif=10, mcs_total=1000, mcs_burn=500, n_atm=80,
                     filename="Optimal_Fractal_Dimension")


# In[6]:


# ==========================================================
# STEP 5: Sweep over reaction probability (Pr)
# - Analyze how probability of reaction at surface impacts rate
# - Interplay between diffusion and reactivity
# ==========================================================
# === Step 5: Pr 스윕 & 레짐 시각화 (uniform vs fractal, Ndif=1/10) ===


def sweep_Pr_curves(struct="fractal", level=3, k_keep=20, Ndif=1, Pr_list=None,
                    mcs_total=1200, mcs_burn=600, n_atm=50, seed=0):
    """
    주어진 구조(struct)와 Ndif에서 Pr를 스윕하여 apparent rate 곡선을 반환
    """
    rng = np.random.default_rng(seed)
    if Pr_list is None:
        Pr_list = np.geomspace(5e-3, 1.0, 10)
    rates = []
    for Pr in Pr_list:
        # structure 생성
        if struct == "fractal":
            solid = build_menger(level, k_keep)
        else:
            solid = build_uniform(level)

        # simulation run
        intr, app, *_ = run_sim_numba(solid, Ndif=Ndif, Pr=Pr, 
                                 mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)
        rates.append(app)
    return np.array(Pr_list), np.array(rates)

def plot_regimes(level=3, k_keep=20, mcs_total=1200, mcs_burn=600, n_atm=50):
    """
    uniform vs fractal, Ndif=1/10에 대해 log–log 플롯과 구간별 기울기(선형회귀)를 출력
    """
    setups = [("uniform", 1), ("uniform", 10), ("fractal", 1), ("fractal", 10)]
    for struct, Ndif in setups:
        Pr, rate = sweep_Pr_curves(struct=struct, level=level, k_keep=k_keep, Ndif=Ndif,
                                   mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)
        m = rate > 0
        Pr, rate = Pr[m], rate[m]
        if len(rate) == 0:
            print(f"[{struct}, Ndif={Ndif}] all-zero rates. Increase mcs_total/n_atm or widen Pr_list.")
            continue

        plt.figure(figsize=(5,4))
        plt.plot(Pr, rate, "o-")
        plt.xscale("log"); plt.yscale("log")
        plt.title(f"{struct}, Ndif={Ndif} — log(rate) vs log(Pr)")
        plt.xlabel("Pr"); plt.ylabel("apparent rate")
        plt.grid(True)
        plt.show()

        # 구간별 기울기(저/중/고 Pr)
        def slope(lo, hi):
            mask = (Pr>=lo) & (Pr<=hi)
            if mask.sum() < 2: return np.nan
            x = np.log10(Pr[mask]); y = np.log10(rate[mask])
            a, b = np.polyfit(x, y, 1)
            return a

        s_low  = slope(5e-3, 3e-2)
        s_mid  = slope(3e-2, 3e-1)
        s_high = slope(3e-1, 1.0)
        print(f"[{struct}, Ndif={Ndif}] slopes  low≈{s_low:.2f}, mid≈{s_mid:.2f}, high≈{s_high:.2f}")

# 바로 run example (시간이 좀 걸릴 수 있음)
# plot_regimes(level=3, k_keep=20, mcs_total=1200, mcs_burn=600, n_atm=50)


# In[ ]:


# ==========================================================
# STEP 6: Thiele modulus analytical model
# - Connect fractal dimension Df to Knudsen diffusivity
# - Compute effectiveness factor η(phi)
# ==========================================================
# === Step 6: Df 스윕으로 최적 D 찾기 ===


def df_from_kkeep(k_keep, level=3):
    """
    Menger-like fractal에서 fractal dimension 계산:
    Df = log(k_keep)/log(3)  (단일 세대 기준)
    """
    return np.log(k_keep) / np.log(3)

def sweep_Df(level=3, k_keep_list=(10, 15, 20), 
             Ndif=1, Pr=0.05, 
             mcs_total=600, mcs_burn=300, n_atm=20, seed=0):
    """
    서로 다른 k_keep (즉 서로 다른 Df)에 대해 apparent rate 비교
    """
    rng = np.random.default_rng(seed)
    Ds, rates = [], []
    for kk in k_keep_list:
        solid = build_menger(level, k_keep=kk)
        intr, app, *_ = run_sim_numba(solid, Ndif=Ndif, Pr=Pr,
                                mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)
        Ds.append(df_from_kkeep(kk, level=level))
        rates.append(app)
    return np.array(Ds), np.array(rates)

def plot_optimal_D(level=3, k_keep_list=(8,12,15,18,20),
                   Ndif=1, Pr=0.05, 
                   mcs_total=600, mcs_burn=300, n_atm=20):
    Ds, rates = sweep_Df(level=level, k_keep_list=k_keep_list,
                         Ndif=Ndif, Pr=Pr,
                         mcs_total=mcs_total, mcs_burn=mcs_burn, n_atm=n_atm)
    plt.figure(figsize=(5,4))
    plt.plot(Ds, rates, "o-")
    plt.xlabel("Fractal dimension $D_f$")
    plt.ylabel("Apparent rate (arb. units)")
    plt.title(f"Ndif={Ndif}, Pr={Pr} — Optimal D search")
    plt.grid(True)
    plt.show()

    idx = np.argmax(rates)
    print(f"최적 Df ≈ {Ds[idx]:.2f}, rate = {rates[idx]:.4f}")

# run example: 여러 Pr 값을 돌려보기
k_keep_list = [12,14,16,18,20,22,24]  # Df ≈ 2.26 ~ 2.89

for Pr in [0.03, 0.05, 0.08, 0.12, 0.2]:
    plot_optimal_D(
        level=3,
        k_keep_list=k_keep_list,
        Ndif=10,               # Knudsen regime
        Pr=Pr,                 # sweep over different Pr
        mcs_total=1200,
        mcs_burn=600,
        n_atm=80
    )


# === Step X: Thiele modulus 기반 해석 모델 추가 ===

def knudsen_diffusivity(Df, r0=1.0):
    """
    단순 모델: D_K ~ (3 - Df)
    (Df ↑ → 복잡해짐 → 유효 반경 ↓ → D_K ↓)
    """
    return np.maximum(r0*(3.0 - Df), 1e-6)

def thiele_modulus(Df, k=1.0, L=1.0):
    DK = knudsen_diffusivity(Df)
    return L * np.sqrt(k / DK)

def eta_effectiveness(phi):
    """
    구형 입자 1차 반응 공식: η(phi)
    """
    if np.isscalar(phi):
        if phi < 1e-8:
            return 1.0
        return (3.0/phi) * (1.0/np.tanh(phi) - 1.0/phi)
    else:
        phi = np.asarray(phi, dtype=float)
        out = np.empty_like(phi)
        small = phi < 1e-8
        out[small] = 1.0
        ps = phi[~small]
        out[~small] = (3.0/ps) * (1.0/np.tanh(ps) - 1.0/ps)
        return out

# Df 스윕 & 시각화
Df_vals = np.linspace(2.0, 3.0, 100)
phi = thiele_modulus(Df_vals, k=1.0, L=1.0)
eta = eta_effectiveness(phi)

plt.figure(figsize=(6,4))
plt.plot(Df_vals, eta, label="Effectiveness factor η(Df)")
plt.xlabel("Fractal dimension Df")
plt.ylabel("η (relative efficiency)")
plt.title("Thiele modulus model vs Df")
plt.legend()
plt.grid(True)
plt.show()


# ==========================================================
# STEP 6: Thiele modulus analytical model
# - Connect fractal dimension Df to Knudsen diffusivity
# - Compute effectiveness factor η(phi)
# ==========================================================
# === Step 6b: MC result와 Thiele 해석 모델 비교 ===
Df_vals = np.linspace(2.0, 3.0, 100)
phi = thiele_modulus(Df_vals, k=1.0, L=1.0)
eta = eta_effectiveness(phi)

# MC result
Ds, rates = sweep_Df(level=3, k_keep_list=[12,14,16,18,20,22,24],
                     Ndif=10, Pr=0.08, mcs_total=1200, mcs_burn=600, n_atm=80)

plt.figure(figsize=(6,4))
plt.plot(Ds, rates/rates.max(), "o-", label="MC simulation (normalized)")
plt.plot(Df_vals, eta/eta.max(), "-", label="Thiele modulus model")
plt.xlabel("Fractal dimension Df")
plt.ylabel("Relative performance")
plt.legend()
plt.title("MC vs Thiele model")
plt.grid(True)
plt.show()


# In[ ]:



