"""
moveCell.py – Overdamped mass–spring, chemo-only, identical agents
Sense → directional memory → noisy heading ODE → active force.
Springs propagate motion; soft repulsion prevents overlap.
"""

import numpy as np
import math
import random

# ---------- Polarisation ----------
epsGrad  = 1e-9      # Safety parameter for gradient sensing     (Numerical safety parameter)
tauSense = 0.25      # Memory time scale in time units           (~15 minutes of signal integration)
kChemo   = 3.0       # Angular relaxation rate per unit time     (~15 minutes for cell alignment)
gSat     = 0.5       # Cue saturation                            (~2× the detection threshold for saturation)

# ----- Mass–spring parameters -----
substeps = 10000     # Euler substeps per unit time              (Numerical safety parameter)
rCore    = 2.0       # Min spacing multiplier of cellRad         (Equilibrium cell separation)
kRep     = 1e3       # Short-range repulsion                     (High short-ranged repulsion) 
                     
kSpring  = 1e3       # Spring stiffness                          (Spring-like attraction)    
f0       = 1e3       # Active-force scale per cell               (Active force from chemotaxis)

# -------------- Helper functions ------------
def wrapAngle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def senseChemoAngle(i, VEGFArray, width, length, lenFilo, xi, c0):
    yy0 = int(round(i.y)); xx0 = int(round(i.x))
    cOld = VEGFArray[yy0, xx0]
    maxAng  = random.uniform(0, 2 * math.pi)
    maxGrad = 0.0
    for filAngle in i.angleList:
        vals = []
        for j in range(1, int(round(lenFilo)) + 1):
            yy = int(round(i.y + j * math.sin(filAngle)))
            xx = int(round(i.x + j * math.cos(filAngle)))
            if 0 <= yy < width and 0 <= xx < length:
                vals.append(VEGFArray[yy, xx])
            else:
                vals.append(0.0)
        cNew = np.mean(vals) if vals else 0.0
        if cNew > maxGrad:
            maxGrad, maxAng = cNew, filAngle

    return maxAng, maxGrad, cOld

def projectToBox(p, cellRad, width, length):
    p.x = min(max(p.x, cellRad), length - cellRad)
    p.y = min(max(p.y, cellRad), width  - cellRad)

def vecToAng(v):
    n = np.linalg.norm(v)
    return None if n == 0 else math.atan2(v[1], v[0])

def unit(a):
    return np.array([math.cos(a), math.sin(a)])

def repositionTrailingCells(cellList, nMove, cellRad, domainLen,
                            gap=None, clearBonds=True):
    if not cellList:
        return
    nMove = min(nMove, len(cellList))
    gap = 2 * cellRad if gap is None else float(gap)

    trailing = sorted(cellList, key=lambda c: c.x)[:nMove]
    leadX = max(cellList, key=lambda c: c.x).x

    x_min = min(c.x for c in trailing)
    x_max = max(c.x for c in trailing)

    dx_target = (leadX + gap) - x_min
    if dx_target <= 0:
        return

    dx_fit = (domainLen - cellRad) - x_max
    dx = min(dx_target, dx_fit)

    for c in trailing:
        c.x = min(max(c.x + dx, cellRad), domainLen - cellRad)
        if clearBonds and hasattr(c, "bonds"):
            c.bonds.clear()

# ---------- Bonds and springs ----------
def ensureBonds(cellList, RLink, RBreak):
    live = set(cellList)
    for c in cellList:
        if not hasattr(c, "bonds"):
            c.bonds = {}

    for i in list(live):
        dead = [j for j in list(i.bonds.keys()) if j not in live]
        for j in dead:
            del i.bonds[j]
            if hasattr(j, "bonds") and i in j.bonds:
                del j.bonds[i]

    for i in cellList:
        for j in cellList:
            if j is i:
                continue
            r = math.hypot(j.x - i.x, j.y - i.y)
            if r <= RLink and j not in i.bonds:
                i.bonds[j] = r
                j.bonds[i] = r

    for i in cellList:
        dead = [j for j, L0 in i.bonds.items()
                if math.hypot(i.x - j.x, i.y - j.y) > RBreak]
        for j in dead:
            del i.bonds[j]
            if i in j.bonds:
                del j.bonds[i]

def massSpringStep(kSpring, f0, cellList, dt, cellRad,
                   width, length, RLink, RBreak, posNoise=0.1):
    ensureBonds(cellList, RLink, RBreak)
    
    nsub = max(1, int(round(substeps * dt)))
    h = dt / nsub

    for _ in range(nsub):
        Fx = {i: 0.0 for i in cellList}
        Fy = {i: 0.0 for i in cellList}

        # Active force
        for i in cellList:
            s = float(getattr(i, "chemoCue", 0.0))
            ux, uy = unit(getattr(i, "chainAngle", 0.0))
            Fx[i] += f0 * s * ux
            Fy[i] += f0 * s * uy

        # Springs
        done = set()
        for i in cellList:
            for j, L0 in list(getattr(i, "bonds", {}).items()):
                if j not in Fx:
                    del i.bonds[j]
                    if hasattr(j, "bonds") and i in j.bonds:
                        del j.bonds[i]
                    continue
                key = (min(id(i), id(j)), max(id(i), id(j)))
                if key in done:
                    continue
                dx, dy = j.x - i.x, j.y - i.y
                r = math.hypot(dx, dy) + 1e-12
                nx, ny = dx / r, dy / r
                F = kSpring * (r - L0)
                Fx_i = F * nx; Fy_i = F * ny
                Fx[i] +=  Fx_i; Fy[i] +=  Fy_i
                Fx[j] += -Fx_i; Fy[j] += -Fy_i
                done.add(key)

        # Soft-core repulsion
        r_target = rCore * cellRad
        for i in cellList:
            for j in cellList:
                if j is i:
                    continue
                dx, dy = j.x - i.x, j.y - i.y
                r = math.hypot(dx, dy) + 1e-12
                if r < r_target:
                    nx, ny = dx / r, dy / r
                    F = kRep * (r_target - r)
                    Fx[i] += -F * nx
                    Fy[i] += -F * ny

        # Integrate 
        for i in cellList:
            jitterx = posNoise * random.uniform(-1, 1) if posNoise else 0.0
            jittery = posNoise * random.uniform(-1, 1) if posNoise else 0.0
            i.x += Fx[i] * h + jitterx
            i.y += Fy[i] * h + jittery
            projectToBox(i, cellRad, width, length)

# ---------------- Main API ----------------
def moveCells(VEGFArray, cellList, filoNum, lenFilo,
              lenFiloMax, xi, c0, cellRad,
              oldLen, newLen, kSpring, f0, dt=1.0/60.0, posNoise=0.1):
    """
    Per call advances time by dt.
      1) Sense chemo, update directional memory, update heading.
      2) Overdamped mass–spring update.
      3) Axial advection for domain growth.
    """
    width, length = VEGFArray.shape
    filopList = []
    random.shuffle(cellList)

    # Small level of hysterisis on bond formation
    RLink  = float(lenFiloMax)
    RBreak = 1.1 * RLink   

    # Chemo sensing + polarisation
    for i in cellList:
        if not hasattr(i, "chainAngle") or i.chainAngle is None:
            i.chainAngle = random.uniform(0, 2 * math.pi)
        if not hasattr(i, 'filPersist'):
            i.filPersist = 0
        if not hasattr(i, "senseAvg"):
            i.senseAvg = np.zeros(2)

        i.angleList = [random.uniform(0, 2 * math.pi) for _ in range(filoNum)]
        i.filPersist = 0

        angChemo, maxGrad, cOld = senseChemoAngle(
            i, VEGFArray, width, length, lenFilo, xi, c0
        )

        # Scalar cue in [0,1]
        try:
            rel = (maxGrad - cOld) / max(cOld, epsGrad)
            thresh = xi * math.sqrt(c0 / max(cOld, epsGrad))
        except Exception:
            rel = (maxGrad - cOld)
            thresh = xi * math.sqrt(c0)
        sChemo = 0.0 if rel <= thresh else min(1.0, (rel - thresh) / (gSat * thresh))

        # Exponential directional memory
        alpha = min(1.0, dt / max(1e-9, tauSense))
        i.senseAvg = (1.0 - alpha) * i.senseAvg + alpha * (sChemo * unit(angChemo))

        # Heading ODE (Euler–Maruyama)
        angMem = vecToAng(i.senseAvg)
        sMem   = float(np.linalg.norm(i.senseAvg))
        theta  = i.chainAngle
        dtheta = 0.0
        if angMem is not None:
            dtheta += kChemo * math.sin(wrapAngle(angMem - theta)) * dt
        i.chainAngle = wrapAngle(theta + dtheta)

        # Propulsion magnitude from memory
        i.chemoCue = sMem

        # Filopodia visualisation
        for angV in i.angleList:
            filopList.append([i.x, i.y, angV * (180 / math.pi) + 270, 'k', lenFilo])

    # Mass–spring step over dt
    massSpringStep(kSpring, f0,
        cellList=cellList, dt=dt,
        cellRad=cellRad, width=width, length=length,
        RLink=RLink, RBreak=RBreak, posNoise=posNoise
    )

    # Axial advection
    scale = (newLen / oldLen)
    for i in cellList:
        i.x *= scale
        i.filPersist += 1

    return filopList
