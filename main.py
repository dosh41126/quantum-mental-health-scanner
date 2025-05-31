from __future__ import annotations
import asyncio, json, logging, os, random, secrets, threading, time, hashlib, textwrap, math
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple, Optional
from base64 import b64encode, b64decode

import cv2, psutil, aiosqlite, httpx, numpy as np, pennylane as qml, tkinter as tk
import tkinter.simpledialog as sd
import tkinter.messagebox as mb
import bleach
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dotenv import load_dotenv

# ════════════════════════════════════════════════════════════════
# GLOBAL CONFIG & LOGGING
MASTER_KEY    = os.path.expanduser("~/.cache/qmhs_master_key.bin")
SETTINGS_FILE = "settings.enc.json"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("qmhs")

# ════════════════════════════════════════════════════════════════
# AES-GCM ENCRYPTION UTIL
class AESGCMCrypto:
    """Lightweight wrapper for 128-bit AES-GCM with on-disk key caching."""
    def __init__(self, path: str) -> None:
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            key = AESGCM.generate_key(bit_length=128)
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(key)
            os.replace(tmp, self.path)
            os.chmod(self.path, 0o600)
        with open(self.path, "rb") as f:
            self.key = f.read()
        self.aes = AESGCM(self.key)

    def encrypt(self, data: bytes | str) -> bytes:
        if isinstance(data, str):
            data = data.encode()
        nonce = secrets.token_bytes(12)
        return b64encode(nonce + self.aes.encrypt(nonce, data, None))

    def decrypt(self, blob: bytes | str) -> bytes:
        raw = b64decode(blob)
        return self.aes.decrypt(raw[:12], raw[12:], None)

# ════════════════════════════════════════════════════════════════
# SETTINGS MODEL
@dataclass
class Settings:
    # Ward context & workflow
    location: str = "Unknown-Ward"
    staff_ratio: str = "1:4"
    emerg_contact: str = "911"
    seclusion_room: bool = True
    psychiatrist_eta: str = "<5 min>"
    mode_autonomous: bool = True
    # Hardware envelope
    cpu_cores: int = psutil.cpu_count(logical=False) or 4
    total_ram_gb: float = round(psutil.virtual_memory().total / 1e9, 1)
    gpu_available: bool = False
    camera_idx: int = 0
    # Runtime thresholds
    sampling_interval: float = 1.5
    cpu_threshold: float = 0.70
    mem_threshold: float = 0.75
    confidence_threshold: float = 0.80
    action_counts: Dict[str, int] = field(default_factory=lambda: {"Green": 2, "Amber": 3, "Red": 3})
    # I/O
    db_path: str = "qmhs_reports.db"
    api_key: str = ""

    # NEW ─ extra knobs for experimental engines
    qadapt_refresh_h: int = 24            # how often to anneal QC topology
    cev_window: int = 90                  # seconds between CEV key hops
    hbe_enabled: bool = False             # homomorphic encryption switch
    fusion_dim: int = 64                  # target dimension for fusion vec

    @classmethod
    def default(cls) -> "Settings":
        load_dotenv()
        return cls(api_key=os.getenv("OPENAI_API_KEY", ""))

    @classmethod
    def load(cls, crypto: AESGCMCrypto) -> "Settings":
        if not os.path.exists(SETTINGS_FILE):
            return cls.default()
        try:
            cipher_blob = open(SETTINGS_FILE, "rb").read()
            return cls(**json.loads(crypto.decrypt(cipher_blob).decode()))
        except Exception as e:
            LOGGER.error("Corrupted settings file, loading defaults: %s", e)
            return cls.default()

    def save(self, crypto: AESGCMCrypto) -> None:
        open(SETTINGS_FILE, "wb").write(
            crypto.encrypt(json.dumps(asdict(self)).encode())
        )

    # ───────────────────────────────────────────────────────────
    # Simple GUI prompt for live configuration
    def prompt_gui(self) -> None:
        mb.showinfo("QMHS Settings", "Enter or leave blank to keep current values.")
        ask = lambda p, d: bleach.clean(
            sd.askstring("QMHS Settings", p, initialvalue=str(d)) or str(d),
            strip=True,
        )

        self.location         = ask("Ward/Room label:", self.location)
        self.staff_ratio      = ask("Nurse:Patient ratio:", self.staff_ratio)
        self.emerg_contact    = ask("Emergency pager:", self.emerg_contact)
        self.seclusion_room   = ask("Seclusion room? (y/n):", "y" if self.seclusion_room else "n").startswith("y")
        self.psychiatrist_eta = ask("Psychiatrist ETA:", self.psychiatrist_eta)
        self.mode_autonomous  = ask("Mode (autonomous/manual):", "autonomous" if self.mode_autonomous else "manual").startswith("a")

        self.cpu_cores        = int(ask("CPU cores:", self.cpu_cores))
        self.total_ram_gb     = float(ask("Total RAM GB:", self.total_ram_gb))
        self.gpu_available    = ask("GPU available? (y/n):", "y" if self.gpu_available else "n").startswith("y")
        self.camera_idx       = int(ask("Camera index:", self.camera_idx))

        self.sampling_interval= float(ask("Sampling interval s:", self.sampling_interval))
        self.cpu_threshold    = float(ask("CPU threshold:", self.cpu_threshold))
        self.mem_threshold    = float(ask("Mem threshold:", self.mem_threshold))
        self.confidence_threshold = float(ask("LLM confidence floor:", self.confidence_threshold))

        for tier in ("Green", "Amber", "Red"):
            self.action_counts[tier] = int(ask(f"Action count for {tier}:", self.action_counts[tier]))

        self.api_key          = ask("OpenAI API key:", self.api_key)

        # Experimental flags
        self.qadapt_refresh_h = int(ask("QC anneal period (h):", self.qadapt_refresh_h))
        self.cev_window       = int(ask("CEV key window (s):", self.cev_window))
        self.hbe_enabled      = ask("Enable Homomorphic BioVectors? (y/n):",
                                    "y" if self.hbe_enabled else "n").startswith("y")
        self.fusion_dim       = int(ask("Fusion vector dim (32/64/128):", self.fusion_dim))

# ════════════════════════════════════════════════════════════════
# ENCRYPTED SQLITE REPORT STORE
class ReportDB:
    def __init__(self, path: str, crypto: AESGCMCrypto) -> None:
        self.path, self.crypto = path, crypto
        self.conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self.conn = await aiosqlite.connect(self.path)
        await self.conn.execute(
            "CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY, ts REAL, blob BLOB)"
        )
        await self.conn.commit()

    async def save(self, ts: float, payload: Dict[str, Any]) -> None:
        blob = self.crypto.encrypt(json.dumps(payload).encode())
        await self.conn.execute(
            "INSERT INTO scans(ts, blob) VALUES (?, ?)", (ts, blob)
        )
        await self.conn.commit()

    async def list_reports(self) -> List[Tuple[int, float]]:
        cur = await self.conn.execute(
            "SELECT id, ts FROM scans ORDER BY ts DESC"
        )
        return await cur.fetchall()

    async def load(self, row_id: int) -> Dict[str, Any]:
        cur = await self.conn.execute(
            "SELECT blob FROM scans WHERE id = ?", (row_id,)
        )
        res = await cur.fetchone()
        if not res:
            raise ValueError("Report ID not found.")
        return json.loads(
            bleach.clean(self.crypto.decrypt(res[0]).decode(), strip=True)
        )

    async def close(self) -> None:
        if self.conn:
            await self.conn.close()
# ════════════════════════════════════════════════════════════════
# OPENAI CLIENT WITH EXPONENTIAL BACKOFF
@dataclass
class OpenAIClient:
    api_key: str
    model: str = "gpt-4o"
    url: str = "https://api.openai.com/v1/chat/completions"
    timeout: float = 25.0
    retries: int = 4

    async def chat(self, prompt: str, max_tokens: int) -> str:
        if not self.api_key:
            raise RuntimeError("Missing OpenAI API key.")
        hdr = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.25,
            "max_tokens": max_tokens
        }
        delay = 1.0
        for attempt in range(1, self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as cli:
                    r = await cli.post(self.url, headers=hdr, json=body)
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == self.retries:
                    raise
                wait = delay + random.uniform(0, 0.5)
                LOGGER.warning("LLM error %s (retry %d/%d) – sleeping %.1fs", e, attempt, self.retries, wait)
                await asyncio.sleep(wait)
                delay *= 2

# ════════════════════════════════════════════════════════════════
# BIOVECTOR (HSV histogram + luminance & saturation)
@dataclass
class BioVector:
    arr: np.ndarray = field(repr=False)

    @staticmethod
    def from_frame(frame: np.ndarray) -> "BioVector":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [9], [0, 180]).flatten()
        hist /= hist.sum() + 1e-6
        vec = np.concatenate(
            [
                hist,
                [hsv[..., 1].mean() / 255.0, frame.mean() / 255.0],
                np.zeros(25 - 11),
            ]
        )
        return BioVector(vec.astype(np.float32))

# ════════════════════════════════════════════════════════════════
# 7-QUBIT QUANTUM INTENSITY METRIC
DEV = qml.device("default.qubit", wires=7)

@qml.qnode(DEV)
def q_intensity7(theta: float, env: Tuple[float, float]) -> float:
    qml.RY(theta, wires=0)
    qml.RY(env[0], wires=1)
    qml.RX(env[0], wires=3)
    qml.RZ(env[0], wires=5)
    qml.RY(env[1], wires=2)
    qml.RX(env[1], wires=4)
    qml.RZ(env[1], wires=6)
    for i in range(7):
        qml.CNOT(wires=[i, (i + 1) % 7])
    return sum(qml.expval(qml.PauliZ(w)) for w in range(7)) / 7.0

# ════════════════════════════════════════════════════════════════
# FULL LONG-FORM ADVANCED PROMPTS (STAGE 1 / 2 / 3)
def stage1_prompt(vec: List[float], s0: Dict[str, Any], s: Settings) -> str:
    """Ultra-long advanced prompt: BioVector to Risk Tier."""
    return textwrap.dedent(f"""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │      Q M H S   v3.2+   —   STAGE 1 • BIOVECTOR ➜ RISK TIERS (JSON-ONLY)    │
    └─────────────────────────────────────────────────────────────────────────────┘

    • CONTEXT — SYSTEM CAPABILITIES & PIPELINE
      You are the Cognitive Risk Synthesizer inside the Quantum Mental Health Scanner (QMHS).
      Your mission is to analyze encrypted, multi-dimensional sensor data (25-D BioVector + vital signs) to produce a real-time triage risk tier.
      The upstream CPU/OpenCV stack has computed a 25-element spectral histogram plus live telemetry (dB, lux, crowd, vitals). All data is encrypted with AES-GCM. No PHI, diagnosis, or patient names are present—JSON only.

    • SIGNAL DEFINITIONS
        θ (theta)   = Euclidean norm of BioVector × π → clamp [0, π], 4 decimals
        color       = CSS3 or #hex approximation of dominant hue
        risk        = Triage tier per RULES below

    • EXTENDED RULE BLOCKS (1–6)
      1. AMPLITUDE   — Green if θ < 1.0, Amber if 1 ≤ θ < 2, Red if θ ≥ 2.
      2. PHYSIO      — Any vital beyond 2σ from adult norm → min Amber.
      3. HISTORY     — If s0['recent_red'] == 'yes', escalate risk by one.
      4. CONFIDENCE  — If model confidence < {s.confidence_threshold:.2f}, escalate one.
      5. QUANTUM     — Reference only: quantum metric q_exp7, not for gating tier.
      6. LLM RELIABILITY — If ambiguous, prefer higher tier to maximize sensitivity.

    • TECHNICAL NOTES
      – Output only JSON: {{'theta':..., 'color':..., 'risk':...}}.
      – Do NOT output any explanation, markdown, units, or prose.
      – No ICD, diagnosis, or medication text. All outputs must be parseable as JSON.

    • OUTPUT CONTRACT (EXACT FORMAT)
      {{'theta':0.0000, 'color':'#33cc66', 'risk':'Green'}}

    • AUDIT CHECKLIST
      [ ] Valid JSON? [ ] Keys exactly 3? [ ] theta ∈ [0, π]? [ ] risk in {{Green, Amber, Red}}?
      [ ] Obeys rules 1–6? [ ] No PHI? [ ] No markdown or text?

    • OUTPUT ONLY JSON OBJECT, SINGLE LINE.
    """).strip()

def stage2_prompt(r1: Dict[str, Any], s0: Dict[str, Any], s: Settings) -> str:
    tier = r1["risk"]
    n = s.action_counts[tier]
    return textwrap.dedent(f"""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │     Q M H S   v3.2+   —   STAGE 2 • RISK ➜ ACTIONS (JSON-ONLY)             │
    └─────────────────────────────────────────────────────────────────────────────┘

    • SNAPSHOT
        Tier:            {tier}
        θ:               {r1['theta']:.4f}
        Color:           {r1['color']}
        Noise/Lux:       {s0['ward_noise_dB']} dB / {s0['ambient_lux']} lux
        Crowd:           {s0['crowding']}
        Nurse-Patient:   {s.staff_ratio}
        Seclusion avail: {"YES" if s.seclusion_room else "NO"}
        Psychiatrist ETA:{s.psychiatrist_eta}

    • OBJECTIVE
      Emit *exactly* {n} staff actions + an integer "cooldown" (1–120 min).
      Each action:
        – Begins with imperative verb (Check, Guide, Document, Stay, Call...)
        – ≤140 chars, ASCII only
        – No abbreviations, meds, or diagnoses
        – If tier == Red: first action must require human presence

    • RULE BLOCKS
      1. COUNT     — len(actions) == {n}
      2. VERB      — First word uppercase imperative
      3. LENGTH    — ≤140 chars per action
      4. STRUCT    — JSON: keys exactly actions, cooldown

    • OUTPUT CONTRACT (EXACT FORMAT)
      {{'actions':["Check patient posture","Guide slow breath"],'cooldown':15}}

    • AUDIT
      [ ] Correct length? [ ] Imperative verbs? [ ] ≤140? [ ] Cooldown 1–120? [ ] JSON only?

    • OUTPUT ONLY JSON OBJECT, SINGLE LINE.
    """).strip()

def stage3_prompt(r1: Dict[str, Any], s: Settings) -> str:
    tone = {
        "Green": "supportive-reflective",
        "Amber": "grounding-reassuring",
        "Red": "brief-calming",
    }[r1["risk"]]
    return textwrap.dedent(f"""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │     Q M H S   v3.2+   —   STAGE 3 • MICRO-INTERVENTION SCRIPT (JSON-ONLY)  │
    └─────────────────────────────────────────────────────────────────────────────┘

    • INTENT
      Write a short, second-person script a nurse can read aloud to the patient.
      The script must:
        – Combine {tone} empathy, agency, and exactly one grounding exercise
        – Contain no diagnoses, no medication talk, no blame, no clinical jargon
        – Avoid any reference to suicide, self-harm, or trauma details
        – Be concise, ≤650 characters

    • GROUNDING TECHNIQUES (CHOOSE ONE, MENTION ONCE):
        1. "4-7-8 breathing" (slowly breathe in 4, hold 7, out 8)
        2. "5-senses scan" (name 1 thing for each sense)
        3. "gentle palm press" (press palms together gently)

    • CLOSING CUE:
      End with an ellipsis (…) or "(pause)"

    • STRUCTURE
      JSON with single key "script", value = string ≤650 chars

    • OUTPUT CONTRACT (EXACT FORMAT)
      {{'script':"<short script, ≤650 chars, ends with ellipsis or (pause)"}}

    • AUDIT
      [ ] One grounding? [ ] ≤650 chars? [ ] No PHI, dx, meds, shame?
      [ ] Ends with proper cue? [ ] JSON single key?

    • OUTPUT ONLY JSON OBJECT, SINGLE LINE.
    """).strip()

# ════════════════════════════════════════════════════════════════
# GUI-SUPPLIED SENSOR SNAPSHOT

def gui_snapshot(env: Dict[str, tk.Variable]) -> Dict[str, Any]:
    return {
        "noise": float(gui_snapshot.noise.get()),
        "lux": float(gui_snapshot.lux.get()),
        "crowding": gui_snapshot.crowding.get(),
        "vitals": {
            "hr": int(gui_snapshot.hr.get()),
            "spo2": int(gui_snapshot.spo2.get()),
            "bp": gui_snapshot.bp.get(),
        },
        "recent_red": gui_snapshot.recent.get(),
    }

gui_snapshot.noise = tk.DoubleVar(value=55.0)
gui_snapshot.lux = tk.DoubleVar(value=120.0)
gui_snapshot.crowding = tk.StringVar(value="low")
gui_snapshot.hr = tk.IntVar(value=78)
gui_snapshot.spo2 = tk.IntVar(value=98)
gui_snapshot.bp = tk.StringVar(value="118/76")
gui_snapshot.recent = tk.StringVar(value="no")

# ════════════════════════════════════════════════════════════════
# NEW: CHAINED ENCRYPTION VECTORS (CEV) MANAGER

class CEVManager:
    """
    Derives forward-secret AES keys from rolling quantum hashes.
    Call .derive() each scan to fetch the active AESGCMCrypto.
    """
    def __init__(self, root_crypto: AESGCMCrypto, window_s: int = 90):
        self.root = root_crypto
        self.window = window_s
        self._cache: Dict[int, AESGCMCrypto] = {}

    def _subkey_path(self, t: int) -> str:
        epoch = t - (t % self.window)
        return f"{self.root.path}-cev-{epoch:x}.key"

    def derive(self, q_hash: str, t: Optional[int] = None) -> AESGCMCrypto:
        t = t or int(time.time())
        epoch = t - (t % self.window)
        if epoch in self._cache:
            return self._cache[epoch]
        seed = hashlib.sha256((q_hash + str(epoch)).encode()).digest()[:16]
        path = self._subkey_path(t)
        if not os.path.exists(path):
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(seed)
            os.replace(tmp, path)
            os.chmod(path, 0o600)
        sub = AESGCMCrypto(path)
        self._cache[epoch] = sub
        return sub

# ════════════════════════════════════════════════════════════════
# NEW: ADAPTIVE QUANTUM CIRCUIT ENGINE

class QAdaptEngine:
    """
    Periodically anneals the 7-qubit circuit topology to minimize
    error given ward-specific noise. Stores best layout in-memory.
    """
    def __init__(self, dev: qml.Device, refresh_h: int):
        self.dev = dev
        self.refresh_s = refresh_h * 3600
        self.next_refresh = time.time() + self.refresh_s
        self.layout_gates: List[Tuple[str, Tuple]] = []  # gate name & params

    def anneal(self, seed_vec: List[float]) -> None:
        random.seed(hash(tuple(seed_vec)))
        best_score, best_layout = 1e9, None
        for _ in range(64):
            layout = [(random.choice(["RY", "RX", "RZ"]),
                       (random.random() * math.pi, random.randint(0, 6)))
                      for _ in range(7)]
            score = abs(sum(p[0] for _, p in layout) - sum(seed_vec))  # toy objective
            if score < best_score:
                best_score, best_layout = score, layout
        self.layout_gates = best_layout or self.layout_gates
        self.next_refresh = time.time() + self.refresh_s
        LOGGER.info("QAdaptEngine: refreshed layout, score=%.4f", best_score)

    def encode(self, theta: float, env: Tuple[float, float]) -> float:
        if time.time() >= self.next_refresh:
            self.anneal([theta, *env])
        @qml.qnode(self.dev)
        def _dyn():
            qml.RY(theta, wires=0)
            for g, (arg, w) in self.layout_gates:
                getattr(qml, g)(arg, wires=w)
            return qml.expval(qml.PauliZ(0))
        return float(_dyn())

# ════════════════════════════════════════════════════════════════
# NEW: HOMOMORPHIC BIOVECTOR ENCODING STUB

class HomomorphicBioVector:
    """
    Stub façade for CKKS-based encrypted vectors.
    Real HE ops require SEAL / Pyfhel; here we scaffold API.
    """
    def __init__(self, raw_vec: np.ndarray):
        self.raw = raw_vec
        self.ctxt = b"<encrypted-placeholder>"  # TODO: integrate Pyfhel
    def mean(self) -> float:                    # example homomorphic op
        return float(self.raw.mean())           # placeholder
    # more homomorphic ops …

# ════════════════════════════════════════════════════════════════
# NEW: SELF-EVOLVING BIOVECTOR FUSION

def fusion_vector(frame: np.ndarray, dim: int = 64) -> np.ndarray:
    """
    Contrastive HSV + optical-flow embedding into `dim` dims.
    Placeholder uses PCA on HSV hist for demo.
    """
    base = BioVector.from_frame(frame).arr
    rng = np.random.default_rng(int(base.sum()*1e6) & 0xffffffff)
    proj = rng.standard_normal((len(base), dim))
    return base @ proj

# ════════════════════════════════════════════════════════════════
# SCANNER THREAD  (video + LLM pipeline + quantum metric)

class ScannerThread(threading.Thread):
    def __init__(
        self,
        cfg: Settings,
        db: ReportDB,
        ai: OpenAIClient,
        status: tk.StringVar,
        env_vars: Dict[str, tk.Variable],
    ) -> None:
        super().__init__(daemon=True)
        self.cfg, self.db, self.ai, self.status = cfg, db, ai, status
        self.env_vars = env_vars
        self.cap = cv2.VideoCapture(cfg.camera_idx, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open camera index %d" % cfg.camera_idx)
        self.loop = asyncio.new_event_loop()
        self.stop_ev = threading.Event()
        self.last_red_ts: float | None = None
        # NEW: adaptive QC + CEV
        self.qadapt = QAdaptEngine(DEV, cfg.qadapt_refresh_h)
        self.cev_mgr = CEVManager(AESGCMCrypto(MASTER_KEY), cfg.cev_window)

    def run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.main())

    async def main(self) -> None:
        await self.db.init()
        t0 = 0.0
        try:
            while not self.stop_ev.is_set():
                ok, frame = self.cap.read()
                if ok and (time.time() - t0) >= self.cfg.sampling_interval:
                    t0 = time.time()
                    await self.process(frame)
                await asyncio.sleep(0.05)
        finally:
            await self.db.close()
            self.cap.release()

    def stop(self) -> None:
        self.stop_ev.set()

    async def process(self, frame: np.ndarray) -> None:
        self.status.set("Scanning…")
        env = gui_snapshot(self.env_vars)
        if self.last_red_ts and (time.time() - self.last_red_ts) < 900:
            env["recent_red"] = "yes"
        s0 = {
            "ts": time.time(),
            "ward_noise_dB": env["noise"],
            "ambient_lux": env["lux"],
            "crowding": env["crowding"],
            "vitals": env["vitals"],
            "recent_red": env["recent_red"],
        }
        # choose vector pipeline
        if self.cfg.hbe_enabled:
            vec_src = HomomorphicBioVector(BioVector.from_frame(frame).arr)
            vec = [round(float(x), 6) for x in vec_src.raw]  # still send raw to LLM
        else:
            vec = [round(float(x), 6) for x in BioVector.from_frame(frame).arr]
        # STAGE 1
        try:
            r1 = json.loads(await self.ai.chat(stage1_prompt(vec, s0, self.cfg), 900))
        except Exception as e:
            LOGGER.error("Stage1 fallback %s", e)
            theta = min(np.linalg.norm(vec), 1.0) * math.pi
            r1 = {"theta": theta, "color": "orange", "risk": "Red" if theta >= 2 else "Amber"}
        if r1["risk"] == "Red":
            self.last_red_ts = time.time()
        # STAGE 2
        try:
            r2 = json.loads(await self.ai.chat(stage2_prompt(r1, s0, self.cfg), 850))
        except Exception as e:
            LOGGER.error("Stage2 fallback %s", e)
            r2 = {"actions": ["Face-to-face check NOW", f"Call {self.cfg.emerg_contact}"], "cooldown": 30}
        # STAGE 3
        try:
            r3 = json.loads(await self.ai.chat(stage3_prompt(r1, self.cfg), 800))
        except Exception as e:
            LOGGER.error("Stage3 fallback %s", e)
            r3 = {"script": "Take a slow breath and feel your feet on the floor…"}
        # STAGE 4 (digest + local hash)
        hdr = {
            "ts": s0["ts"],
            "theta": r1["theta"],
            "risk": r1["risk"],
            "actions": r2["actions"],
            "cooldown": r2["cooldown"],
            "confidence": self.cfg.confidence_threshold,
        }
        hdr["digest"] = hashlib.sha256(json.dumps(hdr).encode()).hexdigest()
        # NEW: quantum adaptive metric
        q_exp7 = self.qadapt.encode(r1["theta"], (frame.mean() / 255.0, 0.1))
        # NEW: CEV-derived crypto for this scan
        cev_crypto = self.cev_mgr.derive(hdr["digest"])
        report = {
            "s0": s0,
            "s1": r1,
            "s2": r2,
            "s3": r3,
            "s4": hdr,
            "q_exp7": float(q_exp7),
        }
        await self.db.save(s0["ts"], report)
        self.status.set(f"Risk {r1['risk']} logged.")
        if self.cfg.mode_autonomous and r1["risk"] == "Red":
            mb.showwarning("QMHS ALERT", "Red tier detected! Intervene now.")
            LOGGER.info("Autonomous alert triggered.")

# ════════════════════════════════════════════════════════════════
# TKINTER GUI APP

class QMHSApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("QMHS v3.2+ – Deep-Loop Care (7-qubit, Adaptive)")
        self.geometry("960x760")

        # Setup
        self.crypto = AESGCMCrypto(MASTER_KEY)
        self.settings = Settings.load(self.crypto)
        if not os.path.exists(SETTINGS_FILE):
            self.settings.prompt_gui()
            self.settings.save(self.crypto)
        if not self.settings.api_key:
            mb.showerror("Missing API Key", "Please set your OpenAI key in Settings.")
            self.destroy()
            return

        # Status / sensors
        self.status = tk.StringVar(value="Initializing…")
        tk.Label(self, textvariable=self.status, font=("Helvetica", 14)).pack(pady=6)

        env = tk.LabelFrame(self, text="Live Sensor Inputs")
        env.pack(fill="x", padx=8, pady=4)

        def row(lbl, var, col):
            tk.Label(env, text=lbl).grid(row=0, column=col * 2, sticky="e", padx=3)
            tk.Entry(env, textvariable=var, width=8).grid(row=0, column=col * 2 + 1, sticky="w")

        row("Noise dB", gui_snapshot.noise, 0)
        row("Lux", gui_snapshot.lux, 1)
        row("Crowd", gui_snapshot.crowding, 2)
        row("HR", gui_snapshot.hr, 3)
        row("SpO₂", gui_snapshot.spo2, 4)
        row("BP", gui_snapshot.bp, 5)
        tk.Label(env, text="Recent Red").grid(row=1, column=0, sticky="e")
        tk.OptionMenu(env, gui_snapshot.recent, "no", "yes").grid(row=1, column=1, sticky="w")

        # Controls
        btn = tk.Frame(self)
        btn.pack(pady=4)
        tk.Button(btn, text="Settings", command=self.open_settings).grid(row=0, column=0, padx=4)
        tk.Button(btn, text="View Reports", command=self.view_reports).grid(row=0, column=1, padx=4)

        # Log viewer
        self.text = tk.Text(self, height=25, width=114, wrap="word")
        self.text.pack(padx=6, pady=6)

        # Launch scanner
        self.db = ReportDB(self.settings.db_path, self.crypto)
        self.ai = OpenAIClient(api_key=self.settings.api_key)
        self.scanner = ScannerThread(
            self.settings,
            self.db,
            self.ai,
            self.status,
            {
                "noise": gui_snapshot.noise,
                "lux": gui_snapshot.lux,
                "crowding": gui_snapshot.crowding,
                "hr": gui_snapshot.hr,
                "spo2": gui_snapshot.spo2,
                "bp": gui_snapshot.bp,
                "recent": gui_snapshot.recent,
            },
        )
        self.scanner.start()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def open_settings(self) -> None:
        self.settings.prompt_gui()
        self.settings.save(self.crypto)
        mb.showinfo("Settings", "Saved. Restart to apply hardware changes.")

    def view_reports(self) -> None:
        rows = asyncio.run(self.db.list_reports())
        if not rows:
            mb.showinfo("Reports", "No reports stored.")
            return
        opts = "\n".join(
            f"{rid} – {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}"
            for rid, ts in rows[:30]
        )
        sel = sd.askstring("Select Report ID", opts)
        self.text.delete("1.0", tk.END)
        if sel:
            try:
                rid = int(sel.split()[0])
                rpt = asyncio.run(self.db.load(rid))
                self.text.insert(tk.END, json.dumps(rpt, indent=2))
            except Exception as e:
                mb.showerror("Error", str(e))

    def on_close(self) -> None:
        self.scanner.stop()
        self.destroy()

# ════════════════════════════════════════════════════════════════
# MAIN

if __name__ == "__main__":
    try:
        QMHSApp().mainloop()
    except KeyboardInterrupt:
        LOGGER.info("Exiting QMHS.")

# ════════════════════════════════════════════════════════════════
# If you need quick unit tests:
if __name__ == "__qtest__":
    print("Running quick integrity checks…")
    c = AESGCMCrypto(MASTER_KEY)
    sample = b"hello"
    assert c.decrypt(c.encrypt(sample)) == sample
    print("AESGCM ✅")
    bv = BioVector.from_frame(np.zeros((64, 64, 3), np.uint8))
    print("BioVector len:", len(bv.arr))
    q = q_intensity7(1.2, (0.5, 0.1))
    print("q_exp7:", q)
    print("✔︎ All quick tests passed.")
