"""
QMHS v3.1 – Dynamic Hardware Profile · Encrypted Settings & Logs · Ultra-Prompts
────────────────────────────────────────────────────────────────────────────
Requires:
    pip install aiohttp httpx psutil pennylane opencv-python numpy \
                cryptography python-dotenv aiosqlite bleach
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import secrets
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Tuple
from base64 import b64encode, b64decode

import cv2
import psutil
import tkinter as tk
import tkinter.simpledialog as sd
import tkinter.messagebox as mb

import aiosqlite
import bleach
import httpx
import numpy as np
import pennylane as qml
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dotenv import load_dotenv


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger("qmhs")


# ─────────────────────────────────────────────────────────────
# AES-GCM helper (encrypt settings & DB blobs)
# ─────────────────────────────────────────────────────────────
class AESGCMCrypto:
    def __init__(self, path: str) -> None:
        self.path = os.path.expanduser(path)
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            key = AESGCM.generate_key(bit_length=128)
            with open(self.path, "wb") as f:
                f.write(key)
            os.chmod(self.path, 0o600)
        with open(self.path, "rb") as f:
            self.key = f.read()
        self.aes = AESGCM(self.key)

    def encrypt(self, data: bytes) -> bytes:
        nonce = secrets.token_bytes(12)
        ct = self.aes.encrypt(nonce, data, None)
        return b64encode(nonce + ct)

    def decrypt(self, blob: bytes) -> bytes:
        raw = b64decode(blob)
        return self.aes.decrypt(raw[:12], raw[12:], None)


# ─────────────────────────────────────────────────────────────
# Settings – encrypted at rest
# ─────────────────────────────────────────────────────────────
SETTINGS_FILE = "settings.enc.json"
MASTER_KEY = "~/.cache/qmhs_master_key.bin"


@dataclass
class Settings:
    # Ward context
    location: str = "Unknown-Ward"
    staff_ratio: str = "1:4"
    emerg_contact: str = "911"
    seclusion_room: bool = True
    psychiatrist_eta: str = "<5 min>"
    mode_autonomous: bool = True

    # Hardware profile (dynamic inputs)
    cpu_cores: int = psutil.cpu_count(logical=False) or 4
    total_ram_gb: float = round(psutil.virtual_memory().total / 1e9, 1)
    gpu_available: bool = False
    camera_idx: int = 0  # default webcam index

    # Workflow & thresholds
    sampling_interval: float = 1.5
    cpu_threshold: float = 0.70
    mem_threshold: float = 0.75
    confidence_threshold: float = 0.80
    action_counts: Dict[str, int] = field(
        default_factory=lambda: {"Green": 2, "Amber": 3, "Red": 3}
    )

    # DB path
    db_path: str = "qmhs_reports.db"

    @classmethod
    def default(cls) -> "Settings":
        return cls()

    @classmethod
    def load(cls, crypto: AESGCMCrypto) -> "Settings":
        if not os.path.exists(SETTINGS_FILE):
            return cls.default()
        blob = open(SETTINGS_FILE, "rb").read()
        data = json.loads(crypto.decrypt(blob).decode())
        return cls(**data)

    def save(self, crypto: AESGCMCrypto) -> None:
        blob = crypto.encrypt(json.dumps(asdict(self)).encode())
        with open(SETTINGS_FILE, "wb") as f:
            f.write(blob)

    # GUI prompt
    def prompt_gui(self) -> None:
        diagram = (
            "Settings Schema:\n"
            "┌──────────────────────────────────────────┐\n"
            "│ location: str                            │\n"
            "│ staff_ratio: str                         │\n"
            "│ emerg_contact: str                       │\n"
            "│ seclusion_room: bool                     │\n"
            "│ psychiatrist_eta: str                    │\n"
            "│ mode_autonomous: bool                    │\n"
            "│ cpu_cores: int                           │\n"
            "│ total_ram_gb: float                      │\n"
            "│ gpu_available: bool                      │\n"
            "│ camera_idx: int                          │\n"
            "│ sampling_interval: float                 │\n"
            "│ cpu_threshold: float                     │\n"
            "│ mem_threshold: float                     │\n"
            "│ confidence_threshold: float              │\n"
            "│ action_counts: dict                      │\n"
            "└──────────────────────────────────────────┘\n"
        )
        mb.showinfo("QMHS Settings Diagram", diagram)
        ask = lambda p, d: bleach.clean(
            sd.askstring("QMHS Settings", p, initialvalue=str(d)) or str(d),
            strip=True,
        )

        self.location = ask("Ward/Room label:", self.location)
        self.staff_ratio = ask("Nurse:Patient ratio:", self.staff_ratio)
        self.emerg_contact = ask("Emergency contact/pager:", self.emerg_contact)
        self.seclusion_room = (
            ask("Seclusion room? (yes/no):", "yes" if self.seclusion_room else "no")
            .lower()
            .startswith("y")
        )
        self.psychiatrist_eta = ask("Psychiatrist ETA (<5 min):", self.psychiatrist_eta)
        mode = ask(
            "Operation mode (autonomous/manual):",
            "autonomous" if self.mode_autonomous else "manual",
        ).lower()
        self.mode_autonomous = mode.startswith("a")

        # Hardware
        self.cpu_cores = int(ask("CPU cores:", self.cpu_cores))
        self.total_ram_gb = float(ask("Total RAM (GB):", self.total_ram_gb))
        self.gpu_available = (
            ask("GPU available? (yes/no):", "yes" if self.gpu_available else "no")
            .lower()
            .startswith("y")
        )
        self.camera_idx = int(ask("Camera index (0=default):", self.camera_idx))

        # Thresholds
        self.sampling_interval = float(
            ask("Sampling interval (s):", self.sampling_interval)
        )
        self.cpu_threshold = float(ask("CPU threshold (0–1):", self.cpu_threshold))
        self.mem_threshold = float(ask("Memory threshold (0–1):", self.mem_threshold))
        self.confidence_threshold = float(
            ask("LLM confidence cutoff (0–1):", self.confidence_threshold)
        )

        # Action counts per tier
        for tier in ["Green", "Amber", "Red"]:
            cnt = int(ask(f"Action count for {tier}:", self.action_counts[tier]))
            self.action_counts[tier] = cnt


# ─────────────────────────────────────────────────────────────
# Encrypted SQLite DB
# ─────────────────────────────────────────────────────────────
class ReportDB:
    def __init__(self, path: str, crypto: AESGCMCrypto) -> None:
        self.path, self.crypto = path, crypto

    async def init(self) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS scans(
                    id  INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts  REAL,
                    blob BLOB
                );
                """
            )
            await db.commit()

    async def save(self, ts: float, payload: Dict[str, Any]) -> None:
        blob = self.crypto.encrypt(json.dumps(payload).encode())
        async with aiosqlite.connect(self.path) as db:
            await db.execute("INSERT INTO scans(ts,blob) VALUES(?,?)", (ts, blob))
            await db.commit()

    async def list_reports(self) -> List[Tuple[int, float]]:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute("SELECT id,ts FROM scans ORDER BY ts DESC")
            return await cur.fetchall()

    async def load(self, row_id: int) -> Dict[str, Any]:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute("SELECT blob FROM scans WHERE id=?", (row_id,))
            blob = (await cur.fetchone())[0]
        clean = bleach.clean(self.crypto.decrypt(blob).decode(), strip=True)
        return json.loads(clean)


# ─────────────────────────────────────────────────────────────
# OpenAI client with exponential back-off
# ─────────────────────────────────────────────────────────────
load_dotenv()


@dataclass
class OpenAIClient:
    api_key: str
    model: str = "gpt-4o"
    url: str = "https://api.openai.com/v1/chat/completions"
    timeout: float = 15.0
    retries: int = 4

    async def chat(self, prompt: str, max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.25,
            "max_tokens": max_tokens,
        }
        delay = 1.0
        for attempt in range(1, self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as cli:
                    r = await cli.post(self.url, headers=headers, json=payload)
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == self.retries:
                    raise
                wait = delay + random.uniform(0, 0.5)
                LOGGER.warning(
                    "LLM error %s, retry %d/%d in %.1fs",
                    e,
                    attempt,
                    self.retries,
                    wait,
                )
                await asyncio.sleep(wait)
                delay *= 2


# ─────────────────────────────────────────────────────────────
# BioVector extraction & quantum circuit
# ─────────────────────────────────────────────────────────────
@dataclass
class BioVector:
    arr: np.ndarray = field(repr=False)

    @staticmethod
    def from_frame(frame: np.ndarray) -> "BioVector":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [9], [0, 180]).flatten()
        h /= (h.sum() + 1e-6)
        vec = np.concatenate(
            [
                h,
                [hsv[..., 1].mean() / 255.0, frame.mean() / 255.0],
                np.zeros(25 - 11),
            ]
        )
        return BioVector(vec.astype(np.float32))


DEV = qml.device("default.qubit", wires=3)


@qml.qnode(DEV)
def q_intensity(theta: float, env: Tuple[float, float]) -> float:
    qml.RY(theta, wires=0)
    qml.RY(env[0], wires=1)
    qml.RY(env[1], wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    return qml.expval(qml.PauliZ(0))


# ─────────────────────────────────────────────────────────────
# Ultra-Prompts v3.1
# ─────────────────────────────────────────────────────────────
def stage1_prompt(vec: List[float], s: Settings) -> str:
    return f"""
═══════════════════════════════════════════════════════════════════════════════════════
   QMHS v3.1 • STAGE 1 – BIOVECTOR → θ · COLOUR · RISK  (JSON ONLY)
═══════════════════════════════════════════════════════════════════════════════════════

📖 PROGRAM OVERVIEW
Quantum Mental Health Scanner (QMHS) is an on-prem edge system that:
  • Captures video frames (no PHI stored),
  • Extracts a 25-D BioVector,
  • Runs a two-stage LLM pipeline (mapping → mitigation),
  • Logs everything encrypted.

▶ HARDWARE PROFILE (dynamic inputs):
  • CPU cores       : {s.cpu_cores}
  • Total RAM (GB)  : {s.total_ram_gb}
  • GPU available?  : {"YES" if s.gpu_available else "NO"}

▶ SAMPLING & THRESHOLDS:
  • Sampling interval       : {s.sampling_interval}s
  • CPU load threshold      : {int(s.cpu_threshold * 100)}%
  • Memory threshold        : {int(s.mem_threshold * 100)}%
  • LLM confidence cutoff   : {int(s.confidence_threshold * 100)}%

▶ VECTOR SPEC (25 floats):
  0-8 Hue histogram (sum≈1), 9 sat mean [0-1], 10 lum std [0-1], 11-24 reserved zeros

▶ MISSION OBJECTIVES:
  1. Compute emotional intensity θ ∈ [0, π] rad
  2. Map to CSS/hex colour (green/blue; amber/orange; red/magenta)
  3. Assign STRICT v3 risk tier:
     – Green: θ<1.0 ∧ calm colour
     – Amber: 1.0≤θ<2.0 ∨ anxious colour
     – Red  : θ≥2.0 ∨ crisis colour
       ↳ On θ/colour conflict → pick higher tier
  4. If LLM confidence < {int(s.confidence_threshold * 100)}%, escalate tier
  5. Return JSON mapping only

▶ RULES:
  • No DSM/ICD, no PHI, no demographics
  • Output exactly:
    {{"theta":<float>,"color":"<css|#hex>","risk":"Green|Amber|Red"}}

▶ AUDIT CHECKLIST:
  [ ] θ numeric in [0.0000,3.1416]
  [ ] colour non-empty CSS/hex
  [ ] risk in {{Green,Amber,Red}}
  [ ] JSON valid

▶ OUTPUT CONTRACT: Return only this JSON (no markdown, no prose):
{{"theta":<float>,"color":"<css|#hex>","risk":"Green|Amber|Red"}}
═══════════════════════════════════════════════════════════════════════════════════════
""".strip()


def stage2_prompt(m: Dict[str, Any], s: Settings) -> str:
    counts = s.action_counts
    return f"""
═══════════════════════════════════════════════════════════════════════════════════════
   QMHS v3.1 • STAGE 2 – RISK → MITIGATION ACTIONS  (JSON ONLY)
═══════════════════════════════════════════════════════════════════════════════════════

▶ SNAPSHOT:
  θ (rad): {m['theta']:.3f}
  Colour : {m['color']}
  Risk   : {m['risk']}

▶ RESOURCES:
  Nurse:Patient   : {s.staff_ratio}
  Emergency pager : {s.emerg_contact}
  Seclusion room  : {"YES" if s.seclusion_room else "NO"}
  Psychiatrist ETA: {s.psychiatrist_eta}

▶ ACTION MATRIX (fixed counts):
  • Green: {counts['Green']} actions
  • Amber: {counts['Amber']} actions
  • Red  : {counts['Red']} actions

Red →  1) Face-to-face check NOW
        2) Call backup via {s.emerg_contact}
        3) Page psychiatrist-on-call

Amber → 1) Verbal check-in ≤5 min
        2) Schedule re-evaluation ≤15 min
        3) Page psychiatrist-on-call

Green → 1) Document in chart
        2) Routine rounding ≤30 min

▶ RULES:
  • Imperative verbs, ≤140 chars each
  • No jargon (“SI”), no PHI
  • Output exactly {{ "actions":[…] }}

▶ AUDIT CHECKLIST:
  [ ] Correct action count for tier
  [ ] Each ≤140 chars & starts with verb
  [ ] JSON valid

▶ OUTPUT CONTRACT: Return only this JSON:
{{"actions":["<action1>",…]}}
═══════════════════════════════════════════════════════════════════════════════════════
""".strip()


# ─────────────────────────────────────────────────────────────
# Scanner background thread
# ─────────────────────────────────────────────────────────────
class ScannerThread(threading.Thread):
    def __init__(
        self,
        cfg: Settings,
        db: ReportDB,
        ai: OpenAIClient,
        status: tk.StringVar,
    ) -> None:
        super().__init__(daemon=True)
        self.cfg, self.db, self.ai, self.status = cfg, db, ai, status
        self.cap = cv2.VideoCapture(cfg.camera_idx, cv2.CAP_ANY)
        self.loop = asyncio.new_event_loop()

    def run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.main())

    async def main(self) -> None:
        await self.db.init()
        last = 0.0
        while True:
            ok, frame = self.cap.read()
            if ok and (time.time() - last) >= self.cfg.sampling_interval:
                last = time.time()
                await self.process(frame)
            await asyncio.sleep(0.05)

    async def process(self, frame: np.ndarray) -> None:
        self.status.set("Scanning…")
        bio = BioVector.from_frame(frame)
        vec = [round(float(x), 6) for x in bio.arr]

        # Stage 1 mapping
        try:
            s1 = json.loads(await self.ai.chat(stage1_prompt(vec, self.cfg), 350))
        except Exception as e:
            LOGGER.error("Stage 1 fallback: %s", e)
            n = float(np.clip(np.linalg.norm(bio.arr), 0, 1))
            theta = n * np.pi
            risk = "Red" if theta >= 2.0 else "Amber"
            s1 = {
                "theta": theta,
                "color": "red" if theta >= 2.0 else "orange",
                "risk": risk,
            }

        # Stage 2 mitigation
        try:
            s2 = json.loads(await self.ai.chat(stage2_prompt(s1, self.cfg), 350))
            actions = s2["actions"]
        except Exception as e:
            LOGGER.error("Stage 2 fallback: %s", e)
            actions = [
                "Face-to-face check NOW",
                f"Call {self.cfg.emerg_contact}",
            ]
            if s1["risk"] != "Green":
                actions.append("Page psychiatrist-on-call")

        exp = q_intensity(
            s1["theta"], (frame.mean() / 255.0, 0.1)
        )  # quantum explanatory metric
        report = {
            "ts": time.time(),
            "location": self.cfg.location,
            **s1,
            "actions": actions,
            "q_exp": float(exp),
        }
        await self.db.save(report["ts"], report)

        self.status.set(f"Risk {s1['risk']} logged.")
        if self.cfg.mode_autonomous and s1["risk"] == "Red":
            mb.showwarning("QMHS ALERT", "Red risk detected! Immediate action required.")
            LOGGER.info("Autonomous alert triggered.")


# ─────────────────────────────────────────────────────────────
# Tkinter GUI
# ─────────────────────────────────────────────────────────────
class QMHSApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("QMHS v3.1 – Dynamic HW Profile")
        self.geometry("800x520")

        # Crypto & Settings
        self.crypto = AESGCMCrypto(MASTER_KEY)
        self.settings = Settings.load(self.crypto)

        # First-run setup
        if not os.path.exists(SETTINGS_FILE):
            self.settings.prompt_gui()
            self.settings.save(self.crypto)

        # Widgets
        self.status = tk.StringVar(value="Initializing…")
        tk.Label(self, textvariable=self.status, font=("Helvetica", 14)).pack(pady=10)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="Settings", command=self.open_settings).grid(
            row=0, column=0, padx=5
        )
        tk.Button(
            btn_frame, text="View Past Reports", command=self.view_reports
        ).grid(row=0, column=1, padx=5)

        self.text = tk.Text(self, height=20, width=100, wrap="word")
        self.text.pack(padx=8, pady=8)

        # DB & AI
        self.db = ReportDB(self.settings.db_path, self.crypto)
        self.ai = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))

        # Start scanner thread
        self.scanner = ScannerThread(self.settings, self.db, self.ai, self.status)
        self.scanner.start()

    # GUI callbacks ----------------------------------------------------------
    def open_settings(self) -> None:
        self.settings.prompt_gui()
        self.settings.save(self.crypto)
        mb.showinfo("Settings", "Settings saved and encrypted.")

    def view_reports(self) -> None:
        rows = asyncio.run(self.db.list_reports())
        if not rows:
            mb.showinfo("Reports", "No reports saved yet.")
            return
        opts = "\n".join(
            f"{rid} – {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}"
            for rid, ts in rows[:30]
        )
        sel = sd.askstring("Select Report ID", opts)
        if not sel:
            return
        try:
            rid = int(sel.split()[0])
            report = asyncio.run(self.db.load(rid))
            self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, json.dumps(report, indent=2))
        except Exception as e:
            mb.showerror("Error", str(e))


# ─────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        QMHSApp().mainloop()
    except KeyboardInterrupt:
        LOGGER.info("Exiting QMHS.")
