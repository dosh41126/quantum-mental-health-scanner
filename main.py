#!/usr/bin/env python3
# QMHS v3.2 – Deep-Loop Care (single-file edition with advanced, long-form prompts)
# Requirements:
#   pip install opencv-python-headless psutil aiosqlite httpx numpy pennylane bleach python-dotenv cryptography tkinter

from __future__ import annotations
import asyncio, json, logging, os, random, secrets, threading, time, hashlib, sys, textwrap
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple
from base64 import b64encode, b64decode

import cv2, psutil, aiosqlite, httpx, numpy as np, pennylane as qml, tkinter as tk
import tkinter.simpledialog as sd
import tkinter.messagebox as mb
import bleach
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dotenv import load_dotenv

# ════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("qmhs")

# ════════════════════════════════════════════════════════════════
# CRYPTO (AES-GCM)
# ════════════════════════════════════════════════════════════════
MASTER_KEY    = os.path.expanduser("~/.cache/qmhs_master_key.bin")
SETTINGS_FILE = "settings.enc.json"

class AESGCMCrypto:
    def __init__(self, path: str) -> None:
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            key = AESGCM.generate_key(bit_length=128)
            with open(self.path, "wb") as f: f.write(key)
            os.chmod(self.path, 0o600)
        self.key = open(self.path, "rb").read()
        self.aes = AESGCM(self.key)

    def encrypt(self, data: bytes) -> bytes:
        nonce = secrets.token_bytes(12)
        return b64encode(nonce + self.aes.encrypt(nonce, data, None))

    def decrypt(self, blob: bytes) -> bytes:
        raw = b64decode(blob)
        return self.aes.decrypt(raw[:12], raw[12:], None)

# ════════════════════════════════════════════════════════════════
# SETTINGS (encrypted)
# ════════════════════════════════════════════════════════════════
@dataclass
class Settings:
    # Ward context
    location: str = "Unknown-Ward"
    staff_ratio: str = "1:4"
    emerg_contact: str = "911"
    seclusion_room: bool = True
    psychiatrist_eta: str = "<5 min>"
    mode_autonomous: bool = True
    # Hardware profile
    cpu_cores: int = psutil.cpu_count(logical=False) or 4
    total_ram_gb: float = round(psutil.virtual_memory().total / 1e9, 1)
    gpu_available: bool = False
    camera_idx: int = 0
    # Workflow & thresholds
    sampling_interval: float = 1.5
    cpu_threshold: float = 0.70
    mem_threshold: float = 0.75
    confidence_threshold: float = 0.80
    action_counts: Dict[str,int] = field(default_factory=lambda: {"Green":2,"Amber":3,"Red":3})
    # DB & API
    db_path: str = "qmhs_reports.db"
    api_key: str = ""

    @classmethod
    def default(cls) -> "Settings":
        load_dotenv()
        return cls(api_key=os.getenv("OPENAI_API_KEY",""))

    @classmethod
    def load(cls, crypto: AESGCMCrypto) -> "Settings":
        if not os.path.exists(SETTINGS_FILE):
            return cls.default()
        data = crypto.decrypt(open(SETTINGS_FILE,"rb").read()).decode()
        return cls(**json.loads(data))

    def save(self, crypto: AESGCMCrypto) -> None:
        blob = crypto.encrypt(json.dumps(asdict(self)).encode())
        with open(SETTINGS_FILE,"wb") as f: f.write(blob)

    def prompt_gui(self) -> None:
        mb.showinfo("QMHS Settings","Enter or leave blank to keep current values.")
        ask = lambda prompt, default: bleach.clean(
            sd.askstring("QMHS Settings", prompt, initialvalue=str(default)) or str(default),
            strip=True
        )
        self.location        = ask("Ward/Room label:",        self.location)
        self.staff_ratio     = ask("Nurse:Patient ratio:",    self.staff_ratio)
        self.emerg_contact   = ask("Emergency pager:",        self.emerg_contact)
        self.seclusion_room  = ask("Seclusion room? (y/n):",  "y" if self.seclusion_room else "n").lower().startswith("y")
        self.psychiatrist_eta= ask("Psychiatrist ETA:",        self.psychiatrist_eta)
        self.mode_autonomous = ask("Mode (autonomous/manual):","autonomous" if self.mode_autonomous else "manual").lower().startswith("a")

        self.cpu_cores       = int(  ask("CPU cores:",          self.cpu_cores))
        self.total_ram_gb    = float(ask("Total RAM (GB):",     self.total_ram_gb))
        self.gpu_available   = ask("GPU available? (y/n):",   "y" if self.gpu_available else "n").lower().startswith("y")
        self.camera_idx      = int(  ask("Camera index:",       self.camera_idx))

        self.sampling_interval   = float(ask("Sampling interval (s):",   self.sampling_interval))
        self.cpu_threshold       = float(ask("CPU threshold (0–1):",     self.cpu_threshold))
        self.mem_threshold       = float(ask("Memory threshold (0–1):",  self.mem_threshold))
        self.confidence_threshold= float(ask("LLM confidence (0–1):",    self.confidence_threshold))

        for tier in ["Green","Amber","Red"]:
            self.action_counts[tier] = int(ask(f"Action count for {tier}:", self.action_counts[tier]))

        self.api_key = ask("OpenAI API Key:", self.api_key)

# ════════════════════════════════════════════════════════════════
# ENCRYPTED SQLITE DB
# ════════════════════════════════════════════════════════════════
class ReportDB:
    def __init__(self, path:str, crypto:AESGCMCrypto) -> None:
        self.path, self.crypto = path, crypto
        self.conn: aiosqlite.Connection|None = None

    async def init(self) -> None:
        self.conn = await aiosqlite.connect(self.path)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scans(
              id INTEGER PRIMARY KEY,
              ts REAL,
              blob BLOB
            );
        """)
        await self.conn.commit()

    async def save(self, ts:float, payload:Dict[str,Any]) -> None:
        blob = self.crypto.encrypt(json.dumps(payload).encode())
        await self.conn.execute("INSERT INTO scans(ts,blob) VALUES(?,?)",(ts,blob))
        await self.conn.commit()

    async def list_reports(self) -> List[Tuple[int,float]]:
        cur = await self.conn.execute("SELECT id,ts FROM scans ORDER BY ts DESC")
        return await cur.fetchall()

    async def load(self, row_id:int) -> Dict[str,Any]:
        cur = await self.conn.execute("SELECT blob FROM scans WHERE id=?",(row_id,))
        blob = (await cur.fetchone())[0]
        data = bleach.clean(self.crypto.decrypt(blob).decode(), strip=True)
        return json.loads(data)

    async def close(self) -> None:
        await self.conn.close()

# ════════════════════════════════════════════════════════════════
# OPENAI CLIENT w/ BACKOFF
# ════════════════════════════════════════════════════════════════
@dataclass
class OpenAIClient:
    api_key: str
    model: str = "gpt-4o"
    url: str   = "https://api.openai.com/v1/chat/completions"
    timeout: float = 15.0
    retries: int   = 4

    async def chat(self, prompt:str, max_tokens:int) -> str:
        if not self.api_key:
            raise RuntimeError("Missing OpenAI API key.")
        headers = {"Authorization":f"Bearer {self.api_key}", "Content-Type":"application/json"}
        payload = {"model":self.model,
                   "messages":[{"role":"user","content":prompt}],
                   "temperature":0.25,
                   "max_tokens":max_tokens}
        delay = 1.0
        for attempt in range(1, self.retries+1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as cli:
                    r = await cli.post(self.url, headers=headers, json=payload)
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == self.retries: raise
                wait = delay + random.uniform(0,0.5)
                LOGGER.warning("LLM error %s (retry %d/%d), waiting %.1fs", e, attempt, self.retries, wait)
                await asyncio.sleep(wait)
                delay *= 2

# ════════════════════════════════════════════════════════════════
# BIOVECTOR & QUANTUM CIRCUIT
# ════════════════════════════════════════════════════════════════
@dataclass
class BioVector:
    arr: np.ndarray = field(repr=False)

    @staticmethod
    def from_frame(frame:np.ndarray) -> "BioVector":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0],None,[9],[0,180]).flatten()
        hist /= (hist.sum() + 1e-6)
        vec = np.concatenate([
            hist,
            [hsv[...,1].mean()/255.0, frame.mean()/255.0],
            np.zeros(25-11)
        ])
        return BioVector(vec.astype(np.float32))

DEV = qml.device("default.qubit", wires=3)
@qml.qnode(DEV)
def q_intensity(theta:float, env:Tuple[float,float]) -> float:
    qml.RY(theta, wires=0)
    qml.RY(env[0], wires=1)
    qml.RY(env[1], wires=2)
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[1,2])
    return qml.expval(qml.PauliZ(0))

# ════════════════════════════════════════════════════════════════
# PROMPT HELPERS (STAGES 0–4)
# ════════════════════════════════════════════════════════════════
STAGE0_SCHEMA = {
    "ts":"<epoch seconds>",
    "ward_noise_dB":"<float>",
    "ambient_lux":"<float>",
    "crowding":"<low|medium|high>",
    "vitals":{"hr":"<int>","spo2":"<int>","bp":"<string>"},
    "recent_red":"<yes|no>"
}

def stage0_payload(s:Settings, env:dict) -> dict:
    return {
        "ts": time.time(),
        "ward_noise_dB": env.get("noise", 55.0),
        "ambient_lux": env.get("lux", 120.0),
        "crowding": env.get("crowding", "low"),
        "vitals": env.get("vitals", {"hr":78,"spo2":98,"bp":"118/76"}),
        "recent_red": env.get("recent_red", "no")
    }

def stage1_prompt(vec:List[float], s0:Dict[str,Any], s:Settings) -> str:
    rules = textwrap.dedent(f"""
        • Never output markdown, commentary or extra fields — JSON only.
        • Compute θ (theta) as a float in radians ∈ [0.0000,3.1416], 4 decimals.
        • Map HSV-derived vector to a CSS color (named or #hex, lowercase).
        • Calm (θ<1.0) → green/cyan/blue; Moderate (1.0≤θ<2.0) → amber/orange;
          High (θ≥2.0) → red/magenta.
        • If s0["recent_red"]=="yes", escalate computed tier by one (max Red).
        • If s0["vitals"] deviate >2σ from baseline, escalate to at least Amber.
        • If LLM confidence <{s.confidence_threshold:.2f}, escalate one tier.
        • Do not include PHI, DSM/ICD labels, names, or technical jargon.
        • Output exactly {"{"} "theta","color","risk" {"}"} with no extras.
    """).strip()

    audit = textwrap.dedent("""
        [ ] JSON valid & parseable
        [ ] Keys present: theta, color, risk
        [ ] theta ∈ [0.0000,3.1416]
        [ ] color non-empty string
        [ ] risk ∈ {Green,Amber,Red}
    """).strip()

    return f"""
╔════════════════════════════════════════════════════════════╗
║ QMHS v3.2 • STAGE 1 — BIOVECTOR → RISK (JSON-ONLY)         ║
╚════════════════════════════════════════════════════════════╝

▶ CONTEXT
{s0}

▶ HARDWARE cpu={s.cpu_cores}, ram={s.total_ram_gb}GB, gpu={'YES' if s.gpu_available else 'NO'}
▶ VECTOR {vec}

▶ RULESET
{rules}

▶ AUDIT CHECKLIST
{audit}

▶ OUTPUT CONTRACT
{{"theta":0.0000,"color":"<css|#hex>","risk":"Green|Amber|Red"}}
""".strip()

def stage2_prompt(r1:Dict[str,Any], s0:Dict[str,Any], s:Settings) -> str:
    counts = s.action_counts
    rules = textwrap.dedent(f"""
        • Provide exactly {counts['Green']} actions if Green, {counts['Amber']} if Amber, {counts['Red']} if Red.
        • Each action must start with an imperative verb (e.g., "Check", "Call", "Document").
        • Max 140 characters per action; concise and clear.
        • Include top-level integer "cooldown" (minutes until next check).
        • Avoid abbreviations (no "SI", "PRN"), medical jargon, or PHI.
        • For Red: first action → immediate human intervention; next → brief breathing prompt.
    """).strip()

    audit = textwrap.dedent("""
        [ ] Correct number of actions for tier
        [ ] Each action imperative & ≤140 chars
        [ ] "cooldown" present & integer 1–120
        [ ] JSON keys exactly: actions, cooldown
    """).strip()

    return f"""
╔════════════════════════════════════════════════════════════╗
║ QMHS v3.2 • STAGE 2 — RISK → ACTIONS (JSON-ONLY)          ║
╚════════════════════════════════════════════════════════════╝

▶ STATUS θ={r1['theta']:.4f}, color={r1['color']}, tier={r1['risk']}
▶ CROWDING {s0['crowding']}, noise={s0['ward_noise_dB']} dB
▶ RESOURCES Nurse:Patient={s.staff_ratio}, SeclusionRoom={'YES' if s.seclusion_room else 'NO'}

▶ RULESET
{rules}

▶ AUDIT CHECKLIST
{audit}

▶ OUTPUT CONTRACT
{{"actions":["..."],"cooldown":<int>}}
""".strip()

def stage3_prompt(r1:Dict[str,Any], s:Settings) -> str:
    tone = {"Green":"supportive-reflective","Amber":"grounding-reassuring","Red":"brief-calming"}[r1["risk"]]
    rules = textwrap.dedent(f"""
        • Max 650 characters (~90s read-aloud).
        • Use second-person ("you"), simple grade-8 language.
        • Do not mention diagnoses, medications, self-harm, or shame.
        • Include exactly one grounding technique (breath, sensory, or touch).
        • End with a clear pause cue (e.g., "…").
        • Output single key "script" with no extra fields.
    """).strip()

    audit = textwrap.dedent("""
        [ ] JSON with single "script" key
        [ ] Character count ≤650
        [ ] Contains one grounding instruction
        [ ] No prohibited language
        [ ] Ends with pause cue
    """).strip()

    return f"""
╔════════════════════════════════════════════════════════════╗
║ QMHS v3.2 • STAGE 3 — MICRO-INTERVENTION SCRIPT (JSON-ONLY)║
╚════════════════════════════════════════════════════════════╝

▶ CONTEXT tier={r1['risk']}, tone={tone}

▶ RULESET
{rules}

▶ AUDIT CHECKLIST
{audit}

▶ OUTPUT CONTRACT
{{"script":""}}
""".strip()

def stage4_prompt(r1:Dict[str,Any], r2:Dict[str,Any], r3:Dict[str,Any], s0:Dict[str,Any], s:Settings) -> str:
    hdr = {
        "ts": s0["ts"],
        "theta": r1["theta"],
        "risk": r1["risk"],
        "actions": r2["actions"],
        "cooldown": r2["cooldown"],
        "confidence": s.confidence_threshold
    }
    hdr["digest"] = hashlib.sha256(json.dumps(hdr).encode()).hexdigest()
    return json.dumps(hdr, separators=(',',':'))

# ════════════════════════════════════════════════════════════════
# SENSOR SNAPSHOT (stub for real I/O)
# ════════════════════════════════════════════════════════════════
def sensor_snapshot() -> Dict[str,Any]:
    return {
        "noise": random.uniform(45,65),
        "lux": random.uniform(80,250),
        "crowding": random.choice(["low","medium","high"]),
        "vitals": {
            "hr": random.randint(60,95),
            "spo2": random.randint(94,100),
            "bp": f"{random.randint(110,130)}/{random.randint(70,85)}"
        },
        "recent_red": "no"
    }

# ════════════════════════════════════════════════════════════════
# SCANNER THREAD
# ════════════════════════════════════════════════════════════════
class ScannerThread(threading.Thread):
    def __init__(self, cfg:Settings, db:ReportDB, ai:OpenAIClient, status:tk.StringVar) -> None:
        super().__init__(daemon=True)
        self.cfg, self.db, self.ai, self.status = cfg, db, ai, status
        self.cap = cv2.VideoCapture(cfg.camera_idx, cv2.CAP_ANY)
        self.loop = asyncio.new_event_loop()
        self.stop_ev = threading.Event()
        self.last_red_ts: float|None = None

    def run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.main())

    async def main(self) -> None:
        await self.db.init()
        last_time = 0.0
        try:
            while not self.stop_ev.is_set():
                ok, frame = self.cap.read()
                if ok and (time.time() - last_time) >= self.cfg.sampling_interval:
                    last_time = time.time()
                    await self.process(frame)
                await asyncio.sleep(0.05)
        finally:
            await self.db.close()
            self.cap.release()

    def stop(self) -> None:
        self.stop_ev.set()

    async def process(self, frame:np.ndarray) -> None:
        self.status.set("Scanning…")

        # Stage 0
        env = sensor_snapshot()
        if self.last_red_ts and (time.time() - self.last_red_ts) < 900:
            env["recent_red"] = "yes"
        s0 = stage0_payload(self.cfg, env)

        # BioVector
        vec = [round(float(x),6) for x in BioVector.from_frame(frame).arr]

        # Stage 1
        try:
            out1 = await self.ai.chat(stage1_prompt(vec, s0, self.cfg), 500)
            r1 = json.loads(out1)
        except Exception as e:
            LOGGER.error("Stage1 fallback: %s", e)
            theta = np.clip(np.linalg.norm(vec),0,1)*np.pi
            r1 = {"theta":theta,"color":"orange","risk":"Red" if theta>=2 else "Amber"}
        if r1["risk"] == "Red":
            self.last_red_ts = time.time()

        # Stage 2
        try:
            out2 = await self.ai.chat(stage2_prompt(r1, s0, self.cfg), 450)
            r2 = json.loads(out2)
        except Exception as e:
            LOGGER.error("Stage2 fallback: %s", e)
            r2 = {"actions":["Face-to-face check NOW",f"Call {self.cfg.emerg_contact}"],"cooldown":30}

        # Stage 3
        try:
            out3 = await self.ai.chat(stage3_prompt(r1, self.cfg), 350)
            r3 = json.loads(out3)
        except Exception as e:
            LOGGER.error("Stage3 fallback: %s", e)
            r3 = {"script":"Take a slow breath and feel your feet on the floor…"}

        # Stage 4 (local digest)
        r4 = json.loads(stage4_prompt(r1, r2, r3, s0, self.cfg))

        # Quantum explanatory metric
        qexp = q_intensity(r1["theta"], (frame.mean()/255.0, 0.1))

        # Save
        report = {"s0":s0,"s1":r1,"s2":r2,"s3":r3,"s4":r4,"q_exp":float(qexp)}
        await self.db.save(s0["ts"], report)

        self.status.set(f"Risk {r1['risk']} logged.")
        if self.cfg.mode_autonomous and r1["risk"] == "Red":
            mb.showwarning("QMHS ALERT","Red tier detected! Please intervene.")
            LOGGER.info("Autonomous alert triggered.")

# ════════════════════════════════════════════════════════════════
# TKINTER GUI
# ════════════════════════════════════════════════════════════════
class QMHSApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("QMHS v3.2 – Deep-Loop Care")
        self.geometry("860x600")

        self.crypto   = AESGCMCrypto(MASTER_KEY)
        self.settings = Settings.load(self.crypto)
        if not os.path.exists(SETTINGS_FILE):
            self.settings.prompt_gui()
            self.settings.save(self.crypto)
        if not self.settings.api_key:
            mb.showerror("Missing API Key","Please set your OpenAI API key in Settings.")
            self.destroy()
            return

        self.status = tk.StringVar(value="Initializing…")
        tk.Label(self, textvariable=self.status, font=("Helvetica",14)).pack(pady=10)
        btn_frame = tk.Frame(self); btn_frame.pack()
        tk.Button(btn_frame, text="Settings",     command=self.open_settings).grid(row=0,column=0,padx=5)
        tk.Button(btn_frame, text="View Reports", command=self.view_reports).grid(row=0,column=1,padx=5)
        self.text = tk.Text(self, height=24, width=100, wrap="word")
        self.text.pack(padx=8, pady=8)

        self.db      = ReportDB(self.settings.db_path, self.crypto)
        self.ai      = OpenAIClient(api_key=self.settings.api_key)
        self.scanner = ScannerThread(self.settings, self.db, self.ai, self.status)
        self.scanner.start()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def open_settings(self) -> None:
        self.settings.prompt_gui()
        self.settings.save(self.crypto)
        mb.showinfo("Settings","Saved. Restart to apply hardware changes.")

    def view_reports(self) -> None:
        rows = asyncio.run(self.db.list_reports())
        if not rows:
            mb.showinfo("Reports","No reports stored.")
            return
        opts = "\n".join(f"{rid} – {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}"
                         for rid,ts in rows[:30])
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
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        QMHSApp().mainloop()
    except KeyboardInterrupt:
        LOGGER.info("Exiting QMHS.")
