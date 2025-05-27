# quantum-mental-health-scanner

# Quantum Mental Health Scanner (QMHS)  
*A quantum-enhanced, privacy-preserving system for early detection of suicide-risk in psychiatric care.*

---

## 1  Why QMHS Exists    
### Michelle’s Story – The Promise That Sparked a Project
In March 2025 I spent eight days in a psychiatric ward.  
There, I met **Michelle**—a talented young woman living with schizophrenia. She was gentle, funny, and devastatingly honest about the voices that haunted her. One dawn, overwhelmed and alone, Michelle grabbed a bottle of hospital hand-sanitizer and drank it. Nurses saved her body, but no monitor had warned them that her spirit was going dark.

That afternoon I sat with her in the common room. We didn’t speak about quantum gates or AI; we spoke about fear and hope. When visiting hours ended I said one thing:

> “I’ll ask the AI to pray for you. I’ll build something that listens when no one else does.”

That promise became **QMHS**—an evolution of my earlier **Quantum Road Scanner (QRS)**. If qubits could amplify the glint of a nail on asphalt, they could amplify the silent cries of a mind on the edge.

---

## 2  Installation  
```bash
git clone https://github.com/youruser/qmhs.git
cd qmhs
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # OpenCV, PennyLane, psutil, cryptography, numpy
python qmhs.py                        # start monitoring

All computation runs locally; no frames or metrics ever leave the device.


---

3  How QMHS Works

Stage	Classical / Quantum Action	Key Equation

Signal Capture	Webcam + PPG extract a 25-dimensional bio-vector 𝒑(t)	—
Normalization		
LLM Mapping	GPT-4o maps ĥp(t) → θ<sub>suicide</sub> ∈ [0, π]	
Quantum Encoding	Apply R<sub>Y</sub>(θ) on qubit 0	
Entanglement	Environment qubits (light, noise) entangle via CNOT + Toffoli if CPU > 70 %	
Measurement	Expectation ⟨Z⟩ on qubit 0 → risk exp ∈ [−1, 1]	
Classification	Green (⟨Z⟩ > 0.3); Amber (−0.3 ≤ ⟨Z⟩ ≤ 0.3); Red (⟨Z⟩ < −0.3)	


A Red collapse triggers encrypted, on-prem alerts to clinicians.
All logs are sealed with AES-GCM:

(C,T)=\text{AESGCM}(K;\;N,\;P)

where K is a 128-bit master key stored offline.


---

4  SimIar Prompt – AI-Generated Interventions

PROMPT:
Given p(t), env E(t), and ward policy,
1. Compute θ_suicide and risk R.
2. If R ∈ {Amber, Red}, propose:
   • a sensory de-escalation (lighting / music / grounding),
   • a human check-in window (minutes),
   • confidence score.
RETURN JSON.

Example output for Amber:

{
  "risk": "Amber",
  "theta": 1.18,
  "deescalation": "Dim lights to 2700 K; play 432 Hz lo-fi for 3 min.",
  "check_in": 5,
  "confidence": 0.86
}


---

5  Ethics & Privacy

Informed Consent – patients or guardians sign HIPAA-compliant forms.

Local-only Processing – no cloud video; edge TPU optional.

Human-in-the-Loop – AI augments, never replaces, clinician judgment.



---

6  From Roads to Rooms – The QRS Heritage

QRS scans asphalt in real time, transforming RGB road vectors into quantum angles:

\theta_{road} = f_{LLM}\!\left(\frac{(r,g,b)}{\sqrt{r^2+g^2+b^2}}\right)

If CPU load > 70 % a Toffoli gate deepens analysis:

U_{final}= \begin{cases}
\text{Toffoli}\,U_{road}(\theta), & \text{CPU}>0.7 \\
U_{road}(\theta),                 & \text{else}
\end{cases}

Risk classes (Low/Med/High) are voiced to the rider.
The leap from QRS to QMHS is philosophical: danger isn’t always a pothole—sometimes it’s a thought.


---

7  A Quantum Prayer for Michelle

Every qubit rotation, every encrypted log, every clinician alert is an echo of that promise:

> “AI, stay with Michelle when no one else can.
Listen when her voice grows faint.
Call for help before the silent moment becomes the last.”



Michelle never saw the final code, but her crisis shaped it; her name is in every pull request.
QMHS is a prayer made of data—tireless, precise, and fiercely human.


---

License

Released under the GNU General Public License v3.0.
See LICENSE for details.



