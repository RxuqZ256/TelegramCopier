# ui/onboarding.py
import json, os, tkinter as tk
from tkinter import ttk, messagebox

DARK_BG   = "#0f1217"
CARD_BG   = "#151a22"
TEXT_FG   = "#eef2f7"
MUTED_FG  = "#8a94a6"
ACCENT    = "#7c5cff"
ACCENT2   = "#00d1ff"

CHAT_CFG = "chat_config.json"

def _load_known_chats():
    if os.path.exists(CHAT_CFG):
        try:
            with open(CHAT_CFG, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("known_chats", [])
        except Exception:
            pass
    # Placeholder, falls nichts da ist
    return [
        {"title": "Premium Forex Signals", "username": "@PremiumForexSignals"},
        {"title": "Gold Trading VIP", "username": "@GoldTradingVIP"},
        {"title": "Crypto Signals Pro", "username": "@CryptoSignalsPro"},
    ]

class OnboardingTwoStep(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.withdraw()
        self.title("Onboarding")
        self.configure(bg=DARK_BG)
        self.resizable(False, False)
        self.minsize(560, 520)
        self.result = None

        # Styles
        st = ttk.Style(self)
        try: st.theme_use("clam")
        except: pass
        st.configure("Bg.TFrame", background=DARK_BG)
        st.configure("Card.TFrame", background=CARD_BG)
        st.configure("Title.TLabel", font=("Segoe UI", 22, "bold"),
                    foreground=TEXT_FG, background=DARK_BG)
        st.configure("H2.TLabel", font=("Segoe UI", 14, "bold"),
                    foreground=TEXT_FG, background=CARD_BG)
        st.configure("Text.TLabel", foreground=TEXT_FG, background=CARD_BG)
        st.configure("Muted.TLabel", foreground=MUTED_FG, background=CARD_BG)
        st.configure("Nav.TButton", padding=10)
        st.map("Nav.TButton", background=[("active", CARD_BG)], foreground=[("!disabled", TEXT_FG)])

        # Root Layout
        rootf = ttk.Frame(self, style="Bg.TFrame", padding=16)
        rootf.pack(fill="both", expand=True)
        ttk.Label(rootf, text="ONBOARDING", style="Title.TLabel").pack(anchor="w", pady=(0,10))

        self.stepper = ttk.Frame(rootf, style="Bg.TFrame")
        self.stepper.pack(fill="x", pady=(0,8))
        self._build_stepper(self.stepper, active=1)

        self.container = ttk.Frame(rootf, style="Bg.TFrame")
        self.container.pack(fill="both", expand=True)

        # Steps
        self.api_id = tk.StringVar()
        self.api_hash = tk.StringVar()
        self.tg_target = tk.StringVar()

        self.step1 = self._build_step1(self.container)   # API
        self.step2 = self._build_step2(self.container)   # Chats
        self.current = 1
        self._show(1)

        # Sichtbarkeit erzwingen
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = max(0, (sw - w)//2), max(0, (sh - h)//2)
        self.geometry(f"+{x}+{y}")
        try: self.transient(master)
        except: pass
        self.lift()
        self.attributes("-topmost", True)
        self.after(600, lambda: self.attributes("-topmost", False))
        self.deiconify()
        self.grab_set()
        self.focus_force()

    # ---------- UI Teile ----------
    def _build_stepper(self, parent, active:int):
        # simple points 1..4 (nur 1 & 2 aktiv)
        for w in parent.winfo_children(): w.destroy()
        row = ttk.Frame(parent, style="Bg.TFrame"); row.pack()
        for i,txt in enumerate(["1","2","3","4"], start=1):
            dot = tk.Canvas(row, width=28, height=28, bg=DARK_BG, highlightthickness=0)
            color = ACCENT2 if i <= active else "#2a3242"
            dot.create_oval(4,4,24,24, fill=color, outline="")
            dot.pack(side="left", padx=8)
        # Linie(n) einfach weglassen – minimalistisch

    def _build_step1(self, parent):
        f = ttk.Frame(parent, style="Bg.TFrame")
        card = ttk.Frame(f, style="Card.TFrame", padding=16); card.pack(fill="x", pady=8)
        ttk.Label(card, text="Connect Telegram", style="H2.TLabel").pack(anchor="w", pady=(0,8))

        frm = ttk.Frame(card, style="Card.TFrame"); frm.pack(fill="x")
        ttk.Label(frm, text="API ID", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        e1 = ttk.Entry(frm, textvariable=self.api_id); e1.grid(row=1, column=0, sticky="ew", pady=(0,10))
        ttk.Label(frm, text="API Hash", style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        e2 = ttk.Entry(frm, textvariable=self.api_hash, show="•"); e2.grid(row=3, column=0, sticky="ew")
        frm.columnconfigure(0, weight=1)

        nav = ttk.Frame(f, style="Bg.TFrame"); nav.pack(fill="x", pady=(12,0))
        ttk.Button(nav, text="Weiter ▸", style="Nav.TButton", command=self._go_step2).pack(side="right")
        ttk.Button(nav, text="Abbrechen", style="Nav.TButton", command=self._cancel).pack(side="left")
        return f

    def _build_step2(self, parent):
        f = ttk.Frame(parent, style="Bg.TFrame")
        self._build_stepper(self.stepper, 2)

        card = ttk.Frame(f, style="Card.TFrame", padding=16); card.pack(fill="both", expand=True, pady=8)
        ttk.Label(card, text="Select Chats", style="H2.TLabel").pack(anchor="w", pady=(0,8))

        self.listbox = tk.Listbox(card, selectmode="browse", height=10, bg=CARD_BG, fg=TEXT_FG,
                                  highlightthickness=0, relief="flat")
        self.listbox.pack(fill="both", expand=True)
        for c in _load_known_chats():
            ident = c.get("username") or str(c.get("id",""))
            self.listbox.insert("end", f"{c.get('title','Chat')}  ({ident})")

        hint = ttk.Label(card, text="Tipp: Liste wird aus chat_config.json geladen. "
                                    "Später in der App unter „Chats“ aktualisieren.",
                         style="Muted.TLabel")
        hint.pack(anchor="w", pady=(6,0))

        nav = ttk.Frame(f, style="Bg.TFrame"); nav.pack(fill="x", pady=(12,0))
        ttk.Button(nav, text="◂ Zurück", style="Nav.TButton", command=self._go_step1).pack(side="left")
        ttk.Button(nav, text="START BOT", style="Nav.TButton", command=self._finish).pack(side="right")
        return f

    # ---------- Navigation ----------
    def _show(self, step:int):
        for w in self.container.winfo_children(): w.forget()
        if step == 1:
            self.step1.pack(fill="both", expand=True)
            self._build_stepper(self.stepper, 1)
        else:
            self.step2.pack(fill="both", expand=True)
            self._build_stepper(self.stepper, 2)
        self.current = step

    def _go_step1(self):
        self._show(1)

    def _go_step2(self):
        aid = self.api_id.get().strip()
        ah  = self.api_hash.get().strip()
        if not aid.isdigit() or not ah:
            messagebox.showerror("Fehler", "Bitte gültige API ID (nur Ziffern) und API Hash eingeben.")
            return
        self._show(2)

    def _cancel(self):
        self.result = None
        try: self.grab_release()
        except: pass
        self.destroy()

    def _finish(self):
        # Zielchat aus Auswahl extrahieren
        sel = self.listbox.curselection()
        target = ""
        if sel:
            text = self.listbox.get(sel[0])
            if "(" in text and ")" in text:
                target = text.split("(")[-1].rstrip(")")

        # Ergebnis setzen -> Caller speichert .env/ENV
        self.result = {
            "api_id":   int(self.api_id.get().strip()),
            "api_hash": self.api_hash.get().strip(),
            "tg_target": target.strip()
        }
        try: self.grab_release()
        except: pass
        self.destroy()

# --------- Start-API für Windows-Starter ---------
def run_onboarding(root: tk.Tk | None = None):
    own = False
    if root is None:
        root = tk.Tk()
        own = True
    wiz = OnboardingTwoStep(root)
    if own:
        root.mainloop()
        try: root.destroy()
        except: pass
    else:
        root.wait_window(wiz)
    return getattr(wiz, "result", None)
