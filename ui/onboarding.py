import os, json, tkinter as tk
from tkinter import ttk, messagebox

DARK_BG  = "#0f1217"
CARD_BG  = "#151a22"
TEXT_FG  = "#eef2f7"
MUTED_FG = "#8a94a6"
ACCENT   = "#00d1ff"

CHAT_CFG = "chat_config.json"

def _load_known_chats():
    if os.path.exists(CHAT_CFG):
        try:
            with open(CHAT_CFG, "r", encoding="utf-8") as f:
                return json.load(f).get("known_chats", [])
        except Exception:
            pass
    return [
        {"title":"Premium Forex Signals","username":"@PremiumForexSignals"},
        {"title":"Gold Trading VIP","username":"@GoldTradingVIP"},
        {"title":"Crypto Signals Pro","username":"@CryptoSignalsPro"},
    ]

class OnboardingWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Onboarding")
        self.configure(bg=DARK_BG)
        # <<< WICHTIG: sichtbare, bedienbare Window-Controls
        self.geometry("820x640+150+120")
        self.minsize(640, 520)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda e: self._cancel())
        # >>>

        # Theme
        st = ttk.Style(self)
        try: st.theme_use("clam")
        except: pass
        st.configure("Bg.TFrame", background=DARK_BG)
        st.configure("Card.TFrame", background=CARD_BG)
        st.configure("Title.TLabel", font=("Segoe UI", 22, "bold"), foreground=TEXT_FG, background=DARK_BG)
        st.configure("H2.TLabel",    font=("Segoe UI", 14, "bold"), foreground=TEXT_FG, background=CARD_BG)
        st.configure("Text.TLabel",  foreground=TEXT_FG, background=CARD_BG)
        st.configure("Muted.TLabel", foreground=MUTED_FG, background=CARD_BG)

        self.result = None
        self.api_id   = tk.StringVar()
        self.api_hash = tk.StringVar()
        self.tg_target= tk.StringVar()

        root = ttk.Frame(self, style="Bg.TFrame", padding=16); root.pack(fill="both", expand=True)
        ttk.Label(root, text="ONBOARDING", style="Title.TLabel").pack(anchor="w", pady=(0,10))
        self.stepbar = ttk.Frame(root, style="Bg.TFrame"); self.stepbar.pack(fill="x", pady=(0,8))
        self.container = ttk.Frame(root, style="Bg.TFrame"); self.container.pack(fill="both", expand=True)

        self.step1 = self._build_step1(self.container)
        self.step2 = self._build_step2(self.container)
        self._show(1)

    # UI parts
    def _stepper(self, active:int):
        for w in self.stepbar.winfo_children(): w.destroy()
        row = ttk.Frame(self.stepbar, style="Bg.TFrame"); row.pack()
        for i in (1,2,3,4):
            c = tk.Canvas(row, width=28, height=28, bg=DARK_BG, highlightthickness=0)
            c.create_oval(4,4,24,24, fill=(ACCENT if i<=active else "#2a3242"), outline="")
            c.pack(side="left", padx=8)

    def _build_step1(self, parent):
        f = ttk.Frame(parent, style="Bg.TFrame")
        card = ttk.Frame(f, style="Card.TFrame", padding=16); card.pack(fill="x", pady=8)
        ttk.Label(card, text="Connect Telegram", style="H2.TLabel").pack(anchor="w", pady=(0,8))

        g = ttk.Frame(card, style="Card.TFrame"); g.pack(fill="x")
        ttk.Label(g, text="API ID", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(g, textvariable=self.api_id).grid(row=1, column=0, sticky="ew", pady=(0,10))
        ttk.Label(g, text="API Hash", style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Entry(g, textvariable=self.api_hash, show="•").grid(row=3, column=0, sticky="ew")
        g.columnconfigure(0, weight=1)

        nav = ttk.Frame(f, style="Bg.TFrame"); nav.pack(fill="x", pady=(12,0))
        ttk.Button(nav, text="Weiter ▸", command=self._go2).pack(side="right")
        ttk.Button(nav, text="Abbrechen", command=self._cancel).pack(side="left")
        return f

    def _build_step2(self, parent):
        f = ttk.Frame(parent, style="Bg.TFrame")
        card = ttk.Frame(f, style="Card.TFrame", padding=16); card.pack(fill="both", expand=True, pady=8)
        ttk.Label(card, text="Select Chats", style="H2.TLabel").pack(anchor="w", pady=(0,8))

        self.listbox = tk.Listbox(card, selectmode="browse", height=12,
                                  bg=CARD_BG, fg=TEXT_FG, highlightthickness=0, relief="flat")
        self.listbox.pack(fill="both", expand=True)
        for c in _load_known_chats():
            ident = c.get("username") or str(c.get("id",""))
            self.listbox.insert("end", f"{c.get('title','Chat')}  ({ident})")

        ttk.Label(card, text="Hinweis: Liste aus chat_config.json (in der App aktualisieren).",
                  style="Muted.TLabel").pack(anchor="w", pady=(6,0))

        nav = ttk.Frame(f, style="Bg.TFrame"); nav.pack(fill="x", pady=(12,0))
        ttk.Button(nav, text="◂ Zurück", command=self._go1).pack(side="left")
        ttk.Button(nav, text="START BOT", command=self._finish).pack(side="right")
        return f

    # Navigation/Actions
    def _show(self, step:int):
        for w in self.container.winfo_children(): w.forget()
        if step == 1:
            self.step1.pack(fill="both", expand=True); self._stepper(1)
        else:
            self.step2.pack(fill="both", expand=True); self._stepper(2)

    def _go1(self): self._show(1)

    def _go2(self):
        aid = self.api_id.get().strip(); ah = self.api_hash.get().strip()
        if not aid.isdigit() or not ah:
            messagebox.showerror("Fehler", "Bitte gültige API ID (nur Ziffern) und API Hash eingeben.")
            return
        self._show(2)

    def _finish(self):
        sel = self.listbox.curselection()
        target = ""
        if sel:
            txt = self.listbox.get(sel[0])
            if "(" in txt and ")" in txt:
                target = txt.split("(")[-1].rstrip(")")
        self.result = {"api_id": int(self.api_id.get().strip()), "api_hash": self.api_hash.get().strip(), "tg_target": target}
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

def run_onboarding(_root=None):
    """Startet den Wizard als Hauptfenster (robust auf Windows) und liefert dict|None."""
    win = OnboardingWindow()
    # in die Bildschirmmitte
    win.update_idletasks()
    w, h = win.winfo_width(), win.winfo_height()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    x, y = max(0,(sw-w)//2), max(0,(sh-h)//2)
    win.geometry(f"+{x}+{y}")
    win.mainloop()
    return getattr(win, "result", None)
