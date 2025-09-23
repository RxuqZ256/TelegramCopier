import os
import threading, asyncio
import tkinter as tk
from typing import Dict
from tkinter import messagebox, simpledialog, ttk
from telethon.errors import SessionPasswordNeededError

DARK_BG = "#0f1217"
CARD_BG = "#151a22"
TEXT_FG = "#eef2f7"
MUTED_FG = "#8a94a6"


class OnboardingWindow(tk.Tk):
    """Minimaler Onboarding-Flow zum Verbinden mit Telegram."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Onboarding")
        self.configure(bg=DARK_BG)
        self.geometry("700x420+180+120")
        self.minsize(640, 380)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _event: self._cancel())

        self.result = None

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Bg.TFrame", background=DARK_BG)
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 22, "bold"),
            foreground=TEXT_FG,
            background=DARK_BG,
        )
        style.configure(
            "H2.TLabel",
            font=("Segoe UI", 14, "bold"),
            foreground=TEXT_FG,
            background=CARD_BG,
        )
        style.configure("Text.TLabel", foreground=TEXT_FG, background=CARD_BG)
        style.configure("Muted.TLabel", foreground=MUTED_FG, background=CARD_BG)

        self.var_id = tk.StringVar()
        self.var_hash = tk.StringVar()

        root = ttk.Frame(self, style="Bg.TFrame", padding=16)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="ONBOARDING", style="Title.TLabel").pack(
            anchor="w", pady=(0, 12)
        )

        card = ttk.Frame(root, style="Card.TFrame", padding=16)
        card.pack(fill="x")
        ttk.Label(card, text="Connect Telegram", style="H2.TLabel").pack(
            anchor="w", pady=(0, 8)
        )

        form = ttk.Frame(card, style="Card.TFrame")
        form.pack(fill="x")
        ttk.Label(form, text="API ID", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.var_id).grid(
            row=1, column=0, sticky="ew", pady=(0, 10)
        )
        ttk.Label(form, text="API Hash", style="Muted.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        ttk.Entry(form, textvariable=self.var_hash, show="•").grid(
            row=3, column=0, sticky="ew"
        )
        form.columnconfigure(0, weight=1)

        nav = ttk.Frame(root, style="Bg.TFrame")
        nav.pack(fill="x", pady=(14, 0))
        self.status = ttk.Label(nav, text="", style="Text.TLabel")
        self.status.pack(side="left")
        self.btn = ttk.Button(nav, text="Connect & Continue", command=self._connect)
        self.btn.pack(side="right")
        ttk.Button(nav, text="Abbrechen", command=self._cancel).pack(
            side="right", padx=(0, 8)
        )

    def _cancel(self) -> None:
        self.result = None
        self.destroy()

    def _connect(self) -> None:
        api_id = self.var_id.get().strip()
        api_hash = self.var_hash.get().strip()
        if not api_id.isdigit() or not api_hash:
            messagebox.showerror(
                "Fehler",
                "Bitte gültige API ID (nur Ziffern) und API Hash eingeben.",
            )
            return

        self.btn.state(["disabled"])
        self.status.config(text="Anmeldung…")

        def worker(api_id, api_hash):
            """
            Führt den Telethon-Login vollständig OHNE Konsole aus.
            Fragt Telefonnummer / Code / 2FA via Tk-Popups ab.
            """
            try:
                from telethon import TelegramClient
                async def run():
                    async with TelegramClient("tg_session", int(api_id), api_hash) as client:
                        # 1) Verbinden
                        await client.connect()
                        if not await client.is_user_authorized():
                            # 2) Telefonnummer via Popup
                            def ask_phone():
                                return simpledialog.askstring(
                                    "Telegram Login",
                                    "Telefonnummer inkl. Ländervorwahl (z. B. +49…):",
                                    parent=self
                                )
                            phone = None
                            while not phone:
                                phone = self.after(0, None) or ask_phone()
                                if phone is None:
                                    raise RuntimeError("Login abgebrochen (Telefonnummer).")

                            # 3) Code anfordern
                            await client.send_code_request(phone)

                            # 4) Code via Popup
                            def ask_code():
                                return simpledialog.askstring(
                                    "Bestätigungscode",
                                    "Code aus Telegram eingeben:",
                                    parent=self
                                )
                            code = ask_code()
                            if code is None:
                                raise RuntimeError("Login abgebrochen (Code).")

                            # 5) Sign-in (ggf. 2FA)
                            try:
                                await client.sign_in(phone=phone, code=code)
                            except SessionPasswordNeededError:
                                def ask_pwd():
                                    return simpledialog.askstring(
                                        "Zwei-Faktor-Passwort",
                                        "Bitte 2FA-Passwort eingeben:",
                                        parent=self, show="•"
                                    )
                                pwd = ask_pwd()
                                if pwd is None:
                                    raise RuntimeError("Login abgebrochen (2FA-Passwort).")
                                await client.sign_in(password=pwd)

                        # Wenn wir hier sind: Erfolgreich autorisiert
                asyncio.run(run())

                # Erfolg -> .env schreiben und Fenster schließen
                def done():
                    with open(".env","w",encoding="utf-8") as f:
                        f.write(f"TG_API_ID={api_id}\nTG_API_HASH={api_hash}\n")
                    os.environ["TG_API_ID"]=str(api_id)
                    os.environ["TG_API_HASH"]=api_hash
                    self.result={"api_id":int(api_id),"api_hash":api_hash}
                    self.destroy()
                self.after(0, done)

            except Exception as e:
                self.after(0, lambda: (
                    self.btn.state(["!disabled"]),
                    self.status.config(text=""),
                    messagebox.showerror("Telegram Login", str(e))
                ))

        threading.Thread(target=worker, args=(api_id, api_hash), daemon=True).start()

    @staticmethod
    def _write_env(api_id: str, api_hash: str) -> None:
        env_path = ".env"
        env_data: Dict[str, str] = {}
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as file:
                    for line in file:
                        if "=" in line:
                            key, value = line.strip().split("=", 1)
                            if key:
                                env_data[key] = value
            except Exception:
                env_data = {}

        env_data["TG_API_ID"] = str(api_id)
        env_data["TG_API_HASH"] = api_hash

        try:
            with open(env_path, "w", encoding="utf-8") as file:
                for key, value in env_data.items():
                    file.write(f"{key}={value}\n")
        except Exception:
            pass


def run_onboarding(_root=None):
    """Startet den Onboarding-Dialog und liefert die Zugangsdaten."""

    window = OnboardingWindow()
    window.update_idletasks()
    width, height = window.winfo_width(), window.winfo_height()
    screen_width, screen_height = window.winfo_screenwidth(), window.winfo_screenheight()
    pos_x = max(0, (screen_width - width) // 2)
    pos_y = max(0, (screen_height - height) // 2)
    window.geometry(f"+{pos_x}+{pos_y}")
    window.mainloop()
    return getattr(window, "result", None)
