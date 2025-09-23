"""Tkinter-based onboarding wizard for Telegram Copier configuration."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from typing import Dict, Optional

ACCENT_COLOR = "#4C8BF5"
INACTIVE_COLOR = "#CBD5E1"
BACKGROUND_COLOR = "#F8FAFC"
TEXT_COLOR = "#0F172A"


class _StepIndicator:
    """Utility widget to display the wizard stepper."""

    def __init__(self, master: tk.Widget, index: int, title: str) -> None:
        self.frame = ttk.Frame(master)

        self.canvas = tk.Canvas(
            self.frame,
            width=32,
            height=32,
            highlightthickness=0,
            bd=0,
            bg=BACKGROUND_COLOR,
        )
        self.circle_id = self.canvas.create_oval(4, 4, 28, 28, fill=INACTIVE_COLOR, outline="")
        self.text_id = self.canvas.create_text(
            16,
            16,
            text=str(index),
            fill="white",
            font=("Segoe UI", 10, "bold"),
        )
        self.canvas.pack()

        self.label = ttk.Label(self.frame, text=title, font=("Segoe UI", 10, "bold"))
        self.label.pack(pady=(4, 0))

    def grid(self, **kwargs: object) -> None:
        self.frame.grid(**kwargs)

    def set_state(self, active: bool, completed: bool = False) -> None:
        if active:
            fill = ACCENT_COLOR
            text_color = "white"
        elif completed:
            fill = ACCENT_COLOR
            text_color = "white"
        else:
            fill = INACTIVE_COLOR
            text_color = "white"
        self.canvas.itemconfig(self.circle_id, fill=fill)
        self.canvas.itemconfig(self.text_id, fill=text_color)


class _StepConnector:
    """Connector between wizard steps."""

    def __init__(self, master: tk.Widget) -> None:
        self.canvas = tk.Canvas(
            master,
            width=72,
            height=6,
            highlightthickness=0,
            bd=0,
            bg=BACKGROUND_COLOR,
        )
        self.line_id = self.canvas.create_line(
            4, 3, 68, 3, fill=INACTIVE_COLOR, width=3, capstyle=tk.ROUND
        )

    def grid(self, **kwargs: object) -> None:
        self.canvas.grid(**kwargs)

    def set_state(self, active: bool) -> None:
        color = ACCENT_COLOR if active else INACTIVE_COLOR
        self.canvas.itemconfig(self.line_id, fill=color)


class _Wizard(tk.Toplevel):
    def __init__(self, root: tk.Tk) -> None:
        super().__init__(root)
        self.result: Optional[Dict[str, str]] = None

        self.title("Telegram Copier Onboarding")
        self.configure(bg=BACKGROUND_COLOR)
        self.resizable(False, False)
        self.transient(root)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BACKGROUND_COLOR)
        style.configure("TLabel", background=BACKGROUND_COLOR, foreground=TEXT_COLOR)
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        self.content = ttk.Frame(self, padding=20)
        self.content.pack(fill="both", expand=True)

        self._create_stepper()
        self._create_steps()

        self._show_step(0)

        # Sichtbarkeit/Position erzwingen
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = max(0, (sw - w)//2), max(0, (sh - h)//2)
        self.geometry(f"+{x}+{y}")
        self.attributes("-topmost", True)
        self.after(400, lambda: self.attributes("-topmost", False))
        self.deiconify()
        self.focus_force()

    def _create_stepper(self) -> None:
        self.stepper_frame = ttk.Frame(self.content)
        self.stepper_frame.pack(fill="x", pady=(0, 20))
        self.stepper_frame.columnconfigure(0, weight=1)
        self.stepper_frame.columnconfigure(1, weight=0)
        self.stepper_frame.columnconfigure(2, weight=1)
        self.stepper_frame.columnconfigure(3, weight=0)
        self.stepper_frame.columnconfigure(4, weight=1)

        self.step1 = _StepIndicator(self.stepper_frame, 1, "Connect Telegram")
        self.connector = _StepConnector(self.stepper_frame)
        self.step2 = _StepIndicator(self.stepper_frame, 2, "Select Chats")

        self.step1.grid(row=0, column=0, sticky="n", padx=10)
        self.connector.grid(row=0, column=2, sticky="we")
        self.step2.grid(row=0, column=4, sticky="n", padx=10)

    def _create_steps(self) -> None:
        self.api_id_var = tk.StringVar()
        self.api_hash_var = tk.StringVar()
        self.target_var = tk.StringVar()
        self.forward_var = tk.StringVar()

        self.api_id_var.trace_add("write", lambda *_: self._update_step1_state())
        self.api_hash_var.trace_add("write", lambda *_: self._update_step1_state())
        self.target_var.trace_add("write", lambda *_: self._update_step2_state())
        self.forward_var.trace_add("write", lambda *_: self._update_step2_state())

        self.step_container = ttk.Frame(self.content)
        self.step_container.pack(fill="both", expand=True)

        self.step_frames = []
        self._create_step1()
        self._create_step2()

    def _create_step1(self) -> None:
        frame = ttk.Frame(self.step_container)
        self.step_frames.append(frame)

        description = (
            "Provide your Telegram API credentials. You can obtain them "
            "from my.telegram.org."
        )
        ttk.Label(frame, text=description, wraplength=360, justify="left").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 15)
        )

        ttk.Label(frame, text="API ID").grid(row=1, column=0, sticky="w")
        vcmd = (frame.register(self._validate_api_id), "%P")
        self.api_id_entry = ttk.Entry(frame, textvariable=self.api_id_var, validate="key", validatecommand=vcmd)
        self.api_id_entry.grid(row=2, column=0, columnspan=2, sticky="we", pady=(0, 12))

        ttk.Label(frame, text="API Hash").grid(row=3, column=0, sticky="w")
        self.api_hash_entry = ttk.Entry(frame, textvariable=self.api_hash_var, show="*")
        self.api_hash_entry.grid(row=4, column=0, columnspan=2, sticky="we", pady=(0, 12))

        self.step1_buttons = ttk.Frame(frame)
        self.step1_buttons.grid(row=5, column=0, columnspan=2, sticky="e", pady=(10, 0))

        cancel_button = ttk.Button(self.step1_buttons, text="Cancel", command=self._cancel)
        cancel_button.grid(row=0, column=0)

        self.next_button = ttk.Button(
            self.step1_buttons,
            text="Next",
            command=lambda: self._show_step(1),
            state="disabled",
            style="Accent.TButton",
        )
        self.next_button.grid(row=0, column=1, padx=(8, 0))

        frame.columnconfigure(0, weight=1)

    def _create_step2(self) -> None:
        frame = ttk.Frame(self.step_container)
        self.step_frames.append(frame)

        ttk.Label(frame, text="Choose the chats for the bot.").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 15)
        )

        ttk.Label(frame, text="TG_TARGET").grid(row=1, column=0, sticky="w")
        self.target_entry = ttk.Entry(frame, textvariable=self.target_var)
        self.target_entry.grid(row=2, column=0, columnspan=2, sticky="we", pady=(0, 12))

        ttk.Label(frame, text="FORWARD_TO (optional)").grid(row=3, column=0, sticky="w")
        self.forward_entry = ttk.Entry(frame, textvariable=self.forward_var)
        self.forward_entry.grid(row=4, column=0, columnspan=2, sticky="we", pady=(0, 12))

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=5, column=0, columnspan=2, sticky="we", pady=(10, 0))
        button_frame.columnconfigure(0, weight=1)

        cancel_button = ttk.Button(button_frame, text="Cancel", command=self._cancel)
        cancel_button.grid(row=0, column=0, sticky="w")

        back_button = ttk.Button(button_frame, text="Back", command=lambda: self._show_step(0))
        back_button.grid(row=0, column=1, padx=(8, 0))

        self.start_button = ttk.Button(
            button_frame,
            text="Start Bot",
            command=self._finish,
            state="disabled",
            style="Accent.TButton",
        )
        self.start_button.grid(row=0, column=2)

        frame.columnconfigure(0, weight=1)

    def _show_step(self, index: int) -> None:
        self.current_step = index
        for i, frame in enumerate(self.step_frames):
            if i == index:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

        self.step1.set_state(active=index == 0, completed=index > 0)
        self.step2.set_state(active=index == 1, completed=index > 1)
        self.connector.set_state(active=index > 0)

        if index == 0:
            self.geometry("")
            self.api_id_entry.focus_set()
        else:
            self.geometry("")
            self.target_entry.focus_set()
        self.update_idletasks()
        self._center_window()
        self._update_step1_state()
        self._update_step2_state()

    def _validate_api_id(self, proposed: str) -> bool:
        return proposed.isdigit() or proposed == ""

    def _is_step1_valid(self) -> bool:
        api_id = self.api_id_var.get().strip()
        api_hash = self.api_hash_var.get().strip()
        return api_id.isdigit() and bool(api_id) and bool(api_hash)

    def _is_step2_valid(self) -> bool:
        target = self.target_var.get().strip()
        return bool(target)

    def _update_step1_state(self) -> None:
        if self._is_step1_valid():
            self.next_button.state(["!disabled"])
        else:
            self.next_button.state(["disabled"])

    def _update_step2_state(self) -> None:
        valid = self._is_step2_valid() and self._is_step1_valid()
        if valid:
            self.start_button.state(["!disabled"])
        else:
            self.start_button.state(["disabled"])

    def _cancel(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()

    def _finish(self) -> None:
        if not (self._is_step1_valid() and self._is_step2_valid()):
            return

        config = {
            "api_id": self.api_id_var.get().strip(),
            "api_hash": self.api_hash_var.get().strip(),
            "tg_target": self.target_var.get().strip(),
            "forward_to": self.forward_var.get().strip(),
        }

        try:
            lines = [
                f"TG_API_ID={config['api_id']}",
                f"TG_API_HASH={config['api_hash']}",
                f"TG_TARGET={config['tg_target']}",
            ]
            if config["forward_to"]:
                lines.append(f"FORWARD_TO={config['forward_to']}")
            Path(".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Error", f"Failed to write .env file: {exc}")
            return

        self.result = config
        self.grab_release()
        self.destroy()

    def _center_window(self) -> None:
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        x = max((screen_width // 2) - (width // 2), 0)
        y = max((screen_height // 2) - (height // 2), 0)
        self.geometry(f"{width}x{height}+{x}+{y}")


def run_onboarding(root: tk.Tk) -> Optional[Dict[str, str]]:
    """Run the onboarding wizard and return configuration values."""

    wizard = _Wizard(root)
    root.wait_window(wizard)
    return wizard.result
