"""Bootstrap module for launching the TelegramCopier UI."""

from __future__ import annotations

from typing import Any, Dict, Optional


def run_app(session: Optional[Dict[str, Any]] = None, initial_page: str = "dashboard") -> None:
    """Start the desktop UI.

    Parameters
    ----------
    session:
        Optional session information. Currently unused, but reserved for
        compatibility with other launchers.
    initial_page:
        Name of the page that should be shown first when the UI opens.
    """

    # Avoid circular imports by importing lazily within the function.
    from TelegramCopier_Windows import (
        ConfigManager,
        SetupAssistant,
        TradingGUI,
        check_first_run,
    )
    import tkinter as tk

    _ = session  # suppress "unused" linter warnings while keeping the signature stable

    try:
        config_manager = ConfigManager()
        cfg = config_manager.load_config()

        def _has_required_credentials(config: Dict) -> bool:
            telegram_cfg = config.get('telegram', {})
            required_fields = ('api_id', 'api_hash', 'phone')
            return all(telegram_cfg.get(field) for field in required_fields)

        def _launch_setup_wizard() -> bool:
            root = tk.Tk()
            root.withdraw()
            setup = SetupAssistant(root)
            setup.show_setup_dialog()
            root.mainloop()
            root.destroy()

            if not setup.config_saved:
                print("Setup abgebrochen. Anwendung wird beendet.")
                return False

            return True

        setup_completed = False

        if check_first_run():
            if not _launch_setup_wizard():
                return

            setup_completed = True
            cfg = config_manager.load_config()

        telegram_cfg = cfg.get('telegram', {})
        missing_credentials = not _has_required_credentials(cfg)

        if not setup_completed and (telegram_cfg.get('prompt_credentials_on_start') or missing_credentials):
            if not _launch_setup_wizard():
                return

            setup_completed = True
            cfg = config_manager.load_config()
            telegram_cfg = cfg.get('telegram', {})
            missing_credentials = not _has_required_credentials(cfg)

        if missing_credentials:
            print("Keine gültigen API-Zugangsdaten vorhanden. Anwendung wird beendet.")
            return

        app = TradingGUI(cfg)
        app.set_initial_page(initial_page)
        app.run()
    except Exception as exc:  # pragma: no cover - defensive error reporting
        print(f"Fehler beim Starten der Anwendung: {exc}")
        input("Drücken Sie Enter zum Beenden...")
