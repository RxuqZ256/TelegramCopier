# -*- coding: utf-8 -*-
# TelegramCopier_Windows.py
# Windows GUI-App (Tkinter) mit Telethon; MT5 ist optional (nur für Live-Mode)
# Start:  python TelegramCopier_Windows.py

import asyncio
import math
import re
import json
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Awaitable, Tuple

# >>> numpy/mt5 guard + onboarding bootstrap
import os, sys, pathlib, traceback, time, glob

# 1) NumPy-Guard (MT5 braucht NumPy 1.x)
try:
    import numpy as _np
    if int(_np.__version__.split(".", 1)[0]) >= 2:
        raise RuntimeError(
            f"NumPy {_np.__version__} erkannt. MetaTrader5 benötigt NumPy 1.x. "
            "Bitte 'scripts\\fix_mt5_numpy.bat' ausführen."
        )
except ModuleNotFoundError:
    print("[startup] No module named 'numpy' (verwende scripts\\fix_mt5_numpy.bat)")
except Exception as _e:
    print("[startup]", _e)
# >>> Robust Onboarding (GUI first, console fallback) + Logging
def _console_onboarding():
    print("\n=== Telegram Console Setup ===", flush=True)
    try:
        api_id = input("API ID (nur Ziffern): ").strip()
        api_hash = input("API Hash: ").strip()
    except KeyboardInterrupt:
        print("[onboarding] cancelled by user")
        sys.exit(0)

    env_values = {}
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as env_file:
                for line in env_file:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        if key:
                            env_values[key] = value
        except Exception:
            env_values = {}

    env_values["TG_API_ID"] = str(api_id)
    env_values["TG_API_HASH"] = api_hash

    try:
        with open(".env", "w", encoding="utf-8") as env_file:
            for key, value in env_values.items():
                env_file.write(f"{key}={value}\n")
    except Exception as exc:
        print(f"[onboarding] could not update .env: {exc}")

    os.environ["TG_API_ID"] = str(api_id)
    os.environ["TG_API_HASH"] = api_hash
    print("[onboarding] configuration saved via console", flush=True)


def run_onboarding_if_needed():
    print(">>> BOOT args:", sys.argv, flush=True)
    reset_session = ("--reset-session" in sys.argv)

    # Session-Datei auf Wunsch löschen -> zwingt Telethon zur Nummer/Code-Abfrage
    if reset_session:
        # KEIN "import os" mehr hier!
        for p in glob.glob("tg_session.session*"):
            try:
                os.remove(p)
            except OSError:
                pass
        print("[onboarding] telegram session reset -> phone/code will be asked", flush=True)
    force_gui_always = ("--always-setup" in sys.argv) or ("--gui-setup" in sys.argv)
    force_console    = ("--console-setup" in sys.argv)
    if force_gui_always or "--setup" in sys.argv:
        try:
            if os.path.exists(".env"):
                os.remove(".env")
        except Exception:
            pass
        for k in ("TG_API_ID", "TG_API_HASH", "TG_TARGET", "FORWARD_TO"):
            os.environ.pop(k, None)

    if force_console:
        print("[onboarding] forcing console setup...", flush=True)
        _console_onboarding()
        return

    env_ready = bool(os.getenv("TG_API_ID") and os.getenv("TG_API_HASH"))
    has_env   = os.path.exists(".env")
    if not (force_gui_always or "--setup" in sys.argv):
        if env_ready and has_env:
            print("[onboarding] env already set + .env present -> skip"); return

    # GUI versuchen – mit eigener Mainloop (siehe ui/onboarding.py)
    try:
        print("[onboarding] showing GUI...", flush=True)
        from ui.onboarding import run_onboarding
        cfg = run_onboarding(None)   # None => eigene Mainloop
        if not cfg:
            raise RuntimeError("GUI onboarding returned no data")

        os.environ["TG_API_ID"] = str(cfg["api_id"])
        os.environ["TG_API_HASH"] = cfg["api_hash"]
        print("[onboarding] configuration loaded (GUI)", flush=True)

    except Exception as e:
        print(f"[onboarding] GUI failed -> {e}\nFalling back to console…", flush=True)
        _console_onboarding()
# <<< numpy/mt5 guard + onboarding bootstrap

# >>> UI bootstrap
def _start_ui():
    from ui.app import run_app
    session_info = {"tg_target": os.getenv("TG_TARGET",""), "user": "local"}
    print("[ui] launching main window…", flush=True)
    run_app(session=session_info, initial_page="settings")
# <<< UI bootstrap


def safe_load_chat_config() -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    """Load chat_config.json defensively and normalise the known chats list."""

    try:
        if not os.path.exists("chat_config.json"):
            return None, []

        with open("chat_config.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        payload: Optional[Dict[str, object]] = data if isinstance(data, dict) else None
        source_entries: List[Dict[str, object]] = []

        if isinstance(data, dict):
            known_chats = data.get("known_chats")
            if isinstance(known_chats, list):
                source_entries.extend(item for item in known_chats if isinstance(item, dict))
            else:
                for key, value in data.items():
                    if key == "selected_chat":
                        continue
                    if isinstance(value, dict):
                        source_entries.append(value)
        elif isinstance(data, list):
            source_entries.extend(item for item in data if isinstance(item, dict))
        else:
            raise ValueError("chat_config.json hat ein unbekanntes Format")

        normalised: List[Dict[str, object]] = []
        for entry in source_entries:
            title = (
                entry.get("title")
                or entry.get("chat_name")
                or entry.get("name")
                or entry.get("first_name")
                or "Chat"
            )
            username = entry.get("username")
            chat_id = entry.get("id")
            if chat_id is None:
                chat_id = entry.get("chat_id")
            normalised.append(
                {
                    "title": str(title).strip() if title else "Chat",
                    "username": username,
                    "id": chat_id,
                }
            )

        return payload, normalised
    except Exception as exc:
        print("Fehler beim Laden der Chat-Konfiguration:", exc)
        try:
            os.replace("chat_config.json", "chat_config.broken.json")
        except OSError:
            pass
        return None, []

# ---- optionale Abhängigkeit: MetaTrader5 (nur für Windows verfügbar) ----
try:
    import MetaTrader5 as mt5  # noqa: F401
    MT5_AVAILABLE = True
except Exception:
    MT5_AVAILABLE = False

from telethon import TelegramClient, events
from telethon.errors import (
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    SessionPasswordNeededError
)
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, font as tkfont, filedialog

# ==================== DATENSTRUKTUREN ====================

@dataclass
class ChatSource:
    """Chat-Quelle definieren"""
    chat_id: int
    chat_name: str
    chat_type: str  # 'channel', 'group', 'private'
    enabled: bool = True
    priority: int = 1  # 1=hoch, 2=mittel, 3=niedrig
    signal_count: int = 0
    last_signal: Optional[datetime] = None


@dataclass
class TradeRecord:
    """Erweiterte Trade-Aufzeichnung mit Quelleninfo"""
    ticket: str
    symbol: str
    direction: str
    lot_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime

    # Quelleninfo
    source_chat_id: int
    source_chat_name: str
    source_type: str
    source_priority: int
    original_message: str

    # Status
    status: str = "open"  # open, closed, pending
    profit_loss: float = 0.0


class ExecutionMode(Enum):
    INSTANT = "instant"
    ZONE_WAIT = "zone_wait"
    DISABLED = "disabled"


# ==================== CHAT MANAGEMENT ====================

class MultiChatManager:
    """Verwaltung mehrerer Chat-Quellen"""

    def __init__(self):
        self.chat_sources: Dict[int, ChatSource] = {}  # chat_id -> ChatSource
        self.config_file = "chat_config.json"
        self.load_config()

    def add_chat_source(self, chat_id: int, name: str, chat_type: str, enabled: bool = True):
        """Neue Chat-Quelle hinzufügen"""
        self.chat_sources[chat_id] = ChatSource(
            chat_id=chat_id,
            chat_name=name,
            chat_type=chat_type,
            enabled=enabled
        )
        self.save_config()

    def remove_chat_source(self, chat_id: int):
        """Chat-Quelle entfernen"""
        if chat_id in self.chat_sources:
            del self.chat_sources[chat_id]
            self.save_config()

    def get_chat_info(self, chat_id: int) -> Optional[ChatSource]:
        """Chat-Info abrufen"""
        return self.chat_sources.get(chat_id)

    def update_signal_stats(self, chat_id: int):
        """Signal-Statistiken aktualisieren"""
        if chat_id in self.chat_sources:
            self.chat_sources[chat_id].signal_count += 1
            self.chat_sources[chat_id].last_signal = datetime.now()
            self.save_config()

    def get_enabled_chats(self) -> List[int]:
        """Liste der aktivierten Chat-IDs"""
        return [
            chat_id for chat_id, source in self.chat_sources.items()
            if source.enabled
        ]

    def save_config(self):
        """Konfiguration speichern"""
        try:
            config_data = {}
            for chat_id, source in self.chat_sources.items():
                config_data[str(chat_id)] = {
                    'chat_name': source.chat_name,
                    'chat_type': source.chat_type,
                    'enabled': source.enabled,
                    'priority': source.priority,
                    'signal_count': source.signal_count
                }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Fehler beim Speichern der Chat-Konfiguration: {e}")

    def load_config(self):
        """Konfiguration laden"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                for chat_id_str, data in config_data.items():
                    chat_id = int(chat_id_str)
                    self.chat_sources[chat_id] = ChatSource(
                        chat_id=chat_id,
                        chat_name=data.get('chat_name', 'Unbekannt'),
                        chat_type=data.get('chat_type', 'group'),
                        enabled=data.get('enabled', True),
                        priority=data.get('priority', 1),
                        signal_count=data.get('signal_count', 0)
                    )
        except Exception as e:
            print(f"Fehler beim Laden der Chat-Konfiguration: {e}")


# ==================== TRADE TRACKING ====================

class TradeTracker:
    """Trade-Verfolgung mit Quelleninfo"""

    def __init__(self):
        self.trade_records: Dict[str, TradeRecord] = {}  # ticket -> TradeRecord

    def add_trade(self, trade_info: Dict, chat_source: ChatSource, original_message: str):
        """Trade mit Quelleninfo hinzufügen"""

        def _ensure_float(value, default=0.0):
            try:
                if isinstance(value, str):
                    value = value.replace(',', '.')
                return float(value)
            except (TypeError, ValueError):
                return default

        record = TradeRecord(
            ticket=str(trade_info['ticket']),
            symbol=trade_info['symbol'],
            direction=trade_info['direction'],
            lot_size=_ensure_float(trade_info.get('lot_size', 0.0), 0.0),
            entry_price=_ensure_float(trade_info.get('price', 0.0), 0.0),
            stop_loss=_ensure_float(trade_info.get('stop_loss', 0.0), 0.0),
            take_profit=_ensure_float(trade_info.get('take_profit', 0.0), 0.0),
            timestamp=datetime.now(),
            source_chat_id=chat_source.chat_id,
            source_chat_name=chat_source.chat_name,
            source_type=chat_source.chat_type,
            source_priority=chat_source.priority,
            original_message=(original_message or "")[:500],
            status=trade_info.get('status', 'executed'),
            profit_loss=_ensure_float(trade_info.get('profit_loss', 0.0), 0.0)
        )
        self.trade_records[record.ticket] = record

    def get_trades_by_source(self, chat_name: str) -> List[TradeRecord]:
        """Trades nach Quelle filtern"""
        return [
            record for record in self.trade_records.values()
            if record.source_chat_name == chat_name
        ]

    def get_source_statistics(self, chat_name: str) -> Dict:
        """Statistiken für eine Quelle"""
        trades = self.get_trades_by_source(chat_name)

        if not trades:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'last_trade': None
            }

        profitable = sum(1 for trade in trades if trade.profit_loss > 0)
        total_profit = sum(trade.profit_loss for trade in trades)

        return {
            'total_trades': len(trades),
            'profitable_trades': profitable,
            'win_rate': (profitable / len(trades)) * 100.0,
            'total_profit': total_profit,
            'last_trade': max(trades, key=lambda t: t.timestamp).timestamp
        }

    def update_trade_levels(self, ticket: str, stop_loss: Optional[float] = None, take_profits: Optional[List[float]] = None) -> Optional[TradeRecord]:
        """Stop-Loss und Take-Profit eines Trades aktualisieren"""
        record = self.trade_records.get(ticket)
        if not record:
            return None

        def _ensure_float(value, default=None):
            try:
                if isinstance(value, str):
                    value = value.replace(',', '.')
                return float(value)
            except (TypeError, ValueError):
                return default

        if stop_loss is not None:
            parsed_sl = _ensure_float(stop_loss, record.stop_loss)
            if parsed_sl is not None:
                record.stop_loss = parsed_sl

        if take_profits:
            parsed_tps = []
            for tp in take_profits:
                parsed_tp = _ensure_float(tp, None)
                if parsed_tp is not None:
                    parsed_tps.append(parsed_tp)
            if parsed_tps:
                record.take_profit = parsed_tps[0]

        return record

    def get_last_trade_for_chat(self, chat_id: int) -> Optional[TradeRecord]:
        """Letzten Trade für einen Chat ermitteln"""
        trades = [
            record for record in self.trade_records.values()
            if record.source_chat_id == chat_id
        ]

        if not trades:
            return None

        return max(trades, key=lambda t: t.timestamp)


# ==================== SIGNAL PROCESSOR ====================

class SignalProcessor:
    """Vereinfachter Signal-Prozessor (Demo)"""

    def __init__(self):
        self.symbol_synonyms = {
            'GOLD': 'XAUUSD',
            'XAU': 'XAUUSD',
            'XAUUSD': 'XAUUSD',
            'SILVER': 'XAGUSD',
            'XAG': 'XAGUSD',
            'XAGUSD': 'XAGUSD',
            'OIL': 'USOIL',
            'WTI': 'USOIL',
            'USOIL': 'USOIL',
            'BRENT': 'UKOIL',
            'UKOIL': 'UKOIL',
            'EURUSD': 'EURUSD',
            'EUR': 'EURUSD',
            'GBPUSD': 'GBPUSD',
            'GBP': 'GBPUSD',
            'USDJPY': 'USDJPY',
            'USD': 'USDJPY'
        }

        self.stopwords = {
            'BUY', 'SELL', 'NOW', 'TP', 'SL', 'STOP', 'LOSS', 'TARGET', 'TARGETS',
            'ENTRY', 'EXIT', 'OPEN', 'CLOSE', 'LONG', 'SHORT', 'MARKET', 'LIMIT',
            'TP1', 'TP2', 'TP3', 'TP4', 'TP5', 'SL1', 'SL2', 'BE'
        }

        self.symbol_pattern = r'[A-Z][A-Z0-9]{1,9}(?:[/\-][A-Z0-9]{1,9})?'
        bound_symbol = rf'(?<![A-Z0-9])(?P<symbol>{self.symbol_pattern})(?![A-Z0-9])'
        price_pattern = r'(?P<price>[0-9]+[0-9\.,]*)'

        def compile_patterns(templates: List[str]) -> List[re.Pattern]:
            return [re.compile(template, re.IGNORECASE) for template in templates]

        self.patterns = {
            'buy_now': compile_patterns([
                rf'{bound_symbol}.*\bbuy\b.*\bnow\b',
                rf'\bbuy\b.*\bnow\b.*{bound_symbol}',
                rf'\bbuy\b.*{bound_symbol}.*\bnow\b'
            ]),
            'sell_now': compile_patterns([
                rf'{bound_symbol}.*\bsell\b.*\bnow\b',
                rf'\bsell\b.*\bnow\b.*{bound_symbol}',
                rf'\bsell\b.*{bound_symbol}.*\bnow\b'
            ]),
            'buy_zone': compile_patterns([
                rf'{bound_symbol}.*\bbuy\b.*?{price_pattern}',
                rf'\bbuy\b.*{bound_symbol}.*?{price_pattern}',
                rf'\bbuy\b.*?{price_pattern}.*{bound_symbol}'
            ]),
            'sell_zone': compile_patterns([
                rf'{bound_symbol}.*\bsell\b.*?{price_pattern}',
                rf'\bsell\b.*{bound_symbol}.*?{price_pattern}',
                rf'\bsell\b.*?{price_pattern}.*{bound_symbol}'
            ])
        }

        self.auto_tp_sl: bool = True

    def _normalize_symbol(self, raw_symbol: str) -> Optional[str]:
        if not raw_symbol:
            return None

        token = raw_symbol.upper().strip()
        compact = re.sub(r'[^A-Z0-9]', '', token)

        if not compact or len(compact) < 3:
            return None

        if compact in self.stopwords or token in self.stopwords:
            return None

        if re.match(r'^(TP|SL)\d*$', compact):
            return None

        if compact in self.symbol_synonyms:
            return self.symbol_synonyms[compact]

        if token in self.symbol_synonyms:
            return self.symbol_synonyms[token]

        return compact

    def _match_patterns(self, key: str, message_text: str) -> Optional[Dict[str, Optional[float]]]:
        patterns = self.patterns.get(key, [])
        for pattern in patterns:
            match = pattern.search(message_text)
            if not match:
                continue

            symbol = self._normalize_symbol(match.group('symbol'))
            if not symbol:
                continue

            price_value = None
            price_text = match.groupdict().get('price')
            if price_text is not None:
                price_value = self._parse_price(price_text)

            return {'symbol': symbol, 'price': price_value}

        return None

    def _parse_price(self, value: str) -> Optional[float]:
        try:
            normalized = value.replace(',', '.').strip()
            return float(normalized)
        except (AttributeError, ValueError):
            return None

    def extract_stop_loss(self, message_text: str) -> Optional[float]:
        match = re.search(r'(?i)\bsl\b[:\s]*([0-9]+[0-9\.,]*)', message_text)
        if match:
            return self._parse_price(match.group(1))
        return None

    def extract_take_profits(self, message_text: str) -> List[float]:
        matches = re.findall(r'(?i)\btp\d*\b[:\s]*([0-9]+[0-9\.,]*)', message_text)
        take_profits: List[float] = []
        for match in matches:
            price = self._parse_price(match)
            if price is not None:
                take_profits.append(price)
        return take_profits

    async def process_signal(self, message_text: str, chat_source: ChatSource) -> Optional[Dict]:
        """Signal verarbeiten"""
        if not message_text:
            return None

        stop_loss = self.extract_stop_loss(message_text) if self.auto_tp_sl else None
        take_profits = self.extract_take_profits(message_text) if self.auto_tp_sl else []

        # Buy Now
        result = self._match_patterns('buy_now', message_text)
        if result:
            return {
                'kind': 'trade',
                'type': 'instant',
                'action': 'BUY',
                'symbol': result['symbol'],
                'source': chat_source.chat_name,
                'execution_mode': ExecutionMode.INSTANT,
                'stop_loss': stop_loss,
                'take_profits': take_profits
            }

        # Sell Now
        result = self._match_patterns('sell_now', message_text)
        if result:
            return {
                'kind': 'trade',
                'type': 'instant',
                'action': 'SELL',
                'symbol': result['symbol'],
                'source': chat_source.chat_name,
                'execution_mode': ExecutionMode.INSTANT,
                'stop_loss': stop_loss,
                'take_profits': take_profits
            }

        # Buy Zone
        result = self._match_patterns('buy_zone', message_text)
        if result and result['price'] is not None:
            return {
                'kind': 'trade',
                'type': 'zone',
                'action': 'BUY',
                'symbol': result['symbol'],
                'entry_price': result['price'],
                'source': chat_source.chat_name,
                'execution_mode': ExecutionMode.ZONE_WAIT,
                'stop_loss': stop_loss,
                'take_profits': take_profits
            }

        # Sell Zone
        result = self._match_patterns('sell_zone', message_text)
        if result and result['price'] is not None:
            return {
                'kind': 'trade',
                'type': 'zone',
                'action': 'SELL',
                'symbol': result['symbol'],
                'entry_price': result['price'],
                'source': chat_source.chat_name,
                'execution_mode': ExecutionMode.ZONE_WAIT,
                'stop_loss': stop_loss,
                'take_profits': take_profits
            }

        if self.auto_tp_sl and (stop_loss is not None or take_profits):
            return {
                'kind': 'update',
                'stop_loss': stop_loss,
                'take_profits': take_profits,
                'source': chat_source.chat_name
            }

        return None


# ==================== MAIN BOT ====================

class MultiChatTradingBot:
    """Haupt-Bot mit Multi-Chat-Unterstützung"""

    def __init__(self, api_id: Optional[str] = None, api_hash: Optional[str] = None,
                 phone: Optional[str] = None, session_name: str = "trading_session"):
        self.api_id = 0
        self.api_hash = ""
        self.phone = ""
        self.session_name = session_name

        # Components
        self.chat_manager = MultiChatManager()
        self.trade_tracker = TradeTracker()
        self.signal_processor = SignalProcessor()

        # Telegram Client (initially None until gültige Zugangsdaten vorhanden)
        self.client: Optional[TelegramClient] = None

        # Message Queue für GUI
        self.message_queue: "queue.Queue" = queue.Queue()

        # Eigene Async-Eventloop für alle Telegram-Operationen
        self.loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self.loop_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.loop_thread.start()

        # Status
        self.is_running = False
        self.demo_mode = True  # Immer mit Demo starten!
        self.pending_trade_updates: Dict[int, Dict] = {}
        self.execution_mode = ExecutionMode.INSTANT

        # Trading-Parameter (werden durch GUI/Config aktualisiert)
        self.default_lot_size: float = 0.01
        self.max_spread_pips: float = 3.0
        self.risk_percent: float = 2.0
        self.max_trades_per_hour: int = 5

        # Signal- und Sicherheitsflags
        self.instant_trading_enabled: bool = True
        self.zone_trading_enabled: bool = True
        self.require_confirmation: bool = True

        # MT5-Verbindungsparameter
        self.mt5_path: Optional[str] = None
        self.mt5_login: Optional[int] = None
        self.mt5_password: Optional[str] = None
        self.mt5_server: Optional[str] = None
        self._mt5_initialized: bool = False
        self._mt5_login_ok: bool = False
        self._last_mt5_error: Optional[str] = None

        # Zugangsdaten anwenden (falls vorhanden)
        self.update_credentials(api_id, api_hash, phone, session_name=session_name)

    def _run_async_loop(self):
        """Startet den dedizierten Async-Loop für alle Telegram-Aufgaben."""
        asyncio.set_event_loop(self.loop)
        self._loop_ready.set()
        self.loop.run_forever()

    def submit_coroutine(self, coro: Awaitable) -> Future:
        """Plant eine Coroutine auf dem internen Eventloop ein."""
        if not self._loop_ready.is_set():
            self._loop_ready.wait()
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def update_credentials(self, api_id: Optional[str], api_hash: Optional[str], phone: Optional[str],
                           session_name: Optional[str] = None):
        """Telegram Zugangsdaten aktualisieren und Client neu initialisieren"""

        if session_name:
            self.session_name = session_name

        # Vorherige Client-Instanz verwerfen
        if self.client:
            try:
                future = self.submit_coroutine(self.client.disconnect())
                future.result(timeout=10)
            except Exception:
                pass
            finally:
                self.client = None

        try:
            self.api_id = int(api_id) if api_id else 0
        except (TypeError, ValueError):
            self.api_id = 0

        self.api_hash = api_hash or ""
        self.phone = phone or ""

        if self.api_id and self.api_hash:
            # Nur erstellen, wenn gültige Daten vorhanden sind
            self.client = TelegramClient(
                self.session_name,
                self.api_id,
                self.api_hash,
                loop=self.loop
            )

    async def ensure_connected(self) -> bool:
        """Stellt sicher, dass der Telegram-Client verbunden ist."""
        if not self.client:
            return False

        try:
            if not self.client.is_connected():
                await self.client.connect()
            return True
        except Exception as e:
            self.log(f"Fehler beim Aufbau der Telegram-Verbindung: {e}", "ERROR")
            return False

    async def ensure_authorized(
        self,
        *,
        request_code: bool = False,
        notify_gui: bool = False,
        message: Optional[str] = None
    ) -> bool:
        """Verbindungs- und Authentifizierungsprüfung mit optionaler Benachrichtigung."""

        if not await self.ensure_connected():
            return False

        try:
            if await self.client.is_user_authorized():
                return True

            if request_code:
                if self.phone:
                    try:
                        await self.client.send_code_request(self.phone)
                        self.log(
                            "Login-Code angefordert. Bitte Code in der Anwendung eingeben.",
                            "INFO"
                        )
                    except Exception as code_error:
                        self.log(f"Fehler beim Anfordern des Login-Codes: {code_error}", "ERROR")
                else:
                    self.log(
                        "Keine Telefonnummer hinterlegt. Login-Code kann nicht angefordert werden.",
                        "ERROR"
                    )

            error_message = message or (
                "Telegram-Login erforderlich. Bitte geben Sie den Login-Code ein."
            )
            self.log(error_message, "ERROR")
            if notify_gui:
                self.notify_auth_required(error_message)
            return False
        except Exception as e:
            self.log(f"Fehler bei der Authentifizierungsprüfung: {e}", "ERROR")
            return False

    def notify_auth_required(self, message: str):
        """GUI informieren, dass ein Login-Code benötigt wird."""
        self.send_message('AUTH_REQUIRED', {'message': message})

    async def complete_login_with_code(self, code: str) -> Dict[str, object]:
        """Login-Code validieren und Telegram-Session abschließen."""

        result: Dict[str, object] = {
            'success': False,
            'message': None,
            'require_password': False
        }

        if not code:
            result['message'] = "Es wurde kein Login-Code übermittelt."
            self.log(result['message'], "WARNING")
            return result

        if not self.client:
            result['message'] = "Telegram-Client nicht initialisiert."
            self.log(result['message'], "ERROR")
            return result

        if not await self.ensure_connected():
            result['message'] = "Telegram-Client konnte nicht verbunden werden."
            return result

        try:
            if await self.client.is_user_authorized():
                self.log("Telegram ist bereits autorisiert. Bot kann gestartet werden.")
                result['success'] = True
                result['message'] = "Telegram ist bereits autorisiert."
                return result

            await self.client.sign_in(phone=self.phone, code=code)
            success_message = "Telegram-Login erfolgreich abgeschlossen."
            self.log(success_message)
            result['success'] = True
            result['message'] = success_message
            return result
        except SessionPasswordNeededError:
            message = (
                "Telegram erfordert zusätzlich ein Passwort (2FA). "
                "Bitte geben Sie das Passwort ein, um den Login abzuschließen."
            )
            self.log(message, "ERROR")
            result['message'] = message
            result['require_password'] = True
        except (PhoneCodeInvalidError, PhoneCodeExpiredError):
            message = "Der eingegebene Telegram-Code ist ungültig oder abgelaufen."
            self.log(message, "ERROR")
            result['message'] = message
            await self._request_new_code_with_notification(
                "Der eingegebene Code war ungültig oder abgelaufen. Bitte geben Sie den neu gesendeten Code ein."
            )
        except Exception as e:
            message = f"Fehler bei der Telegram-Anmeldung: {e}"
            self.log(message, "ERROR")
            result['message'] = message
            await self._request_new_code_with_notification(
                "Der Telegram-Login ist fehlgeschlagen. Bitte geben Sie den neu gesendeten Code ein."
            )

        return result

    async def _request_new_code_with_notification(self, notify_message: Optional[str]) -> bool:
        """Neuen Login-Code anfordern und GUI benachrichtigen."""

        extra_info = ""
        success = False

        if not self.client:
            extra_info = "Telegram-Client nicht initialisiert. Neuer Code kann nicht angefordert werden."
            self.log(extra_info, "ERROR")
        elif not self.phone:
            extra_info = "Keine Telefonnummer hinterlegt. Neuer Code kann nicht angefordert werden."
            self.log(extra_info, "ERROR")
        elif not await self.ensure_connected():
            extra_info = "Telegram-Verbindung konnte nicht aufgebaut werden. Neuer Code wurde nicht angefordert."
            self.log(extra_info, "ERROR")
        else:
            try:
                await self.client.send_code_request(self.phone)
                extra_info = "Neuer Login-Code wurde angefordert. Bitte prüfen Sie Ihre Telegram-App."
                self.log(extra_info, "INFO")
                success = True
            except Exception as code_error:
                extra_info = f"Fehler beim erneuten Anfordern des Login-Codes: {code_error}"
                self.log(extra_info, "ERROR")

        if notify_message:
            message = notify_message
            if extra_info:
                message = f"{notify_message}\n\n{extra_info}"
            self.notify_auth_required(message)

        return success

    async def start(self) -> bool:
        """Bot starten"""

        if not self.client:
            self.log("Telegram-Konfiguration fehlt. Bitte führen Sie das Setup aus.", "ERROR")
            return False

        try:
            authorized = await self.ensure_authorized(
                request_code=True,
                notify_gui=True,
                message="Telegram-Login erforderlich. Bitte geben Sie den Login-Code ein, um den Bot zu starten."
            )
            if not authorized:
                return False

            @self.client.on(events.NewMessage)
            async def message_handler(event):
                await self.handle_new_message(event)

            self.is_running = True
            self.log("Bot gestartet - Multi-Chat-Modus aktiv")

            # Client im Hintergrund laufen lassen
            asyncio.create_task(self.client.run_until_disconnected())
            return True

        except Exception as e:
            self.log(f"Fehler beim Starten: {e}", "ERROR")
            return False

    async def handle_new_message(self, event):
        """Neue Nachricht verarbeiten"""
        try:
            chat_id = event.chat_id
            message_text = event.message.message or ""

            # Chat-Quelle prüfen
            chat_source = self.chat_manager.get_chat_info(chat_id)
            if not chat_source or not chat_source.enabled:
                return

            self.log(f"Signal von '{chat_source.chat_name}': {message_text[:80]}...")

            # Signal-Statistiken aktualisieren
            self.chat_manager.update_signal_stats(chat_id)

            # Signal verarbeiten
            signal = await self.signal_processor.process_signal(message_text, chat_source)

            if signal:
                kind = signal.get('kind', 'trade')
                if kind == 'trade':
                    signal_type = signal.get('type', 'instant')
                    action = (signal.get('action') or '').upper()
                    symbol = signal.get('symbol', 'Unbekannt')

                    if signal_type == 'instant' and not self.instant_trading_enabled:
                        self.log(
                            (
                                f"Sofort-Trading deaktiviert - "
                                f"{action or 'Signal'} {symbol} von {chat_source.chat_name} ignoriert."
                            ),
                            "INFO"
                        )
                        return

                    if signal_type == 'zone' and not self.zone_trading_enabled:
                        self.log(
                            (
                                f"Zonen-Trading deaktiviert - "
                                f"{action or 'Signal'} {symbol} von {chat_source.chat_name} ignoriert."
                            ),
                            "INFO"
                        )
                        return

                    execution_mode = self._determine_execution_mode(signal)
                    if execution_mode == ExecutionMode.DISABLED:
                        self.log(
                            f"Trading deaktiviert - Signal von {chat_source.chat_name} ignoriert.",
                            "INFO"
                        )
                        return

                    if self.require_confirmation:
                        self.log(
                            (
                                f"Bestätigung erforderlich für Signal aus {chat_source.chat_name}: "
                                f"{action or 'TRADE'} {symbol}."
                            ),
                            "INFO"
                        )
                        confirmed = await self.request_trade_confirmation(
                            signal,
                            chat_source,
                            message_text
                        )
                        if not confirmed:
                            self.log(
                                f"Signal von {chat_source.chat_name} abgelehnt.",
                                "INFO"
                            )
                            return
                        else:
                            self.log(
                                f"Signal von {chat_source.chat_name} bestätigt. Ausführung startet.",
                                "INFO"
                            )

                    trade_result = await self.execute_signal(signal, chat_source, message_text)
                    if trade_result and trade_result.get('status') == 'executed':
                        auto_levels = getattr(self.signal_processor, 'auto_tp_sl', True)
                        pending_info = {
                            'ticket': trade_result['ticket'],
                            'symbol': trade_result['symbol'],
                            'awaiting_sl': auto_levels and signal.get('stop_loss') is None,
                            'awaiting_tp': auto_levels and not signal.get('take_profits'),
                            'timestamp': datetime.now()
                        }
                        if pending_info['awaiting_sl'] or pending_info['awaiting_tp']:
                            self.pending_trade_updates[chat_id] = pending_info
                        else:
                            self.pending_trade_updates.pop(chat_id, None)
                    elif trade_result:
                        self.pending_trade_updates.pop(chat_id, None)
                elif kind == 'update':
                    await self.apply_trade_update(chat_source, signal)

        except Exception as e:
            self.log(f"Fehler bei Nachrichtenverarbeitung: {e}", "ERROR")

    def _executed_trades_within(self, duration: timedelta) -> int:
        """Anzahl ausgeführter Trades innerhalb eines Zeitfensters zählen."""
        cutoff = datetime.now() - duration
        return sum(
            1
            for record in self.trade_tracker.trade_records.values()
            if record.timestamp >= cutoff and record.status == 'executed'
        )

    def _extract_signal_spread(self, signal: Dict) -> Optional[float]:
        """Spread-Information aus dem Signal extrahieren."""
        for key in ('spread', 'spread_pips', 'current_spread'):
            if key in signal:
                try:
                    value = signal[key]
                    if isinstance(value, str):
                        value = value.replace(',', '.').strip()
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    def _determine_execution_mode(self, signal: Dict) -> ExecutionMode:
        """Ermittelt den effektiven Ausführungsmodus für ein Signal."""
        mode = signal.get('execution_mode')
        if isinstance(mode, ExecutionMode):
            return mode
        if isinstance(mode, str):
            try:
                return ExecutionMode(mode)
            except ValueError:
                pass
        return self.execution_mode

    async def request_trade_confirmation(
        self,
        signal: Dict,
        chat_source: ChatSource,
        original_message: str
    ) -> bool:
        """Fragt die GUI nach einer Trade-Bestätigung und wartet auf die Antwort."""

        confirmation_future: Future = Future()
        try:
            self.send_message(
                'CONFIRM_TRADE',
                {
                    'signal': signal,
                    'chat_name': chat_source.chat_name,
                    'chat_id': chat_source.chat_id,
                    'message': original_message,
                    'future': confirmation_future
                }
            )
        except Exception as exc:
            self.log(f"Bestätigungsanfrage konnte nicht gesendet werden: {exc}", "ERROR")
            return False

        try:
            result = await asyncio.wrap_future(confirmation_future)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log(f"Fehler beim Warten auf die Trade-Bestätigung: {exc}", "ERROR")
            return False
        return bool(result)

    async def execute_signal(self, signal: Dict, chat_source: ChatSource, original_message: str):
        """Signal ausführen (Demo oder Live)"""
        try:
            execution_mode = self._determine_execution_mode(signal)
            entry_price_raw = signal.get('entry_price')
            try:
                entry_price = float(entry_price_raw) if entry_price_raw is not None else None
            except (TypeError, ValueError):
                entry_price = None

            stop_loss_value = signal.get('stop_loss')
            stop_loss = float(stop_loss_value) if stop_loss_value is not None else 0.0
            take_profits_raw = signal.get('take_profits') or []
            normalized_tps = []
            for tp in take_profits_raw:
                try:
                    normalized_tps.append(float(tp))
                except (TypeError, ValueError):
                    continue
            take_profit = normalized_tps[0] if normalized_tps else (
                float(signal.get('take_profit', 0.0)) if signal.get('take_profit') else 0.0
            )

            lot_size = float(self.default_lot_size or 0.0)
            if lot_size <= 0:
                self.log(
                    "Ungültige Standard-Lotgröße konfiguriert. Signal wird übersprungen.",
                    "WARNING"
                )
                return None

            base_trade_info = {
                'symbol': signal['symbol'],
                'direction': signal['action'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'take_profits': normalized_tps,
                'lot_size': lot_size
            }

            if execution_mode == ExecutionMode.DISABLED:
                self.log(
                    f"Ausführungsmodus 'Ausgeschaltet': Signal von {chat_source.chat_name} wird ignoriert.",
                    "INFO"
                )
                return None

            if execution_mode == ExecutionMode.ZONE_WAIT:
                pending_result = {
                    'ticket': f"PENDING_{int(datetime.now().timestamp())}",
                    'price': entry_price if entry_price is not None else 0.0,
                    'status': 'pending',
                    'profit_loss': 0.0,
                    **base_trade_info
                }
                self.trade_tracker.add_trade(pending_result, chat_source, original_message)
                self.log(
                    f"Signal als Pending markiert (Zone Monitoring): {signal['action']} {signal['symbol']}"
                )
                self.send_message('TRADE_EXECUTED', {
                    **pending_result,
                    'source': chat_source.chat_name,
                    'demo': True
                })
                return pending_result

            if self.max_trades_per_hour and self.max_trades_per_hour > 0:
                trades_last_hour = self._executed_trades_within(timedelta(hours=1))
                if trades_last_hour >= int(self.max_trades_per_hour):
                    self.log(
                        (
                            f"Signal von {chat_source.chat_name} ignoriert: "
                            f"Maximale Trade-Anzahl pro Stunde erreicht "
                            f"({trades_last_hour}/{self.max_trades_per_hour})."
                        ),
                        "WARNING"
                    )
                    return None

            max_spread_limit = float(self.max_spread_pips or 0.0)
            if max_spread_limit > 0.0:
                spread_value = self._extract_signal_spread(signal)
                if spread_value is not None and spread_value > max_spread_limit:
                    self.log(
                        (
                            f"Signal von {chat_source.chat_name} ignoriert: Spread {spread_value:.2f} Pips "
                            f"überschreitet das Limit von {max_spread_limit:.2f} Pips."
                        ),
                        "WARNING"
                    )
                    return None

            if self.demo_mode:
                default_price = 1.0850 if 'EUR' in signal['symbol'] else 2660.00
                price = entry_price if entry_price is not None else default_price
                demo_result = {
                    'ticket': f"DEMO_{int(datetime.now().timestamp())}",
                    'price': price,
                    'lot_size': lot_size,
                    'status': 'executed',
                    'profit_loss': 0.0,
                    **base_trade_info
                }

                # Trade zu Tracker hinzufügen
                self.trade_tracker.add_trade(demo_result, chat_source, original_message)

                self.log(f"DEMO-Trade ausgeführt: {signal['action']} {signal['symbol']} von {chat_source.chat_name}")

                # GUI benachrichtigen
                self.send_message('TRADE_EXECUTED', {
                    **demo_result,
                    'source': chat_source.chat_name,
                    'demo': True
                })
                return demo_result
            if not self.ensure_mt5_session():
                return None

            symbol = base_trade_info['symbol']
            direction = (base_trade_info['direction'] or '').upper()

            if direction not in {'BUY', 'SELL'}:
                self.log(f"Unbekannte Handelsrichtung '{direction}' für {symbol}.", "ERROR")
                return None

            try:
                if not mt5.symbol_select(symbol, True):
                    error = self._format_mt5_error(mt5.last_error())
                    self.log(f"Symbol {symbol} konnte nicht für den Handel aktiviert werden: {error}", "ERROR")
                    return None
            except Exception as exc:
                self.log(f"Symbol {symbol} konnte nicht vorbereitet werden: {exc}", "ERROR")
                return None

            try:
                symbol_info = mt5.symbol_info(symbol)
            except Exception:
                symbol_info = None
            if not symbol_info:
                self.log(f"Keine Symbolinformationen für {symbol} verfügbar.", "ERROR")
                return None

            trade_mode_disabled = getattr(mt5, 'SYMBOL_TRADE_MODE_DISABLED', None)
            if trade_mode_disabled is not None and getattr(symbol_info, 'trade_mode', None) == trade_mode_disabled:
                self.log(f"Symbol {symbol} ist zum Handel deaktiviert.", "ERROR")
                return None

            try:
                tick_info = mt5.symbol_info_tick(symbol)
            except Exception:
                tick_info = None
            if not tick_info:
                self.log(f"Tick-Daten für {symbol} konnten nicht abgerufen werden.", "ERROR")
                return None

            order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
            market_price = tick_info.ask if order_type == mt5.ORDER_TYPE_BUY else tick_info.bid
            if market_price is None or market_price <= 0:
                self.log(f"Ungültige Marktdaten für {symbol}.", "ERROR")
                return None
            if entry_price is not None and entry_price > 0:
                price = float(entry_price)
            else:
                price = float(market_price)

            filling_mode = getattr(symbol_info, 'filling_mode', None)
            if filling_mode is None:
                filling_mode = getattr(mt5, 'ORDER_FILLING_IOC', 0)
            else:
                try:
                    filling_mode = int(filling_mode)
                except (TypeError, ValueError):
                    filling_mode = getattr(mt5, 'ORDER_FILLING_IOC', 0)

            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': float(lot_size),
                'type': order_type,
                'price': price,
                'sl': float(stop_loss) if stop_loss else 0.0,
                'tp': float(take_profit) if take_profit else 0.0,
                'deviation': 20,
                'magic': 0,
                'comment': f"Telegram {chat_source.chat_name}"[:31],
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': filling_mode
            }

            try:
                result = mt5.order_send(request)
            except Exception as exc:
                self.log(f"MT5-Order konnte nicht gesendet werden: {exc}", "ERROR")
                return None

            if result is None:
                error = self._format_mt5_error(mt5.last_error())
                self.log(f"MT5 lieferte kein Ergebnis für Order {symbol}: {error}", "ERROR")
                return None

            success_codes = set()
            for name in ('TRADE_RETCODE_DONE', 'TRADE_RETCODE_DONE_PARTIAL', 'TRADE_RETCODE_PLACED'):
                if hasattr(mt5, name):
                    success_codes.add(getattr(mt5, name))
            if not success_codes and hasattr(mt5, 'TRADE_RETCODE_DONE'):
                success_codes.add(getattr(mt5, 'TRADE_RETCODE_DONE'))

            if result.retcode not in success_codes:
                error_text = result.comment or self._format_mt5_error(mt5.last_error())
                self.log(
                    f"MT5-Order für {symbol} abgelehnt (Retcode {result.retcode}): {error_text}",
                    "ERROR"
                )
                return None

            ticket_value = getattr(result, 'order', 0) or getattr(result, 'deal', 0)
            if not ticket_value:
                ticket_value = f"LIVE_{int(datetime.now().timestamp())}"

            executed_price = getattr(result, 'price', 0.0) or float(price)
            live_result = {
                'ticket': str(ticket_value),
                'price': executed_price,
                'lot_size': lot_size,
                'status': 'executed',
                'profit_loss': 0.0,
                **base_trade_info
            }

            self.trade_tracker.add_trade(live_result, chat_source, original_message)
            self.log(
                f"LIVE-Trade ausgeführt: {direction} {symbol} (Ticket {live_result['ticket']})",
                "INFO"
            )

            self.send_message('TRADE_EXECUTED', {
                **live_result,
                'source': chat_source.chat_name,
                'demo': False
            })
            return live_result

        except Exception as e:
            self.log(f"Fehler bei Signal-Ausführung: {e}", "ERROR")
        return None

    async def apply_trade_update(self, chat_source: ChatSource, update_signal: Dict):
        """Offene Trades mit SL/TP aus nachfolgenden Nachrichten aktualisieren"""
        pending = self.pending_trade_updates.get(chat_source.chat_id)

        target_ticket = None
        if pending:
            target_ticket = pending['ticket']
        else:
            last_trade = self.trade_tracker.get_last_trade_for_chat(chat_source.chat_id)
            if last_trade:
                target_ticket = last_trade.ticket

        if not target_ticket:
            self.log(f"Keine offene Order für Update aus {chat_source.chat_name} gefunden.", "WARNING")
            return

        stop_loss = update_signal.get('stop_loss')
        take_profits = update_signal.get('take_profits') or []

        record = self.trade_tracker.update_trade_levels(
            target_ticket,
            stop_loss=stop_loss,
            take_profits=take_profits
        )

        if not record:
            self.log(f"Trade {target_ticket} konnte nicht aktualisiert werden.", "WARNING")
            return

        def _safe_float(value):
            if value is None:
                return None
            try:
                if isinstance(value, str):
                    value = value.replace(',', '.')
                return float(value)
            except (TypeError, ValueError):
                return None

        mt5_success = True
        if not self.demo_mode:
            mt5_success = False
            if not self.ensure_mt5_session():
                self.log(
                    f"MT5-Session für Update von Ticket {record.ticket} nicht verfügbar.",
                    "ERROR"
                )
            else:
                ticket_int = None
                ticket_value = str(record.ticket).strip()
                try:
                    ticket_int = int(ticket_value)
                except (TypeError, ValueError):
                    try:
                        ticket_int = int(float(ticket_value))
                    except (TypeError, ValueError):
                        self.log(
                            f"Ticket {record.ticket} ist keine gültige MT5-Ticketnummer.",
                            "ERROR"
                        )

                position = None
                if ticket_int is not None:
                    try:
                        positions = mt5.positions_get(ticket=ticket_int)
                    except Exception as exc:
                        self.log(
                            f"MT5-Positionen für Ticket {record.ticket} konnten nicht abgerufen werden: {exc}",
                            "ERROR"
                        )
                        positions = None
                    if positions:
                        position = positions[0]
                    else:
                        self.log(
                            f"Keine offene MT5-Position für Ticket {record.ticket} gefunden.",
                            "ERROR"
                        )

                if position is not None:
                    modify_action = getattr(mt5, 'TRADE_ACTION_SLTP', None) or getattr(mt5, 'TRADE_ACTION_MODIFY', None)
                    if modify_action is None:
                        self.log(
                            "MT5 unterstützt keine Aktualisierung von Stop-Loss/Take-Profit.",
                            "ERROR"
                        )
                    else:
                        sl_value = _safe_float(record.stop_loss)
                        tp_value = None
                        if take_profits:
                            tp_value = _safe_float(take_profits[0])
                        if tp_value is None:
                            tp_value = _safe_float(record.take_profit)

                        if sl_value is None and tp_value is None:
                            self.log(
                                f"Keine gültigen SL/TP-Werte für Ticket {record.ticket} vorhanden.",
                                "ERROR"
                            )
                        else:
                            current_sl = _safe_float(getattr(position, 'sl', None)) or 0.0
                            current_tp = _safe_float(getattr(position, 'tp', None)) or 0.0
                            request = {
                                'action': modify_action,
                                'position': getattr(position, 'ticket', ticket_int),
                                'symbol': getattr(position, 'symbol', record.symbol),
                                'sl': float(sl_value) if sl_value is not None else float(current_sl),
                                'tp': float(tp_value) if tp_value is not None else float(current_tp),
                                'magic': getattr(position, 'magic', 0),
                                'comment': f"Telegram Update {chat_source.chat_name}"[:31]
                            }

                            try:
                                result = mt5.order_send(request)
                            except Exception as exc:
                                self.log(
                                    f"MT5-Update für Ticket {record.ticket} konnte nicht gesendet werden: {exc}",
                                    "ERROR"
                                )
                                result = None

                            if result is None:
                                error = self._format_mt5_error(mt5.last_error())
                                self.log(
                                    f"MT5 lieferte kein Ergebnis für Update von Ticket {record.ticket}: {error}",
                                    "ERROR"
                                )
                            else:
                                success_codes = set()
                                for name in ('TRADE_RETCODE_DONE', 'TRADE_RETCODE_DONE_PARTIAL', 'TRADE_RETCODE_PLACED'):
                                    if hasattr(mt5, name):
                                        success_codes.add(getattr(mt5, name))
                                if not success_codes and hasattr(mt5, 'TRADE_RETCODE_DONE'):
                                    success_codes.add(getattr(mt5, 'TRADE_RETCODE_DONE'))

                                if result.retcode not in success_codes:
                                    error_text = result.comment or self._format_mt5_error(mt5.last_error())
                                    self.log(
                                        f"MT5-Update für Ticket {record.ticket} abgelehnt (Retcode {result.retcode}): {error_text}",
                                        "ERROR"
                                    )
                                else:
                                    mt5_success = True
                                    self.log(
                                        f"MT5-Position {record.ticket} aktualisiert: SL={request['sl']:.2f} TP={request['tp']:.2f}",
                                        "INFO"
                                    )

        if mt5_success:
            sl_display = _safe_float(record.stop_loss)
            tp_display = _safe_float(record.take_profit)
            sl_text = f"{sl_display:.2f}" if sl_display is not None else "-"
            tp_text = f"{tp_display:.2f}" if tp_display is not None else "-"

            self.log(
                f"Trade {record.ticket} aktualisiert: SL={sl_text} TP={tp_text}",
                "INFO"
            )

            self.send_message('TRADE_UPDATED', {
                'ticket': record.ticket,
                'symbol': record.symbol,
                'stop_loss': record.stop_loss,
                'take_profit': record.take_profit,
                'take_profits': take_profits,
                'source': chat_source.chat_name
            })

            if pending:
                if stop_loss is not None:
                    pending['awaiting_sl'] = False
                if take_profits:
                    pending['awaiting_tp'] = False
                if not pending['awaiting_sl'] and not pending['awaiting_tp']:
                    self.pending_trade_updates.pop(chat_source.chat_id, None)

    async def load_all_chats(self):
        """Alle verfügbaren Chats laden"""
        chats_data = []
        if not self.client:
            self.log("Telegram-Konfiguration fehlt. Chats können nicht geladen werden.", "ERROR")
            return chats_data

        try:
            authorized = await self.ensure_authorized(
                request_code=True,
                notify_gui=True,
                message=(
                    "Telegram-Login erforderlich. Ein neuer Login-Code wurde angefordert. "
                    "Bitte geben Sie ihn ein, um die Chats zu laden."
                )
            )
            if not authorized:
                return chats_data

            async for dialog in self.client.iter_dialogs(limit=200):
                chat_info = {
                    'id': dialog.id,
                    'name': dialog.name or "Unbekannt",
                    'type': 'channel' if dialog.is_channel else 'group' if dialog.is_group else 'private',
                    'participants': getattr(dialog.entity, 'participants_count', 0),
                    'archived': getattr(dialog, 'archived', False)
                }
                chats_data.append(chat_info)
        except Exception as e:
            self.log(f"Fehler beim Laden der Chats: {e}", "ERROR")
        return chats_data

    def log(self, message: str, level: str = "INFO"):
        """Log-Nachricht"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {level}: {message}"
        print(log_msg)
        self.send_message('LOG', log_msg)

    def send_message(self, msg_type: str, data):
        """Nachricht an GUI senden"""
        try:
            self.message_queue.put((msg_type, data), block=False)
        except queue.Full:
            pass

    @staticmethod
    def _format_mt5_error(error) -> str:
        """Formatiert eine MT5-Fehlermeldung."""
        if isinstance(error, tuple):
            if len(error) >= 2:
                return f"{error[0]}: {error[1]}"
            if error:
                return str(error[0])
        return str(error)

    def _enforce_demo_mode(self, reason: str):
        """Erzwingt den Demo-Modus und informiert die GUI."""
        if reason:
            self.log(reason, "ERROR")
        if not self.demo_mode:
            self.demo_mode = True
            self.send_message('DEMO_MODE_ENFORCED', {'message': reason})

    def update_mt5_credentials(
        self,
        login: Optional[object],
        password: Optional[str],
        server: Optional[str],
        path: Optional[str] = None
    ):
        """Aktualisiert die MT5-Zugangsdaten."""
        if not MT5_AVAILABLE:
            self.mt5_login = None
            self.mt5_password = None
            self.mt5_server = None
            self.mt5_path = None
            self._mt5_initialized = False
            self._mt5_login_ok = False
            self._last_mt5_error = "MetaTrader5 ist nicht installiert oder verfügbar."
            return

        try:
            self.mt5_login = int(login) if login not in (None, "", False) else None
        except (TypeError, ValueError):
            self.mt5_login = None

        self.mt5_password = password or None
        self.mt5_server = server or None
        self.mt5_path = path or None
        self._mt5_login_ok = False
        self._last_mt5_error = None

        if self._mt5_initialized:
            try:
                mt5.shutdown()
            except Exception:
                pass
            self._mt5_initialized = False

    def ensure_mt5_session(self, enforce_demo_on_fail: bool = True) -> bool:
        """Stellt sicher, dass eine aktive MT5-Session verfügbar ist."""
        self._last_mt5_error = None

        def handle_failure(message: str) -> bool:
            self._last_mt5_error = message
            if enforce_demo_on_fail:
                self._enforce_demo_mode(message)
            else:
                self.log(message, "ERROR")
            return False

        if not MT5_AVAILABLE:
            return handle_failure("MetaTrader5-Bibliothek nicht verfügbar. Live-Modus deaktiviert.")

        if not self.mt5_login or not self.mt5_password or not self.mt5_server:
            return handle_failure("MT5-Zugangsdaten unvollständig. Bitte prüfen Sie Login, Passwort und Server.")

        if self._mt5_initialized:
            try:
                if mt5.terminal_info() is None:
                    self._mt5_initialized = False
                    self._mt5_login_ok = False
            except Exception:
                self._mt5_initialized = False
                self._mt5_login_ok = False

        if not self._mt5_initialized:
            try:
                init_kwargs = {}
                if self.mt5_path:
                    init_kwargs['path'] = self.mt5_path
                if not mt5.initialize(**init_kwargs):
                    error = self._format_mt5_error(mt5.last_error())
                    message = f"MT5-Initialisierung fehlgeschlagen: {error}"
                    return handle_failure(message)
            except Exception as exc:
                message = f"MT5 konnte nicht initialisiert werden: {exc}"
                return handle_failure(message)
            self._mt5_initialized = True

        try:
            account_info = mt5.account_info()
        except Exception:
            account_info = None

        if not account_info or account_info.login != self.mt5_login:
            try:
                if not mt5.login(self.mt5_login, password=self.mt5_password, server=self.mt5_server):
                    error = self._format_mt5_error(mt5.last_error())
                    message = f"MT5-Login fehlgeschlagen: {error}"
                    return handle_failure(message)
            except Exception as exc:
                message = f"MT5-Login nicht möglich: {exc}"
                return handle_failure(message)
            try:
                account_info = mt5.account_info()
            except Exception:
                account_info = None

        if not account_info or account_info.login != self.mt5_login:
            return handle_failure("MT5-Account konnte nicht bestätigt werden.")

        self._mt5_login_ok = True
        return True

    def get_last_mt5_error(self) -> Optional[str]:
        """Gibt die letzte MT5-Fehlermeldung zurück."""
        return self._last_mt5_error


# ==================== KONFIGURATION ====================

class ConfigManager:
    """Konfigurationsverwaltung"""

    def __init__(self):
        self.config_file = "trading_config.json"
        self.default_config = {
            "telegram": {
                "api_id": "",
                "api_hash": "",
                "phone": "",
                "session_name": "trading_session",
                "prompt_credentials_on_start": False
            },
            "trading": {
                "demo_mode": True,
                "execution_mode": ExecutionMode.INSTANT.value,
                "default_lot_size": 0.01,
                "max_spread_pips": 3.0,
                "risk_percent": 2.0,
                "max_trades_per_hour": 5
            },
            "mt5": {
                "path": "",
                "login": "",
                "password": "",
                "server": ""
            },
            "signals": {
                "instant_trading_enabled": True,
                "zone_trading_enabled": True,
                "require_confirmation": True,
                "auto_tp_sl": True
            }
        }

    def load_config(self):
        """Konfiguration laden"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.save_config(self.default_config)
                return self.default_config
        except Exception as e:
            print(f"Fehler beim Laden der Konfiguration: {e}")
            return self.default_config

    def save_config(self, config):
        """Konfiguration speichern"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Fehler beim Speichern der Konfiguration: {e}")


# ==================== GUI ====================


class ApiCredentialDialog(simpledialog.Dialog):
    """Dialog zum Abfragen der Telegram-API-Zugangsdaten."""

    def __init__(self, parent: tk.Tk, current_values: Optional[Dict]):
        self.current_values = current_values or {}
        self.api_id_var = tk.StringVar(value=str(self.current_values.get('api_id', "")))
        self.api_hash_var = tk.StringVar(value=str(self.current_values.get('api_hash', "")))
        self.phone_var = tk.StringVar(value=str(self.current_values.get('phone', "")))
        self.result: Optional[Dict[str, str]] = None
        self._validated_data: Optional[Dict[str, str]] = None
        super().__init__(parent, title="Telegram API Zugangsdaten")

    def body(self, master):  # type: ignore[override]
        master.columnconfigure(1, weight=1)

        ttk.Label(
            master,
            text=(
                "Bitte geben Sie die Telegram API-Zugangsdaten ein, bevor der Bot gestartet wird.\n"
                "Diese Informationen erhalten Sie unter https://my.telegram.org."
            ),
            wraplength=420,
            justify='left'
        ).grid(row=0, column=0, columnspan=2, sticky='w', padx=10, pady=(10, 12))

        ttk.Label(master, text="API ID:").grid(row=1, column=0, sticky='e', padx=(10, 5), pady=(0, 8))
        api_id_entry = ttk.Entry(master, textvariable=self.api_id_var)
        api_id_entry.grid(row=1, column=1, sticky='we', padx=(0, 10), pady=(0, 8))

        ttk.Label(master, text="API Hash:").grid(row=2, column=0, sticky='e', padx=(10, 5), pady=(0, 8))
        api_hash_entry = ttk.Entry(master, textvariable=self.api_hash_var)
        api_hash_entry.grid(row=2, column=1, sticky='we', padx=(0, 10), pady=(0, 8))

        ttk.Label(master, text="Telefonnummer (+49...):").grid(row=3, column=0, sticky='e', padx=(10, 5), pady=(0, 8))
        phone_entry = ttk.Entry(master, textvariable=self.phone_var)
        phone_entry.grid(row=3, column=1, sticky='we', padx=(0, 10), pady=(0, 12))

        return api_id_entry

    def validate(self) -> bool:  # type: ignore[override]
        api_id = (self.api_id_var.get() or "").strip()
        api_hash = (self.api_hash_var.get() or "").strip()
        phone = (self.phone_var.get() or "").strip()

        if not api_id or not api_hash or not phone:
            messagebox.showerror(
                "Fehlende Angaben",
                "Bitte füllen Sie alle Felder aus, um den Bot zu starten.",
                parent=self
            )
            return False

        if not api_id.isdigit():
            messagebox.showerror(
                "Ungültige API ID",
                "Die API ID darf nur Ziffern enthalten.",
                parent=self
            )
            return False

        self._validated_data = {
            'api_id': api_id,
            'api_hash': api_hash,
            'phone': phone
        }
        return True

    def apply(self):  # type: ignore[override]
        if self._validated_data:
            self.result = self._validated_data


class AuthCodeDialog(simpledialog.Dialog):
    """Einfache Dialogbox zur Eingabe des Telegram-Login-Codes."""

    def __init__(self, parent: tk.Tk, message: str):
        self.message = message
        self.code: Optional[str] = None
        super().__init__(parent, title="Telegram-Login erforderlich")

    def body(self, master):  # type: ignore[override]
        master.columnconfigure(1, weight=1)

        ttk.Label(
            master,
            text=self.message,
            wraplength=360,
            justify='left'
        ).grid(row=0, column=0, columnspan=2, sticky='w', padx=10, pady=(10, 10))

        ttk.Label(master, text="Login-Code:").grid(
            row=1, column=0, sticky='e', padx=(10, 5), pady=(0, 10)
        )

        self.code_entry = ttk.Entry(master)
        self.code_entry.grid(row=1, column=1, sticky='we', padx=(0, 10), pady=(0, 10))
        return self.code_entry

    def apply(self):  # type: ignore[override]
        self.code = (self.code_entry.get() or "").strip()


class TradingGUI:
    """Haupt-GUI für Multi-Chat-Trading"""

    MT5_INVALID_LOGIN_MESSAGE = "MT5-Login ungültig. Bitte geben Sie eine numerische Kontonummer ein."

    def __init__(self, config: Optional[Dict] = None):
        self.root = tk.Tk()
        self.root.title("Multi-Chat Trading Bot (Windows)")
        self.root.geometry("1200x800")

        # Style & Theme
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use('clam')
        except tk.TclError:
            pass
        self._configure_styles()

        # Konfiguration & Ausführungsmodi
        self.config_manager = ConfigManager()
        self.current_config: Dict = config or self.config_manager.load_config()
        self.execution_mode_labels: Dict[ExecutionMode, str] = {
            ExecutionMode.INSTANT: "Sofortausführung",
            ExecutionMode.ZONE_WAIT: "Zone Monitoring",
            ExecutionMode.DISABLED: "Ausgeschaltet"
        }
        self.execution_mode_label_to_enum: Dict[str, ExecutionMode] = {
            label: mode for mode, label in self.execution_mode_labels.items()
        }

        trading_defaults = self.config_manager.default_config.get('trading', {})
        initial_trading_cfg = self.current_config.get('trading', {})
        self._updating_trading_vars = False
        self.default_lot_var = tk.DoubleVar(
            master=self.root,
            value=self._coerce_to_float(
                initial_trading_cfg.get('default_lot_size'),
                trading_defaults.get('default_lot_size', 0.01)
            )
        )
        self.max_spread_var = tk.DoubleVar(
            master=self.root,
            value=self._coerce_to_float(
                initial_trading_cfg.get('max_spread_pips'),
                trading_defaults.get('max_spread_pips', 0.0)
            )
        )
        self.risk_percent_var = tk.DoubleVar(
            master=self.root,
            value=self._coerce_to_float(
                initial_trading_cfg.get('risk_percent'),
                trading_defaults.get('risk_percent', 0.0)
            )
        )
        self.max_trades_per_hour_var = tk.IntVar(
            master=self.root,
            value=self._coerce_to_int(
                initial_trading_cfg.get('max_trades_per_hour'),
                trading_defaults.get('max_trades_per_hour', 0)
            )
        )
        self.demo_var = tk.BooleanVar(
            master=self.root,
            value=self._coerce_to_bool(
                initial_trading_cfg.get('demo_mode'),
                trading_defaults.get('demo_mode', True)
            )
        )
        self._float_validate_cmd = self.root.register(self._validate_float_value)
        self._int_validate_cmd = self.root.register(self._validate_int_value)
        self._trading_var_map = {
            'default_lot_size': (self.default_lot_var, float),
            'max_spread_pips': (self.max_spread_var, float),
            'risk_percent': (self.risk_percent_var, float),
            'max_trades_per_hour': (self.max_trades_per_hour_var, int)
        }

        signals_defaults = self.config_manager.default_config.get('signals', {})
        initial_signals_cfg = self.current_config.get('signals', {})
        self._updating_signal_flags = False
        self.instant_trading_var = tk.BooleanVar(
            master=self.root,
            value=self._coerce_to_bool(
                initial_signals_cfg.get('instant_trading_enabled'),
                signals_defaults.get('instant_trading_enabled', True)
            )
        )
        self.zone_trading_var = tk.BooleanVar(
            master=self.root,
            value=self._coerce_to_bool(
                initial_signals_cfg.get('zone_trading_enabled'),
                signals_defaults.get('zone_trading_enabled', True)
            )
        )
        self.require_confirmation_var = tk.BooleanVar(
            master=self.root,
            value=self._coerce_to_bool(
                initial_signals_cfg.get('require_confirmation'),
                signals_defaults.get('require_confirmation', True)
            )
        )
        self.auto_tp_sl_var = tk.BooleanVar(
            master=self.root,
            value=self._coerce_to_bool(
                initial_signals_cfg.get('auto_tp_sl'),
                signals_defaults.get('auto_tp_sl', True)
            )
        )
        self._signal_flag_vars = {
            'instant_trading_enabled': self.instant_trading_var,
            'zone_trading_enabled': self.zone_trading_var,
            'require_confirmation': self.require_confirmation_var,
            'auto_tp_sl': self.auto_tp_sl_var
        }
        self._signal_flag_labels = {
            'instant_trading_enabled': 'Sofort-Trading',
            'zone_trading_enabled': 'Zonen-Trading',
            'require_confirmation': 'Bestätigungspflicht',
            'auto_tp_sl': 'Automatische SL/TP-Erkennung'
        }

        mt5_defaults = self.config_manager.default_config.get('mt5', {})
        initial_mt5_cfg = self.current_config.get('mt5', {})
        self.mt5_login_var = tk.StringVar(
            master=self.root,
            value=self._coerce_to_str(initial_mt5_cfg.get('login', mt5_defaults.get('login', "")))
        )
        self.mt5_password_var = tk.StringVar(
            master=self.root,
            value=self._coerce_to_str(initial_mt5_cfg.get('password', mt5_defaults.get('password', "")))
        )
        self.mt5_server_var = tk.StringVar(
            master=self.root,
            value=self._coerce_to_str(initial_mt5_cfg.get('server', mt5_defaults.get('server', "")))
        )
        self.mt5_path_var = tk.StringVar(
            master=self.root,
            value=self._coerce_to_str(initial_mt5_cfg.get('path', mt5_defaults.get('path', "")))
        )
        self.mt5_status_card: Optional[ttk.Frame] = None
        self.mt5_status_message_var: Optional[tk.StringVar] = None
        self.mt5_status_label: Optional[ttk.Label] = None

        # Dashboard & UI helper states
        self.hero_status_var: Optional[tk.StringVar] = None
        self.dashboard_state_var: Optional[tk.StringVar] = None
        self.dashboard_alert_var: Optional[tk.StringVar] = None
        self.open_signal_percentage_var: Optional[tk.StringVar] = None
        self.open_signal_active_var: Optional[tk.StringVar] = None
        self.dashboard_drawdown_var: Optional[tk.StringVar] = None
        self.dashboard_daily_loss_var: Optional[tk.StringVar] = None
        self.dashboard_compliance_var: Optional[tk.StringVar] = None
        self.chat_total_var: Optional[tk.StringVar] = None
        self.chat_active_var: Optional[tk.StringVar] = None
        self.chat_signal_sum_var: Optional[tk.StringVar] = None
        self.chat_last_sync_var: Optional[tk.StringVar] = None
        self.chat_summary_var: Optional[tk.StringVar] = None
        self.latency_var: Optional[tk.StringVar] = None
        self.alert_score_var: Optional[tk.StringVar] = None

        self.automation_rules_container: Optional[ttk.Frame] = None
        self.exposure_list_frame: Optional[ttk.Frame] = None

        self.chat_status_var: Optional[tk.StringVar] = None
        self.chats_listbox: Optional[tk.Listbox] = None
        self.refresh_chats_button: Optional[ttk.Button] = None
        self.save_chats_button: Optional[ttk.Button] = None
        self._chat_entries: List[Dict[str, object]] = []

        # Statistik- und Diagrammzustände
        self.stats_sharpe_var: Optional[tk.StringVar] = None
        self.stats_sortino_var: Optional[tk.StringVar] = None
        self.stats_drawdown_var: Optional[tk.StringVar] = None
        self.stats_win_rate_var: Optional[tk.StringVar] = None
        self.stats_profit_var: Optional[tk.StringVar] = None
        self.detail_winrate_var: Optional[tk.StringVar] = None
        self.detail_rr_var: Optional[tk.StringVar] = None
        self.detail_signals_var: Optional[tk.StringVar] = None
        self.detail_profit_var: Optional[tk.StringVar] = None
        self.chat_detail_title_var: Optional[tk.StringVar] = None
        self.chat_detail_latency_var: Optional[tk.StringVar] = None
        self.chat_detail_quality_var: Optional[tk.StringVar] = None
        self.chat_detail_risk_var: Optional[tk.StringVar] = None

        self.equity_curve_canvas: Optional[tk.Canvas] = None
        self.profit_distribution_canvas: Optional[tk.Canvas] = None
        self.monthly_heatmap_canvas: Optional[tk.Canvas] = None
        self.session_heatmap_canvas: Optional[tk.Canvas] = None
        self.pair_distribution_canvas: Optional[tk.Canvas] = None
        self.accuracy_gauge_canvas: Optional[tk.Canvas] = None

        self._default_equity_curve = self._generate_equity_curve_series()
        self._equity_curve_data = list(self._default_equity_curve)
        self._profit_distribution_data = self._generate_profit_distribution()
        self._monthly_profit_matrix = self._generate_monthly_profit_matrix()
        self._default_session_heatmap = self._generate_session_heatmap_matrix()
        self._default_pair_distribution = [('EUR/USD', 0.6), ('GBP/USD', 0.25), ('XAU/USD', 0.15)]
        self._current_session_heatmap = [row[:] for row in self._default_session_heatmap]
        self._current_pair_distribution = list(self._default_pair_distribution)
        self._current_accuracy_ratio = 0.7

        # Bot-Instanz (setzt später Config/Setup)
        self.bot = MultiChatTradingBot(None, None, None)
        self.bot_starting = False
        self._auth_dialog_open = False
        self._last_auth_message: Optional[str] = None
        self._pending_auth_message: Optional[str] = None

        for key, (var, cast) in self._trading_var_map.items():
            self._add_trading_var_trace(key, var, cast)

        # Buttons (werden in create_widgets gesetzt)
        self.start_button: Optional[ttk.Button] = None

        # GUI-Komponenten
        self.create_widgets()
        self.setup_message_processing()

        self.apply_config(self.current_config)

    def _configure_styles(self):
        """Globale Styles, Farben und Schriftarten setzen."""
        base_bg = '#050b19'
        surface_bg = '#0d1628'
        surface_alt = '#111f30'
        surface_soft = '#091223'
        accent_color = '#2563ff'
        accent_hover = '#1d4fe8'
        accent_light = '#1a2d52'
        accent_soft = '#1b2d4a'
        text_color = '#f8fafc'
        subtle_text = '#8ea1c7'
        border_color = '#1c2744'
        success_color = '#34d399'
        warning_color = '#f97316'
        danger_color = '#f87171'
        info_color = '#38bdf8'

        self.theme_colors: Dict[str, str] = {
            'base_bg': base_bg,
            'surface_bg': surface_bg,
            'surface_alt': surface_alt,
            'surface_soft': surface_soft,
            'accent': accent_color,
            'accent_hover': accent_hover,
            'accent_light': accent_light,
            'accent_soft': accent_soft,
            'text': text_color,
            'subtle_text': subtle_text,
            'border': border_color,
            'success': success_color,
            'warning': warning_color,
            'danger': danger_color,
            'info': info_color,
            'highlight': '#22d3ee'
        }

        self.root.configure(bg=base_bg)
        # Fonts with spaces in the family name must be wrapped in braces so that
        # Tk does not interpret the second word ("UI") as the font size.  When
        # this happens Tk raises the error "expected integer but got UI" during
        # startup, resulting in a blank window for the user.  By adding braces
        # we ensure the whole family name is passed correctly.
        self.root.option_add('*Font', '{Segoe UI} 10')
        self.root.option_add('*TCombobox*Listbox.Font', '{Segoe UI} 10')
        self.root.option_add('*TEntry.Font', '{Segoe UI} 10')

        # Standard-Schriftarten über die Tk-Font-Objekte setzen, damit Tk sie korrekt
        # interpretiert und alle Widgets (inkl. ttk) konsistent aktualisiert werden.
        default_font = tkfont.nametofont('TkDefaultFont')
        default_font.configure(family='Segoe UI', size=10)
        text_font = tkfont.nametofont('TkTextFont')
        text_font.configure(family='Segoe UI', size=10)

        self.root.option_add('*TCombobox*Listbox.font', text_font)
        self.style.configure('TEntry', font=default_font)
        self.style.configure('TCombobox', font=default_font)

        self.style.configure('TFrame', background=base_bg)
        self.style.configure('Main.TFrame', background=base_bg)
        self.style.configure('Header.TFrame', background=base_bg)
        self.style.configure('Toolbar.TFrame', background=surface_alt)
        self.style.configure('InfoBar.TFrame', background=surface_bg)
        self.style.configure('Metric.TFrame', background=surface_bg, relief='flat', borderwidth=0)
        self.style.configure('Card.TFrame', background=surface_bg, relief='flat', borderwidth=0)
        self.style.configure('GlassCard.TFrame', background=surface_alt, relief='flat', borderwidth=0)
        self.style.configure('Card.TLabelframe', background=surface_bg, relief='flat', borderwidth=1, bordercolor=border_color)
        self.style.configure('Card.TLabelframe.Label', background=surface_bg, foreground=text_color, font=('Segoe UI Semibold', 11))

        self.style.configure('Hero.TFrame', background='#101b33')
        self.style.configure('HeroTitle.TLabel', background='#101b33', foreground=text_color, font=('Segoe UI Semibold', 22))
        self.style.configure('HeroSubtitle.TLabel', background='#101b33', foreground=subtle_text, font=('Segoe UI', 11))
        self.style.configure('HeroTag.TLabel', background='#1a2f52', foreground=info_color, font=('Segoe UI Semibold', 10), padding=(12, 4))

        self.style.configure('Statusbar.TFrame', background=surface_alt)
        self.style.configure('Statusbar.TLabel', background=surface_alt, foreground=subtle_text, font=('Segoe UI', 10))

        self.style.configure('TNotebook', background=base_bg, borderwidth=0)
        self.style.configure('TNotebook.Tab', font=('Segoe UI Semibold', 10), padding=(20, 12), background=base_bg, foreground=subtle_text)
        self.style.map(
            'TNotebook.Tab',
            background=[('selected', surface_bg), ('!selected', base_bg)],
            foreground=[('selected', text_color), ('!selected', subtle_text)]
        )

        self.style.configure('TLabel', background=base_bg, foreground=text_color, font=('Segoe UI', 10))
        self.style.configure('Title.TLabel', background=base_bg, foreground=text_color, font=('Segoe UI Semibold', 18))
        self.style.configure('Subtitle.TLabel', background=base_bg, foreground=subtle_text, font=('Segoe UI', 11))
        self.style.configure('SectionTitle.TLabel', background=base_bg, foreground=text_color, font=('Segoe UI Semibold', 15))
        self.style.configure('Info.TLabel', background=base_bg, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('InfoBar.TLabel', background=surface_bg, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('FieldLabel.TLabel', background=base_bg, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('Warning.TLabel', background=base_bg, foreground=warning_color, font=('Segoe UI Semibold', 11))
        self.style.configure('Success.TLabel', background=base_bg, foreground=success_color, font=('Segoe UI Semibold', 11))
        self.style.configure('Danger.TLabel', background=base_bg, foreground=danger_color, font=('Segoe UI Semibold', 11))
        self.style.configure('MetricTitle.TLabel', background=surface_bg, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('MetricValue.TLabel', background=surface_bg, foreground=text_color, font=('Segoe UI Semibold', 18))
        self.style.configure('MetricPositive.TLabel', background=surface_alt, foreground=success_color, font=('Segoe UI Semibold', 18))
        self.style.configure('MetricNegative.TLabel', background=surface_alt, foreground=danger_color, font=('Segoe UI Semibold', 18))
        self.style.configure('CardTitle.TLabel', background=surface_bg, foreground=text_color, font=('Segoe UI Semibold', 13))
        self.style.configure('CardSubtitle.TLabel', background=surface_bg, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('GlassCardTitle.TLabel', background=surface_alt, foreground=text_color, font=('Segoe UI Semibold', 13))
        self.style.configure('GlassCardSubtitle.TLabel', background=surface_alt, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('InfoBadge.TLabel', background=accent_soft, foreground=info_color, font=('Segoe UI Semibold', 9), padding=(10, 3))
        self.style.configure('BadgeSuccess.TLabel', background='#0f2a1f', foreground=success_color, font=('Segoe UI Semibold', 10), padding=(10, 4))
        self.style.configure('BadgeWarning.TLabel', background='#321909', foreground=warning_color, font=('Segoe UI Semibold', 10), padding=(10, 4))
        self.style.configure('BadgeDanger.TLabel', background='#331316', foreground=danger_color, font=('Segoe UI Semibold', 10), padding=(10, 4))
        self.style.configure('BadgeInfo.TLabel', background='#112943', foreground=info_color, font=('Segoe UI Semibold', 10), padding=(10, 4))
        self.style.configure('CardIcon.TLabel', background=surface_bg, foreground=info_color, font=('Segoe UI', 16))
        self.style.configure('GlassCardIcon.TLabel', background=surface_alt, foreground=info_color, font=('Segoe UI', 16))

        self.style.configure('TButton', font=('Segoe UI', 10), padding=(16, 9), relief='flat', background=surface_alt, foreground=text_color, borderwidth=0)
        self.style.map('TButton', background=[('active', accent_light), ('pressed', accent_light)])
        self.style.configure('Accent.TButton', background=accent_color, foreground='#f8fafc', padding=(18, 10), relief='flat')
        self.style.map(
            'Accent.TButton',
            background=[('active', accent_hover), ('disabled', accent_soft)],
            foreground=[('disabled', '#4c5c7a')]
        )
        self.style.configure('Toolbar.TButton', background=surface_alt, foreground=text_color, padding=(12, 8))
        self.style.map('Toolbar.TButton', background=[('active', accent_light)], foreground=[('active', info_color)])
        self.style.configure('Link.TButton', background=base_bg, foreground=info_color, padding=0)
        self.style.map('Link.TButton', foreground=[('active', accent_hover)])

        self.style.configure('Treeview', background=surface_bg, fieldbackground=surface_bg, foreground=text_color, font=('Segoe UI', 10), rowheight=28, borderwidth=0, relief='flat')
        self.style.configure(
            'Treeview.Heading',
            background=surface_alt,
            foreground=subtle_text,
            font=('Segoe UI Semibold', 10),
            padding=8,
            relief='flat'
        )
        self.style.configure('Dashboard.Treeview', rowheight=30)
        self.style.map('Treeview', background=[('selected', accent_soft)], foreground=[('selected', info_color)])
        self.style.map('Treeview.Heading', background=[('active', accent_light)])

        self.style.configure('TCheckbutton', background=base_bg, foreground=text_color, font=('Segoe UI', 10))
        self.style.configure('Switch.TCheckbutton', background=base_bg, foreground=text_color, font=('Segoe UI', 10, 'bold'))
        self.style.map('Switch.TCheckbutton', foreground=[('selected', info_color)])

        entry_style = {
            'fieldbackground': surface_alt,
            'background': surface_alt,
            'foreground': text_color,
            'bordercolor': surface_alt,
            'relief': 'flat'
        }
        self.style.configure('TEntry', padding=8, **entry_style)
        self.style.configure('TCombobox', padding=8, **entry_style)
        self.style.map('TCombobox', fieldbackground=[('readonly', surface_alt)], selectbackground=[('readonly', surface_alt)], selectforeground=[('readonly', text_color)])

        self.style.configure('Accent.Horizontal.TProgressbar', background=accent_color, troughcolor=surface_soft, bordercolor=surface_soft, lightcolor=accent_color, darkcolor=accent_color, thickness=10)
        self.style.configure('Success.Horizontal.TProgressbar', background=success_color, troughcolor=surface_soft, bordercolor=surface_soft, lightcolor=success_color, darkcolor=success_color, thickness=10)
        self.style.configure('Warning.Horizontal.TProgressbar', background=warning_color, troughcolor=surface_soft, bordercolor=surface_soft, lightcolor=warning_color, darkcolor=warning_color, thickness=10)

        self.style.configure('StatusGood.TFrame', background='#0f2a21')
        self.style.configure('StatusAlert.TFrame', background='#32131a')
        self.style.configure('StatusGoodHeading.TLabel', background='#0f2a21', foreground=success_color, font=('Segoe UI Black', 22))
        self.style.configure('StatusAlertHeading.TLabel', background='#32131a', foreground=danger_color, font=('Segoe UI Black', 18))
        self.style.configure('StatusSub.TLabel', background='#0f2a21', foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('StatusAlertSub.TLabel', background='#32131a', foreground=subtle_text, font=('Segoe UI', 10))

        self.style.configure('Rule.TFrame', background=surface_bg)
        self.style.configure('RuleAccent.TFrame', background=surface_alt)
        self.style.configure('RuleTitle.TLabel', background=surface_bg, foreground=text_color, font=('Segoe UI Semibold', 11))
        self.style.configure('RuleSubtitle.TLabel', background=surface_bg, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('RulePillAccent.TLabel', background='#1c2f55', foreground=info_color, font=('Segoe UI Semibold', 9), padding=(10, 3))
        self.style.configure('RulePillNeutral.TLabel', background='#1b2336', foreground=text_color, font=('Segoe UI Semibold', 9), padding=(10, 3))
        self.style.configure('RulePillWarning.TLabel', background='#332116', foreground=warning_color, font=('Segoe UI Semibold', 9), padding=(10, 3))

    def create_widgets(self):
        """GUI-Widgets erstellen"""
        self.main_frame = ttk.Frame(self.root, padding=(24, 24, 24, 18), style='Main.TFrame')
        self.main_frame.pack(fill='both', expand=True)

        header_frame = ttk.Frame(self.main_frame, style='Hero.TFrame', padding=(34, 28))
        header_frame.pack(fill='x', pady=(0, 24))

        hero_header = ttk.Frame(header_frame, style='Hero.TFrame')
        hero_header.pack(fill='x')
        ttk.Label(hero_header, text="🛡 Risk Control Center", style='HeroTitle.TLabel').pack(side='left')

        hero_actions = ttk.Frame(hero_header, style='Hero.TFrame')
        hero_actions.pack(side='right')
        self.hero_status_var = tk.StringVar(value="SAFE")
        ttk.Label(hero_actions, textvariable=self.hero_status_var, style='BadgeSuccess.TLabel').pack(side='left', padx=(0, 16))
        ttk.Button(
            hero_actions,
            text="⚙ Bot Einstellungen",
            style='Link.TButton',
            command=self._open_bot_settings_from_header
        ).pack(side='left', padx=(0, 16))
        ttk.Button(
            hero_actions,
            text="🔔",
            style='Link.TButton',
            command=lambda: self.log_message("Benachrichtigungen folgen in einer späteren Version.")
        ).pack(side='left')

        ttk.Label(
            header_frame,
            text="Überwache deine Signalquellen, Automation-Rules & Risikoparameter in Echtzeit",
            style='HeroSubtitle.TLabel'
        ).pack(anchor='w', pady=(12, 0))

        tag_frame = ttk.Frame(header_frame, style='Hero.TFrame')
        tag_frame.pack(anchor='w', pady=(22, 0))
        for tag_text in ("Live Automation", "Compliance Guard", "Realtime Sync"):
            ttk.Label(tag_frame, text=tag_text, style='HeroTag.TLabel').pack(side='left', padx=(0, 12))

        ttk.Separator(self.main_frame).pack(fill='x', pady=(0, 16))

        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill='both', expand=True)

        self.create_chats_tab()
        self.create_chat_overview_tab()
        self.create_bot_settings_tab()
        self.create_mt5_settings_tab()
        self.create_statistics_tab()

        self.status_frame = ttk.Frame(self.main_frame, style='Statusbar.TFrame', padding=(18, 14))
        self.status_frame.pack(fill='x', pady=(18, 0))

        self.status_label = ttk.Label(self.status_frame, text="Bot gestoppt", style='Statusbar.TLabel')
        self.status_label.pack(side='left')

        button_frame = ttk.Frame(self.status_frame, style='Statusbar.TFrame')
        button_frame.pack(side='right')
        self.start_button = ttk.Button(
            button_frame,
            text="▶ Bot starten",
            command=self.start_bot,
            style='Accent.TButton'
        )
        self.start_button.pack(side='left', padx=(0, 12))
        ttk.Button(button_frame, text="■ Bot stoppen", command=self.stop_bot).pack(side='left')

        self._update_status_badges()

    def create_chats_tab(self):
        """Tab zum Aktualisieren und Auswählen der Telegram-Chats."""

        chats_tab = ttk.Frame(self.notebook, padding=(24, 24, 24, 20), style='Main.TFrame')
        self.chats_tab = chats_tab
        self.notebook.add(chats_tab, text="Chats")

        ttk.Label(chats_tab, text="Chats", style='SectionTitle.TLabel').pack(anchor='w')
        ttk.Label(
            chats_tab,
            text="Lade deine Dialoge aus Telegram und wähle die gewünschte Quelle aus.",
            style='Info.TLabel'
        ).pack(anchor='w', pady=(6, 12))

        list_container = ttk.Frame(chats_tab, style='Main.TFrame')
        list_container.pack(fill='both', expand=True)

        listbox_bg = self.theme_colors.get('surface_bg', '#151a22')
        listbox_fg = self.theme_colors.get('text', '#eef2f7')
        select_bg = self.theme_colors.get('accent_soft', '#1b2d4a')
        select_fg = self.theme_colors.get('text', '#eef2f7')

        self.chats_listbox = tk.Listbox(
            list_container,
            height=18,
            activestyle='none',
            exportselection=False,
            bg=listbox_bg,
            fg=listbox_fg,
            highlightthickness=0,
            relief='flat',
            selectbackground=select_bg,
            selectforeground=select_fg,
            font=('Segoe UI', 10)
        )
        self.chats_listbox.pack(fill='both', expand=True)

        toolbar = ttk.Frame(chats_tab, style='Toolbar.TFrame', padding=(18, 12))
        toolbar.pack(fill='x', pady=(12, 0))

        self.chat_status_var = tk.StringVar(master=self.root, value="Keine Chats geladen")
        ttk.Label(toolbar, textvariable=self.chat_status_var, style='InfoBar.TLabel').pack(side='left')

        self.refresh_chats_button = ttk.Button(
            toolbar,
            text="Refresh from Telegram",
            style='Toolbar.TButton',
            command=self._refresh_chats_from_telegram
        )
        self.refresh_chats_button.pack(side='right')

        self.save_chats_button = ttk.Button(
            toolbar,
            text="Save selection",
            style='Toolbar.TButton',
            command=self._save_selected_chat
        )
        self.save_chats_button.pack(side='right', padx=(0, 12))

        self._load_chats_from_file()

    def set_initial_page(self, page_key: str) -> None:
        """Select the requested notebook page when the UI starts."""
        if not hasattr(self, 'notebook'):
            return

        normalized = (page_key or '').strip().lower()
        target: Optional[object] = None

        if normalized in {'settings', 'bot', 'bot_settings'} and hasattr(self, 'bot_settings_tab'):
            target = self.bot_settings_tab
        elif normalized in {'chats', 'chat'} and hasattr(self, 'chats_tab'):
            target = self.chats_tab
        elif normalized in {'dashboard', 'home'} and hasattr(self, 'dashboard_tab'):
            target = self.dashboard_tab
        elif hasattr(self, 'dashboard_tab'):
            target = self.dashboard_tab

        if target is None:
            tabs = self.notebook.tabs()
            if tabs:
                target = tabs[0]

        if target is not None:
            try:
                self.notebook.select(target)
            except Exception:
                pass

    def _load_chats_from_file(self) -> None:
        """Chats aus chat_config.json in die Liste laden."""
        if not self.chats_listbox:
            return

        config_data, raw_entries = safe_load_chat_config()
        selected_identifier = None

        if isinstance(config_data, dict):
            selected_entry = self._normalize_chat_entry(config_data.get('selected_chat'))
            if selected_entry:
                selected_identifier = self._entry_identifier(selected_entry)

        entries: List[Dict[str, object]] = []
        if raw_entries:
            entries = [
                entry for entry in (self._normalize_chat_entry(item) for item in raw_entries) if entry
            ]
        elif isinstance(config_data, dict):
            entries = self._normalize_chat_entries(config_data)

        self._set_chat_entries(entries, selected_identifier)

        if entries:
            self._set_chat_status(f"Geladene Chats: {len(entries)}")
        else:
            self._set_chat_status("Keine Chats geladen")

    def _read_chat_config(self) -> Optional[Dict]:
        """chat_config.json einlesen."""
        data, _ = safe_load_chat_config()
        return data

    def _write_chat_config(
        self,
        known_chats: List[Dict[str, object]],
        selected_chat: Optional[Dict[str, object]] = None
    ) -> None:
        """chat_config.json aktualisieren."""

        payload: Dict[str, object] = {}
        existing = self._read_chat_config()
        if isinstance(existing, dict):
            payload.update(existing)

        payload['known_chats'] = [
            {
                'title': entry.get('title', 'Chat'),
                'username': entry.get('username'),
                'id': entry.get('id')
            }
            for entry in known_chats
        ]

        if selected_chat is not None:
            payload['selected_chat'] = {
                'title': selected_chat.get('title', 'Chat'),
                'username': selected_chat.get('username'),
                'id': selected_chat.get('id')
            }
        else:
            payload.pop('selected_chat', None)

        with open('chat_config.json', 'w', encoding='utf-8') as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def _normalize_chat_entries(self, data) -> List[Dict[str, object]]:
        """Rohdaten aus der Konfiguration vereinheitlichen."""

        entries: List[Dict[str, object]] = []
        if isinstance(data, dict):
            known_chats = data.get('known_chats')
            if isinstance(known_chats, list):
                source = known_chats
            else:
                source = []
                for key, value in data.items():
                    if key == 'selected_chat':
                        continue
                    normalized = self._normalize_chat_entry(value, key)
                    if normalized:
                        entries.append(normalized)
                return entries
        elif isinstance(data, list):
            source = data
        else:
            return entries

        for item in source:
            normalized = self._normalize_chat_entry(item)
            if normalized:
                entries.append(normalized)
        return entries

    @staticmethod
    def _normalize_chat_entry(entry, fallback_id: Optional[object] = None) -> Optional[Dict[str, object]]:
        """Einzelnen Chat-Eintrag normalisieren."""

        if not isinstance(entry, dict):
            return None

        raw_title = entry.get('title') or entry.get('chat_name') or entry.get('name') or entry.get('first_name')
        title = str(raw_title).strip() if raw_title else 'Chat'

        username = entry.get('username')
        if username:
            username = str(username).strip()
            if username and not username.startswith('@'):
                username = f"@{username}"
        else:
            username = None

        chat_id = entry.get('id')
        if chat_id is None:
            chat_id = entry.get('chat_id')
        if chat_id is None and fallback_id is not None:
            try:
                chat_id = int(str(fallback_id))
            except (TypeError, ValueError):
                chat_id = str(fallback_id)

        return {
            'title': title,
            'username': username,
            'id': chat_id
        }

    @staticmethod
    def _entry_identifier(entry: Optional[Dict[str, object]]) -> str:
        if not entry:
            return ""
        username = entry.get('username')
        if username:
            return str(username)
        chat_id = entry.get('id')
        if chat_id is None:
            return ""
        return str(chat_id)

    def _set_chat_entries(
        self,
        entries: List[Dict[str, object]],
        selected_identifier: Optional[str] = None
    ) -> None:
        """Listbox mit den übergebenen Einträgen befüllen."""

        self._chat_entries = list(entries)
        if not self.chats_listbox:
            return

        self.chats_listbox.delete(0, 'end')
        for entry in entries:
            identifier = self._entry_identifier(entry)
            title = entry.get('title') or 'Chat'
            display = f"{title}  ({identifier})" if identifier else title
            self.chats_listbox.insert('end', display)

        if selected_identifier:
            for index, entry in enumerate(entries):
                if self._entry_identifier(entry) == selected_identifier:
                    self.chats_listbox.selection_set(index)
                    self.chats_listbox.see(index)
                    break

    def _set_chat_status(self, text: str) -> None:
        if self.chat_status_var is not None:
            self.chat_status_var.set(text)

    def _set_chat_buttons_state(self, busy: bool) -> None:
        state = ['disabled'] if busy else ['!disabled']
        for button in (self.refresh_chats_button, self.save_chats_button):
            if button is not None:
                button.state(state)

    def _resolve_telegram_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        telegram_cfg = (self.current_config or {}).get('telegram', {}) if isinstance(self.current_config, dict) else {}
        api_id = telegram_cfg.get('api_id') or os.getenv('TG_API_ID')
        api_hash = telegram_cfg.get('api_hash') or os.getenv('TG_API_HASH')
        api_id_str = str(api_id).strip() if api_id not in (None, '') else None
        api_hash_str = str(api_hash).strip() if api_hash not in (None, '') else None
        return api_id_str, api_hash_str

    def _refresh_chats_from_telegram(self) -> None:
        api_id, api_hash = self._resolve_telegram_credentials()
        if not api_id or not api_hash:
            messagebox.showerror(
                "Telegram",
                "Bitte hinterlegen Sie zunächst eine gültige API ID und einen API Hash."
            )
            return

        try:
            api_id_int = int(api_id)
        except (TypeError, ValueError):
            messagebox.showerror("Telegram", "Die API ID muss numerisch sein.")
            return

        self._set_chat_buttons_state(True)
        self._set_chat_status("Lade… (Nummer/Code in Konsole eingeben)")

        def worker() -> None:
            try:
                from telethon import TelegramClient

                async def run() -> List[Dict[str, object]]:
                    async with TelegramClient('tg_session', api_id_int, api_hash) as client:
                        await client.start()
                        dialogs = await client.get_dialogs()
                        rows: List[Dict[str, object]] = []
                        for dialog in dialogs:
                            entity = dialog.entity
                            title = (
                                getattr(entity, 'title', None)
                                or getattr(entity, 'first_name', None)
                                or getattr(entity, 'username', None)
                                or 'Chat'
                            )
                            username = getattr(entity, 'username', None)
                            if username:
                                username = str(username).strip()
                                if username and not username.startswith('@'):
                                    username = f"@{username}"
                            rows.append({
                                'title': str(title).strip() if title else 'Chat',
                                'username': username if username else None,
                                'id': getattr(entity, 'id', None)
                            })
                        return rows

                new_entries = asyncio.run(run())
                self.root.after(0, lambda: self._handle_chat_refresh_success(new_entries))
            except Exception as exc:
                self.root.after(0, lambda: self._handle_chat_refresh_error(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _handle_chat_refresh_success(self, entries: List[Dict[str, object]]) -> None:
        self._set_chat_buttons_state(False)

        normalized = [entry for entry in (self._normalize_chat_entry(item) for item in entries) if entry]

        existing_config = self._read_chat_config() or {}
        selected_entry = None
        if isinstance(existing_config, dict):
            selected_entry = self._normalize_chat_entry(existing_config.get('selected_chat'))

        try:
            self._write_chat_config(normalized, selected_entry)
        except Exception as exc:
            self._set_chat_status("Fehler beim Speichern")
            messagebox.showerror("Chats", f"chat_config.json konnte nicht aktualisiert werden: {exc}")
            return

        selected_identifier = self._entry_identifier(selected_entry) if selected_entry else None
        self._set_chat_entries(normalized, selected_identifier)

        count = len(normalized)
        status_text = f"Fertig ({count} Chats)" if count else "Keine Chats gefunden"
        self._set_chat_status(status_text)

    def _handle_chat_refresh_error(self, error: Exception) -> None:
        self._set_chat_buttons_state(False)
        self._set_chat_status("Fehler beim Laden")
        messagebox.showerror("Telegram", str(error))

    def _save_selected_chat(self) -> None:
        if not self.chats_listbox:
            return

        selection = self.chats_listbox.curselection()
        if not selection:
            messagebox.showinfo("Chats", "Bitte einen Chat auswählen.")
            return

        index = selection[0]
        if index >= len(self._chat_entries):
            messagebox.showinfo("Chats", "Die Auswahl ist nicht verfügbar.")
            return

        entry = self._chat_entries[index]
        identifier = self._entry_identifier(entry)
        if not identifier:
            messagebox.showinfo("Chats", "Der ausgewählte Chat besitzt keine eindeutige Kennung.")
            return

        try:
            self._write_chat_config(self._chat_entries, entry)
        except Exception as exc:
            messagebox.showerror("Chats", f"chat_config.json konnte nicht aktualisiert werden: {exc}")
            return

        env_values = self._read_env_file()
        env_values['TG_TARGET'] = identifier

        try:
            self._write_env_file(env_values)
        except Exception as exc:
            messagebox.showerror("Chats", f".env konnte nicht aktualisiert werden: {exc}")
            return

        os.environ['TG_TARGET'] = identifier
        self._set_chat_status(f"Gespeichert: {identifier}")
        messagebox.showinfo("Chats", f"Gespeichert: {identifier}")

    def _read_env_file(self) -> Dict[str, str]:
        env_path = '.env'
        values: Dict[str, str] = {}
        if not os.path.exists(env_path):
            return values

        try:
            with open(env_path, 'r', encoding='utf-8') as env_file:
                for line in env_file:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        if key:
                            values[key] = value
        except Exception:
            return values
        return values

    def _write_env_file(self, values: Dict[str, str]) -> None:
        with open('.env', 'w', encoding='utf-8') as env_file:
            for key, value in values.items():
                env_file.write(f"{key}={value}\n")

    def _open_bot_settings_from_header(self):
        """Wechselt vom Header direkt zum Bot-Einstellungs-Tab."""
        self.set_initial_page('settings')

    def create_chat_overview_tab(self):
        """Tab für die Chat-Übersicht"""
        chat_frame = ttk.Frame(self.notebook, padding=(24, 24, 24, 20), style='Main.TFrame')
        self.dashboard_tab = chat_frame
        self.notebook.add(chat_frame, text="Risk Monitor")

        header = ttk.Frame(chat_frame, style='Main.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text="Signal & Risiko Übersicht", style='SectionTitle.TLabel').pack(side='left')
        self.chat_summary_var = tk.StringVar(master=self.root, value="Noch keine Chats geladen")
        self.chat_summary_label = ttk.Label(header, textvariable=self.chat_summary_var, style='Info.TLabel')
        self.chat_summary_label.pack(side='right')

        status_row = ttk.Frame(chat_frame, style='Main.TFrame')
        status_row.pack(fill='x', pady=(18, 18))
        status_row.columnconfigure((0, 1), weight=1, uniform='status')

        self.dashboard_state_var = tk.StringVar(master=self.root, value="SAFE")
        self.open_signal_active_var = tk.StringVar(master=self.root, value="Keine offenen Signale")
        safe_card = ttk.Frame(status_row, style='StatusGood.TFrame', padding=(24, 20))
        safe_card.grid(row=0, column=0, sticky='nsew', padx=(0, 18))
        ttk.Label(safe_card, textvariable=self.dashboard_state_var, style='StatusGoodHeading.TLabel').pack(anchor='w')
        ttk.Label(safe_card, textvariable=self.open_signal_active_var, style='StatusSub.TLabel').pack(anchor='w', pady=(12, 0))

        self.dashboard_alert_var = tk.StringVar(master=self.root, value="Keine aktiven Warnungen")
        alert_card = ttk.Frame(status_row, style='StatusAlert.TFrame', padding=(24, 20))
        alert_card.grid(row=0, column=1, sticky='nsew')
        ttk.Label(alert_card, text="EMERGENCY FLAT", style='StatusAlertHeading.TLabel').pack(anchor='w')
        ttk.Label(alert_card, textvariable=self.dashboard_alert_var, style='StatusAlertSub.TLabel').pack(anchor='w', pady=(12, 0))

        metrics_row = ttk.Frame(chat_frame, style='Main.TFrame')
        metrics_row.pack(fill='x')
        metrics_row.columnconfigure((0, 1, 2, 3), weight=1, uniform='metrics')

        self.chat_total_var = tk.StringVar(master=self.root, value="0")
        total_card = self._build_dashboard_card(
            metrics_row,
            title="Signalquellen",
            value=self.chat_total_var,
            icon="🛰"
        )
        total_card.grid(row=0, column=0, sticky='nsew', padx=(0, 18))

        self.chat_active_var = tk.StringVar(master=self.root, value="0")
        active_card = self._build_dashboard_card(
            metrics_row,
            title="Aktiv überwacht",
            value=self.chat_active_var,
            icon="⚡"
        )
        active_card.grid(row=0, column=1, sticky='nsew', padx=(0, 18))

        self.chat_signal_sum_var = tk.StringVar(master=self.root, value="0")
        signals_card = self._build_dashboard_card(
            metrics_row,
            title="Gesamt-Signale",
            value=self.chat_signal_sum_var,
            icon="📈",
            subtitle="Seit Start"
        )
        signals_card.grid(row=0, column=2, sticky='nsew', padx=(0, 18))

        exposure_card = ttk.Frame(metrics_row, style='GlassCard.TFrame', padding=(24, 20))
        exposure_card.grid(row=0, column=3, sticky='nsew')
        ttk.Label(exposure_card, text="Exposure nach Symbol", style='GlassCardTitle.TLabel').pack(anchor='w')
        ttk.Label(
            exposure_card,
            text="Anteil offener Positionen je Instrument",
            style='GlassCardSubtitle.TLabel'
        ).pack(anchor='w', pady=(4, 12))
        self.exposure_list_frame = ttk.Frame(exposure_card, style='GlassCard.TFrame')
        self.exposure_list_frame.pack(fill='x')

        signals_overview = ttk.Frame(chat_frame, style='Card.TFrame', padding=(26, 24))
        signals_overview.pack(fill='x', pady=(24, 20))
        overview_header = ttk.Frame(signals_overview, style='Card.TFrame')
        overview_header.pack(fill='x')
        ttk.Label(overview_header, text="Open Signals", style='CardTitle.TLabel').pack(side='left')
        self.open_signal_percentage_var = tk.StringVar(master=self.root, value="0.0% Auslastung")
        ttk.Label(overview_header, textvariable=self.open_signal_percentage_var, style='BadgeInfo.TLabel').pack(side='right')

        progress_frame = ttk.Frame(signals_overview, style='Card.TFrame')
        progress_frame.pack(fill='x', pady=(16, 18))
        self.open_signal_progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(
            progress_frame,
            variable=self.open_signal_progress_var,
            maximum=100,
            style='Success.Horizontal.TProgressbar'
        ).pack(fill='x')

        limits_frame = ttk.Frame(signals_overview, style='Card.TFrame')
        limits_frame.pack(fill='x')
        limits_frame.columnconfigure((0, 1), weight=1)
        ttk.Label(limits_frame, text="Session Limits", style='CardSubtitle.TLabel').grid(row=0, column=0, sticky='w')
        self.dashboard_drawdown_var = tk.StringVar(master=self.root, value="-0.0%")
        self.dashboard_daily_loss_var = tk.StringVar(master=self.root, value="€0")
        ttk.Label(limits_frame, text="Max Drawdown", style='FieldLabel.TLabel').grid(row=1, column=0, sticky='w', pady=(12, 2))
        ttk.Label(limits_frame, textvariable=self.dashboard_drawdown_var, style='MetricValue.TLabel').grid(row=1, column=1, sticky='e', pady=(12, 2))
        ttk.Label(limits_frame, text="Tagesverlust", style='FieldLabel.TLabel').grid(row=2, column=0, sticky='w', pady=(4, 0))
        ttk.Label(limits_frame, textvariable=self.dashboard_daily_loss_var, style='MetricValue.TLabel').grid(row=2, column=1, sticky='e', pady=(4, 0))

        self.chat_last_sync_var = tk.StringVar(master=self.root, value="Letzte Synchronisierung: –")
        ttk.Label(signals_overview, textvariable=self.chat_last_sync_var, style='CardSubtitle.TLabel').pack(anchor='w', pady=(18, 0))

        self.dashboard_compliance_var = tk.StringVar(master=self.root, value="0 Compliance Alerts")
        ttk.Label(signals_overview, textvariable=self.dashboard_compliance_var, style='CardSubtitle.TLabel').pack(anchor='w', pady=(6, 0))

        controls_frame = ttk.Frame(chat_frame, style='Toolbar.TFrame', padding=(20, 14))
        controls_frame.pack(fill='x', pady=(0, 20))
        controls_frame.columnconfigure((0, 1, 2, 3), weight=1)

        ttk.Button(
            controls_frame,
            text="🔄 Quellen synchronisieren",
            command=self.load_chats,
            style='Toolbar.TButton'
        ).grid(row=0, column=0, sticky='w', padx=(0, 16))

        ttk.Button(
            controls_frame,
            text="🟢 Monitoring aktivieren",
            command=self.enable_monitoring,
            style='Toolbar.TButton'
        ).grid(row=0, column=1, sticky='w', padx=(0, 16))

        ttk.Button(
            controls_frame,
            text="⛔ Monitoring pausieren",
            command=self.disable_monitoring,
            style='Toolbar.TButton'
        ).grid(row=0, column=2, sticky='w', padx=(0, 16))

        ttk.Button(
            controls_frame,
            text="💾 Konfiguration sichern",
            command=self.export_chat_config,
            style='Toolbar.TButton'
        ).grid(row=0, column=3, sticky='w')

        table_card = ttk.Frame(chat_frame, style='Card.TFrame', padding=(0, 0, 0, 18))
        table_card.pack(fill='both', expand=True)

        table_header = ttk.Frame(table_card, style='Card.TFrame', padding=(26, 20, 26, 10))
        table_header.pack(fill='x')
        ttk.Label(table_header, text="Signal-Quellen Übersicht", style='CardTitle.TLabel').pack(side='left')
        ttk.Button(
            table_header,
            text="ℹ Hilfe",
            style='Link.TButton',
            command=lambda: messagebox.showinfo(
                "Information",
                "Markieren Sie Chats und nutzen Sie die Toolbar, um die Überwachung anzupassen."
            )
        ).pack(side='right')

        columns = ('Name', 'ID', 'Typ', 'Teilnehmer', 'Überwacht', 'Signale')
        heading_texts = {
            'Name': 'Chat-Name',
            'ID': 'Chat-ID',
            'Typ': 'Art',
            'Teilnehmer': 'Mitglieder',
            'Überwacht': 'Monitoring',
            'Signale': 'Signale gesamt'
        }
        column_widths = {
            'Name': 260,
            'ID': 150,
            'Typ': 110,
            'Teilnehmer': 140,
            'Überwacht': 140,
            'Signale': 140
        }

        table_container = ttk.Frame(table_card, style='Card.TFrame', padding=(26, 0, 26, 0))
        table_container.pack(fill='both', expand=True)
        self.chats_tree = ttk.Treeview(
            table_container,
            columns=columns,
            show='headings',
            height=14,
            style='Dashboard.Treeview'
        )
        self.chats_tree.pack(side='left', fill='both', expand=True)
        for col in columns:
            self.chats_tree.heading(col, text=heading_texts.get(col, col))
            self.chats_tree.column(col, width=column_widths.get(col, 120), anchor='w')

        chat_scroll = ttk.Scrollbar(table_container, orient='vertical', command=self.chats_tree.yview)
        chat_scroll.pack(side='right', fill='y')
        self.chats_tree.configure(yscrollcommand=chat_scroll.set)

        self._refresh_symbol_exposure()

    def create_bot_settings_tab(self):
        """Tab für Bot-Einstellungen und Status"""
        settings_tab = ttk.Frame(self.notebook, padding=(24, 24, 24, 20), style='Main.TFrame')
        self.bot_settings_tab = settings_tab
        self.notebook.add(settings_tab, text="Automationen")

        header = ttk.Frame(settings_tab, style='Main.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text="Automation & Bot-Steuerung", style='SectionTitle.TLabel').pack(side='left')
        self.trade_status_label = ttk.Label(header, text="Demo-Modus aktiv", style='BadgeSuccess.TLabel')
        self.trade_status_label.pack(side='right')

        overview_row = ttk.Frame(settings_tab, style='Main.TFrame')
        overview_row.pack(fill='x', pady=(24, 22))
        overview_row.columnconfigure((0, 1, 2), weight=1, uniform='overview')

        mode_card = ttk.Frame(overview_row, style='Card.TFrame', padding=(26, 22))
        mode_card.grid(row=0, column=0, sticky='nsew', padx=(0, 18))
        ttk.Label(mode_card, text="Betriebsmodus", style='CardTitle.TLabel').pack(anchor='w')
        ttk.Label(mode_card, text="Steuert, ob Trades live ausgeführt werden.", style='CardSubtitle.TLabel').pack(anchor='w', pady=(4, 14))
        ttk.Checkbutton(
            mode_card,
            text="Demo-Modus (empfohlen)",
            variable=self.demo_var,
            command=self.toggle_demo_mode,
            style='Switch.TCheckbutton'
        ).pack(anchor='w')

        execution_card = ttk.Frame(overview_row, style='Card.TFrame', padding=(26, 22))
        execution_card.grid(row=0, column=1, sticky='nsew', padx=(0, 18))
        ttk.Label(execution_card, text="Ausführungsmodus", style='CardTitle.TLabel').pack(anchor='w')
        ttk.Label(execution_card, text="Legt fest, wie Signale umgesetzt werden.", style='CardSubtitle.TLabel').pack(anchor='w', pady=(4, 14))
        default_execution_label = self.execution_mode_labels.get(ExecutionMode.INSTANT, "Sofortausführung")
        self.execution_mode_var = tk.StringVar(value=default_execution_label)
        self.execution_mode_combobox = ttk.Combobox(
            execution_card,
            textvariable=self.execution_mode_var,
            values=list(self.execution_mode_labels.values()),
            state='readonly'
        )
        self.execution_mode_combobox.pack(fill='x')
        self.execution_mode_combobox.bind('<<ComboboxSelected>>', self.on_execution_mode_change)

        telemetry_card = ttk.Frame(overview_row, style='Card.TFrame', padding=(26, 22))
        telemetry_card.grid(row=0, column=2, sticky='nsew')
        ttk.Label(telemetry_card, text="Realtime Telemetrie", style='CardTitle.TLabel').pack(anchor='w')
        ttk.Label(telemetry_card, text="Einblicke in Latenz & Alarmqualität.", style='CardSubtitle.TLabel').pack(anchor='w', pady=(4, 14))
        self.latency_var = tk.StringVar(master=self.root, value="210 ms")
        self.alert_score_var = tk.StringVar(master=self.root, value="0.63")
        ttk.Label(telemetry_card, text="Latenz zum Broker", style='FieldLabel.TLabel').pack(anchor='w')
        ttk.Label(telemetry_card, textvariable=self.latency_var, style='MetricValue.TLabel').pack(anchor='w', pady=(4, 12))
        ttk.Label(telemetry_card, text="Alert Quality Score", style='FieldLabel.TLabel').pack(anchor='w')
        ttk.Label(telemetry_card, textvariable=self.alert_score_var, style='MetricValue.TLabel').pack(anchor='w', pady=(4, 0))

        automation_card = ttk.Frame(settings_tab, style='GlassCard.TFrame', padding=(28, 24))
        automation_card.pack(fill='x', pady=(0, 22))
        automation_header = ttk.Frame(automation_card, style='GlassCard.TFrame')
        automation_header.pack(fill='x')
        ttk.Label(automation_header, text="AUTOMATIONS", style='GlassCardTitle.TLabel').pack(side='left')
        ttk.Label(automation_header, text="LIVE", style='BadgeSuccess.TLabel').pack(side='left', padx=(12, 0))
        ttk.Button(
            automation_header,
            text="+ Neue Regel",
            style='Link.TButton',
            command=lambda: self.log_message("Benutzerdefinierte Automationsregeln folgen in einer späteren Version.")
        ).pack(side='right')

        self.automation_rules_container = ttk.Frame(automation_card, style='GlassCard.TFrame')
        self.automation_rules_container.pack(fill='x', pady=(20, 0))

        risk_card = ttk.Frame(settings_tab, style='Card.TFrame', padding=(28, 24))
        risk_card.pack(fill='x', pady=(0, 22))
        risk_card.columnconfigure((1, 3), weight=1)
        ttk.Label(risk_card, text="Risikoparameter", style='CardTitle.TLabel').grid(row=0, column=0, columnspan=4, sticky='w')
        ttk.Label(risk_card, text="Definiert Positionsgrößen & Sicherheitsgrenzen.", style='CardSubtitle.TLabel').grid(row=1, column=0, columnspan=4, sticky='w', pady=(4, 18))

        trading_spinboxes = [
            (
                "Standard-Lotgröße", self.default_lot_var, 0.01, 100.0, 0.01,
                '%.2f', self._float_validate_cmd, '0.01', 'default_lot_size'
            ),
            (
                "Max. Spread (Pips)", self.max_spread_var, 0.0, 50.0, 0.1,
                '%.1f', self._float_validate_cmd, '0.0', 'max_spread_pips'
            ),
            (
                "Risiko pro Trade (%)", self.risk_percent_var, 0.0, 100.0, 0.1,
                '%.1f', self._float_validate_cmd, '0.0', 'risk_percent'
            ),
            (
                "Max. Trades / Stunde", self.max_trades_per_hour_var, 0, 50, 1,
                None, self._int_validate_cmd, '0', 'max_trades_per_hour'
            )
        ]

        for idx, (label_text, variable, minimum, maximum, step, number_format, validator, min_text, key) in enumerate(trading_spinboxes):
            row = 2 + idx // 2
            column = (idx % 2) * 2
            ttk.Label(risk_card, text=label_text, style='FieldLabel.TLabel').grid(row=row, column=column, sticky='w', pady=(0, 6))

            spinbox_kwargs = {
                'textvariable': variable,
                'from_': minimum,
                'to': maximum,
                'increment': step,
                'width': 12,
                'validate': 'focusout',
                'validatecommand': (validator, '%P', str(min_text), key)
            }
            if number_format:
                spinbox_kwargs['format'] = number_format
            spinbox = ttk.Spinbox(risk_card, **spinbox_kwargs)
            spinbox.grid(row=row, column=column + 1, sticky='w', padx=(12, 0))

        toolbar = ttk.Frame(settings_tab, style='Toolbar.TFrame', padding=(20, 14))
        toolbar.pack(fill='x', pady=(0, 22))
        ttk.Button(toolbar, text="📥 Signale abrufen", style='Toolbar.TButton', command=self.load_chats).pack(side='left')
        ttk.Button(toolbar, text="🧹 Log leeren", style='Toolbar.TButton', command=self.clear_log).pack(side='left', padx=(12, 0))
        ttk.Button(toolbar, text="📊 Statistiken aktualisieren", style='Toolbar.TButton', command=self.refresh_statistics).pack(side='left', padx=(12, 0))

        metrics_frame = ttk.Frame(settings_tab, style='Main.TFrame')
        metrics_frame.pack(fill='x', pady=(0, 22))
        metrics_frame.columnconfigure((0, 1, 2), weight=1)
        metric_titles = [
            ("Aktive Chats", "0"),
            ("Überwachte Signale", "0"),
            ("Heute synchronisiert", "0")
        ]
        for idx, (title, value) in enumerate(metric_titles):
            metric_card = ttk.Frame(metrics_frame, style='GlassCard.TFrame', padding=(22, 18))
            metric_card.grid(row=0, column=idx, sticky='nsew', padx=(0 if idx == 0 else 18, 0))
            ttk.Label(metric_card, text=title, style='GlassCardSubtitle.TLabel').pack(anchor='w')
            ttk.Label(metric_card, text=value, style='MetricValue.TLabel').pack(anchor='w', pady=(6, 0))

        log_frame = ttk.Frame(settings_tab, style='Card.TFrame', padding=(0, 0, 0, 0))
        log_frame.pack(fill='both', expand=True)
        log_header = ttk.Frame(log_frame, style='Card.TFrame', padding=(24, 18, 24, 8))
        log_header.pack(fill='x')
        ttk.Label(log_header, text="Live-Aktivitätsprotokoll", style='CardTitle.TLabel').pack(side='left')

        log_container = ttk.Frame(log_frame, style='Card.TFrame', padding=(24, 0, 24, 24))
        log_container.pack(fill='both', expand=True)
        self.log_text = tk.Text(
            log_container,
            height=16,
            wrap='word',
            bg=self.theme_colors['surface_alt'],
            fg=self.theme_colors['text'],
            insertbackground=self.theme_colors['accent'],
            font=('Consolas', 11),
            borderwidth=0,
            relief='flat',
            highlightthickness=1,
            highlightbackground=self.theme_colors['border'],
            highlightcolor=self.theme_colors['border']
        )
        self.log_text.pack(side='left', fill='both', expand=True)

        log_scroll = ttk.Scrollbar(log_container, orient='vertical', command=self.log_text.yview)
        log_scroll.pack(side='right', fill='y')
        self.log_text.configure(yscrollcommand=log_scroll.set, padx=16, pady=14, spacing3=6)

        self._refresh_automation_rules_display()

    def create_mt5_settings_tab(self):
        """Tab für MetaTrader-5-Zugangsdaten und Verbindung"""
        mt5_tab = ttk.Frame(self.notebook, padding=(24, 24, 24, 20), style='Main.TFrame')
        self.notebook.add(mt5_tab, text="MetaTrader 5")

        header = ttk.Frame(mt5_tab, style='Main.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text="MetaTrader 5 Integration", style='SectionTitle.TLabel').pack(side='left')
        status_text = "MT5-Unterstützung aktiv" if MT5_AVAILABLE else "MT5-Modul nicht gefunden"
        badge_style = 'BadgeSuccess.TLabel' if MT5_AVAILABLE else 'BadgeWarning.TLabel'
        ttk.Label(header, text=status_text, style=badge_style).pack(side='right')

        summary_row = ttk.Frame(mt5_tab, style='Main.TFrame')
        summary_row.pack(fill='x', pady=(24, 20))
        summary_row.columnconfigure((0, 1, 2), weight=1, uniform='mt5summary')

        login_card = ttk.Frame(summary_row, style='Card.TFrame', padding=(24, 20))
        login_card.grid(row=0, column=0, sticky='nsew', padx=(0, 18))
        ttk.Label(login_card, text="Login Status", style='CardTitle.TLabel').pack(anchor='w')
        ttk.Label(login_card, text="Hinterlegte Kontonummer", style='CardSubtitle.TLabel').pack(anchor='w', pady=(4, 12))
        ttk.Label(login_card, textvariable=self.mt5_login_var, style='MetricValue.TLabel').pack(anchor='w')

        server_card = ttk.Frame(summary_row, style='Card.TFrame', padding=(24, 20))
        server_card.grid(row=0, column=1, sticky='nsew', padx=(0, 18))
        ttk.Label(server_card, text="Server", style='CardTitle.TLabel').pack(anchor='w')
        ttk.Label(server_card, text="Konfigurierter MT5-Server", style='CardSubtitle.TLabel').pack(anchor='w', pady=(4, 12))
        ttk.Label(server_card, textvariable=self.mt5_server_var, style='MetricValue.TLabel').pack(anchor='w')

        mode_card = ttk.Frame(summary_row, style='Card.TFrame', padding=(24, 20))
        mode_card.grid(row=0, column=2, sticky='nsew')
        ttk.Label(mode_card, text="Verbindungsmodus", style='CardTitle.TLabel').pack(anchor='w')
        ttk.Label(mode_card, text="LIVE-Verbindung benötigt MT5-Modul", style='CardSubtitle.TLabel').pack(anchor='w', pady=(4, 12))
        ttk.Label(mode_card, text="Verbindung bereit" if MT5_AVAILABLE else "Nur Demo-Modus", style='MetricValue.TLabel').pack(anchor='w')

        mt5_frame = ttk.Frame(mt5_tab, style='Card.TFrame', padding=(28, 24))
        mt5_frame.pack(fill='x', pady=(0, 22))
        mt5_frame.columnconfigure(1, weight=1)
        ttk.Label(mt5_frame, text="Zugangsdaten", style='CardTitle.TLabel').grid(row=0, column=0, columnspan=3, sticky='w')
        ttk.Label(mt5_frame, text="Erforderlich für LIVE-Trading.", style='CardSubtitle.TLabel').grid(row=1, column=0, columnspan=3, sticky='w', pady=(4, 18))

        entry_state = 'normal'
        browse_button_state = 'normal'
        save_button_state = 'normal'
        test_button_state = 'normal' if MT5_AVAILABLE else 'disabled'

        ttk.Label(mt5_frame, text="Login (Kontonummer)", style='FieldLabel.TLabel').grid(row=2, column=0, sticky='w', pady=(0, 4))
        ttk.Entry(mt5_frame, textvariable=self.mt5_login_var, state=entry_state).grid(row=2, column=1, sticky='ew', padx=(12, 0))

        ttk.Label(mt5_frame, text="Passwort", style='FieldLabel.TLabel').grid(row=3, column=0, sticky='w', pady=(12, 4))
        ttk.Entry(mt5_frame, textvariable=self.mt5_password_var, show='•', state=entry_state).grid(row=3, column=1, sticky='ew', padx=(12, 0))

        ttk.Label(mt5_frame, text="Server", style='FieldLabel.TLabel').grid(row=4, column=0, sticky='w', pady=(12, 4))
        ttk.Entry(mt5_frame, textvariable=self.mt5_server_var, state=entry_state).grid(row=4, column=1, sticky='ew', padx=(12, 0))

        ttk.Label(mt5_frame, text="MT5-Terminal (optional)", style='FieldLabel.TLabel').grid(row=5, column=0, sticky='w', pady=(12, 4))
        ttk.Entry(mt5_frame, textvariable=self.mt5_path_var, state=entry_state).grid(row=5, column=1, sticky='ew', padx=(12, 0))
        ttk.Button(mt5_frame, text="Durchsuchen…", command=self.browse_mt5_path, state=browse_button_state).grid(row=5, column=2, sticky='w', padx=(10, 0))

        button_row = ttk.Frame(mt5_frame, style='Card.TFrame')
        button_row.grid(row=6, column=0, columnspan=3, sticky='w', pady=(18, 0))
        ttk.Button(button_row, text="Zugangsdaten speichern", command=self.save_mt5_credentials, state=save_button_state).pack(side='left')
        ttk.Button(button_row, text="Verbindung testen", command=self.test_mt5_connection, state=test_button_state).pack(side='left', padx=(12, 0))

        self.mt5_status_card = ttk.Frame(mt5_frame, style='GlassCard.TFrame', padding=(18, 16))
        self.mt5_status_card.grid(row=7, column=0, columnspan=3, sticky='ew', pady=(20, 0))
        self.mt5_status_message_var = tk.StringVar(value="Noch keine MT5-Zugangsdaten gespeichert.")
        self.mt5_status_label = ttk.Label(
            self.mt5_status_card,
            textvariable=self.mt5_status_message_var,
            style='GlassCardSubtitle.TLabel',
            wraplength=760,
            justify='left'
        )
        self.mt5_status_label.pack(anchor='w')

        info_text = (
            "Hinweis: Hinterlegen Sie ein MT5-Demokonto, um den LIVE-Modus vorab zu testen."
            if MT5_AVAILABLE
            else "MetaTrader5-Python-Modul wurde nicht gefunden. Installieren Sie MetaTrader 5 inklusive Python-Paket."
        )
        info_style = 'Info.TLabel' if MT5_AVAILABLE else 'Warning.TLabel'
        ttk.Label(mt5_tab, text=info_text, style=info_style, wraplength=820, justify='left').pack(fill='x', pady=(0, 12))

        self._refresh_mt5_status_display()

    def create_statistics_tab(self):
        """Tab für Statistiken des Kopierers"""
        stats_frame = ttk.Frame(self.notebook, padding=(24, 24, 24, 20), style='Main.TFrame')
        self.notebook.add(stats_frame, text="Performance")

        header = ttk.Frame(stats_frame, style='Main.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text="Performance des Kopierers", style='SectionTitle.TLabel').pack(side='left')
        self.statistics_hint = ttk.Label(header, text="Letzte Aktualisierung: –", style='Info.TLabel')
        self.statistics_hint.pack(side='right')

        performance_row = ttk.Frame(stats_frame, style='Main.TFrame')
        performance_row.pack(fill='x', pady=(24, 22))
        performance_row.columnconfigure(0, weight=3)
        performance_row.columnconfigure(1, weight=2)

        performance_card = ttk.Frame(performance_row, style='Card.TFrame', padding=(28, 24))
        performance_card.grid(row=0, column=0, sticky='nsew', padx=(0, 20))
        perf_header = ttk.Frame(performance_card, style='Card.TFrame')
        perf_header.pack(fill='x')
        ttk.Label(perf_header, text="Trading Bot Performance", style='CardTitle.TLabel').pack(side='left')
        header_right = ttk.Frame(perf_header, style='Card.TFrame')
        header_right.pack(side='right')
        ttk.Button(
            header_right,
            text="Zeitraum wählen",
            style='Link.TButton',
            command=lambda: self.log_message("Zeitraumfilter folgt.")
        ).pack(side='right')
        ttk.Button(header_right, text="CSV exportieren", style='Link.TButton', command=self.export_statistics).pack(side='right', padx=(16, 0))

        self.equity_curve_canvas = tk.Canvas(
            performance_card,
            height=240,
            bg=self.theme_colors['surface_alt'],
            highlightthickness=0,
            borderwidth=0
        )
        self.equity_curve_canvas.pack(fill='both', expand=True, pady=(24, 0))
        self.equity_curve_canvas.bind('<Configure>', self._draw_equity_curve)

        kpi_card = ttk.Frame(performance_row, style='GlassCard.TFrame', padding=(26, 22))
        kpi_card.grid(row=0, column=1, sticky='nsew')
        kpi_card.columnconfigure((0, 1), weight=1)

        self.stats_sharpe_var = tk.StringVar(master=self.root, value="1.45")
        self.stats_sortino_var = tk.StringVar(master=self.root, value="2.30")
        self.stats_drawdown_var = tk.StringVar(master=self.root, value="-13.7%")
        self.stats_win_rate_var = tk.StringVar(master=self.root, value="67.5%")
        self.stats_profit_var = tk.StringVar(master=self.root, value="+4 453")

        kpi_values = [
            ("Sharpe", self.stats_sharpe_var, 'MetricValue.TLabel'),
            ("Sortino", self.stats_sortino_var, 'MetricValue.TLabel'),
            ("Max Drawdown", self.stats_drawdown_var, 'MetricNegative.TLabel'),
            ("Win Rate", self.stats_win_rate_var, 'MetricValue.TLabel'),
            ("Profit", self.stats_profit_var, 'MetricPositive.TLabel')
        ]
        for idx, (title, var, style) in enumerate(kpi_values):
            row = idx // 2
            column = idx % 2
            card = ttk.Frame(kpi_card, style='GlassCard.TFrame', padding=(12, 10))
            card.grid(row=row, column=column, sticky='nsew', padx=(0 if column == 0 else 16, 0), pady=(0 if row == 0 else 16, 0))
            ttk.Label(card, text=title, style='GlassCardSubtitle.TLabel').pack(anchor='w')
            ttk.Label(card, textvariable=var, style=style).pack(anchor='w', pady=(6, 0))

        analytics_row = ttk.Frame(stats_frame, style='Main.TFrame')
        analytics_row.pack(fill='x', pady=(0, 22))
        analytics_row.columnconfigure((0, 1), weight=1, uniform='analytics')

        monthly_card = ttk.Frame(analytics_row, style='Card.TFrame', padding=(26, 22))
        monthly_card.grid(row=0, column=0, sticky='nsew', padx=(0, 20))
        ttk.Label(monthly_card, text="Monthly Profit", style='CardTitle.TLabel').pack(anchor='w')
        ttk.Label(monthly_card, text="Heatmap der Monatsperformance", style='CardSubtitle.TLabel').pack(anchor='w', pady=(4, 18))
        self.monthly_heatmap_canvas = tk.Canvas(
            monthly_card,
            height=170,
            bg=self.theme_colors['surface_alt'],
            highlightthickness=0,
            borderwidth=0
        )
        self.monthly_heatmap_canvas.pack(fill='both', expand=True)
        self.monthly_heatmap_canvas.bind('<Configure>', self._draw_monthly_profit_heatmap)

        distribution_card = ttk.Frame(analytics_row, style='Card.TFrame', padding=(26, 22))
        distribution_card.grid(row=0, column=1, sticky='nsew')
        ttk.Label(distribution_card, text="Profit Distribution", style='CardTitle.TLabel').pack(anchor='w')
        ttk.Label(distribution_card, text="Histogramm der Trade-Ergebnisse", style='CardSubtitle.TLabel').pack(anchor='w', pady=(4, 18))
        self.profit_distribution_canvas = tk.Canvas(
            distribution_card,
            height=170,
            bg=self.theme_colors['surface_alt'],
            highlightthickness=0,
            borderwidth=0
        )
        self.profit_distribution_canvas.pack(fill='both', expand=True)
        self.profit_distribution_canvas.bind('<Configure>', self._draw_profit_distribution)

        detail_row = ttk.Frame(stats_frame, style='Main.TFrame')
        detail_row.pack(fill='both', expand=True)
        detail_row.columnconfigure(0, weight=2)
        detail_row.columnconfigure(1, weight=3)

        detail_card = ttk.Frame(detail_row, style='GlassCard.TFrame', padding=(28, 24))
        detail_card.grid(row=0, column=0, sticky='nsew', padx=(0, 20))
        detail_header = ttk.Frame(detail_card, style='GlassCard.TFrame')
        detail_header.pack(fill='x')
        self.chat_detail_title_var = tk.StringVar(master=self.root, value="Chat-Statistiken – Auswahl")
        ttk.Label(detail_header, textvariable=self.chat_detail_title_var, style='GlassCardTitle.TLabel').pack(side='left')
        ttk.Label(detail_header, text="LIVE", style='BadgeInfo.TLabel').pack(side='left', padx=(12, 0))
        ttk.Button(detail_header, text="EXPORT CSV", style='Link.TButton', command=self.export_statistics).pack(side='right')

        metrics_grid = ttk.Frame(detail_card, style='GlassCard.TFrame')
        metrics_grid.pack(fill='x', pady=(18, 18))
        metrics_grid.columnconfigure((0, 1), weight=1)

        self.detail_winrate_var = tk.StringVar(master=self.root, value="69.4%")
        self.detail_rr_var = tk.StringVar(master=self.root, value="2.18")
        self.detail_signals_var = tk.StringVar(master=self.root, value="125")
        self.detail_profit_var = tk.StringVar(master=self.root, value="+9 432")

        detail_metrics = [
            ("Winrate", self.detail_winrate_var),
            ("Risk/Reward", self.detail_rr_var),
            ("Signale", self.detail_signals_var),
            ("Profit", self.detail_profit_var)
        ]
        for idx, (title, var) in enumerate(detail_metrics):
            row = idx // 2
            column = idx % 2
            card = ttk.Frame(metrics_grid, style='GlassCard.TFrame', padding=(12, 10))
            card.grid(row=row, column=column, sticky='nsew', padx=(0 if column == 0 else 16, 0), pady=(0 if row == 0 else 16, 0))
            ttk.Label(card, text=title, style='GlassCardSubtitle.TLabel').pack(anchor='w')
            ttk.Label(card, textvariable=var, style='MetricValue.TLabel').pack(anchor='w', pady=(6, 0))

        ttk.Label(detail_card, text="Session Heatmap", style='GlassCardSubtitle.TLabel').pack(anchor='w')
        self.session_heatmap_canvas = tk.Canvas(
            detail_card,
            height=170,
            bg=self.theme_colors['surface_alt'],
            highlightthickness=0,
            borderwidth=0
        )
        self.session_heatmap_canvas.pack(fill='both', expand=True, pady=(12, 20))
        self.session_heatmap_canvas.bind('<Configure>', self._draw_session_heatmap)

        distribution_section = ttk.Frame(detail_card, style='GlassCard.TFrame')
        distribution_section.pack(fill='x')
        distribution_section.columnconfigure((0, 1), weight=1)

        pair_frame = ttk.Frame(distribution_section, style='GlassCard.TFrame')
        pair_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 12))
        ttk.Label(pair_frame, text="Pair Distribution", style='GlassCardSubtitle.TLabel').pack(anchor='w')
        self.pair_distribution_canvas = tk.Canvas(
            pair_frame,
            height=140,
            bg=self.theme_colors['surface_alt'],
            highlightthickness=0,
            borderwidth=0
        )
        self.pair_distribution_canvas.pack(fill='both', expand=True, pady=(12, 0))
        self.pair_distribution_canvas.bind('<Configure>', self._draw_pair_distribution)

        accuracy_frame = ttk.Frame(distribution_section, style='GlassCard.TFrame')
        accuracy_frame.grid(row=0, column=1, sticky='nsew', padx=(12, 0))
        ttk.Label(accuracy_frame, text="Accuracy by Pair", style='GlassCardSubtitle.TLabel').pack(anchor='w')
        self.accuracy_gauge_canvas = tk.Canvas(
            accuracy_frame,
            height=140,
            bg=self.theme_colors['surface_alt'],
            highlightthickness=0,
            borderwidth=0
        )
        self.accuracy_gauge_canvas.pack(fill='both', expand=True, pady=(12, 0))
        self.accuracy_gauge_canvas.bind('<Configure>', self._draw_accuracy_gauge)

        stats_meta = ttk.Frame(detail_card, style='GlassCard.TFrame')
        stats_meta.pack(fill='x', pady=(18, 0))
        stats_meta.columnconfigure((0, 1, 2), weight=1)
        self.chat_detail_latency_var = tk.StringVar(master=self.root, value="210 ms")
        self.chat_detail_quality_var = tk.StringVar(master=self.root, value="0.63")
        self.chat_detail_risk_var = tk.StringVar(master=self.root, value=f"{self.risk_percent_var.get():.1f}% Risiko")
        meta_values = [
            ("Latenz", self.chat_detail_latency_var),
            ("Alert Quality", self.chat_detail_quality_var),
            ("Risk Parameter", self.chat_detail_risk_var)
        ]
        for idx, (title, var) in enumerate(meta_values):
            ttk.Label(stats_meta, text=title, style='GlassCardSubtitle.TLabel').grid(row=0, column=idx, sticky='w')
            ttk.Label(stats_meta, textvariable=var, style='MetricValue.TLabel').grid(row=1, column=idx, sticky='w', pady=(4, 0))

        source_card = ttk.Frame(detail_row, style='Card.TFrame', padding=(0, 0, 0, 24))
        source_card.grid(row=0, column=1, sticky='nsew')
        source_header = ttk.Frame(source_card, style='Card.TFrame', padding=(26, 20, 26, 10))
        source_header.pack(fill='x')
        ttk.Label(source_header, text="Signal-Quellen Übersicht", style='CardTitle.TLabel').pack(side='left')
        ttk.Button(source_header, text="🔁 Aktualisieren", style='Link.TButton', command=self.refresh_statistics).pack(side='right')

        stats_columns = ('Quelle', 'Trades', 'Gewinnrate', 'Profit', 'Letzter Trade')
        heading_texts = {
            'Quelle': 'Signal-Quelle',
            'Trades': 'Trades gesamt',
            'Gewinnrate': 'Trefferquote',
            'Profit': 'Netto-Profit',
            'Letzter Trade': 'Letzte Aktivität'
        }
        column_widths = {
            'Quelle': 240,
            'Trades': 140,
            'Gewinnrate': 150,
            'Profit': 150,
            'Letzter Trade': 180
        }

        table_container = ttk.Frame(source_card, style='Card.TFrame', padding=(26, 0, 26, 0))
        table_container.pack(fill='both', expand=True)
        self.stats_tree = ttk.Treeview(
            table_container,
            columns=stats_columns,
            show='headings',
            height=13,
            style='Dashboard.Treeview'
        )
        self.stats_tree.pack(side='left', fill='both', expand=True)
        for col in stats_columns:
            self.stats_tree.heading(col, text=heading_texts.get(col, col))
            self.stats_tree.column(col, width=column_widths.get(col, 140), anchor='w')
        self.stats_tree.bind('<<TreeviewSelect>>', self._on_stats_tree_select)

        stats_scroll = ttk.Scrollbar(table_container, orient='vertical', command=self.stats_tree.yview)
        stats_scroll.pack(side='right', fill='y')
        self.stats_tree.configure(yscrollcommand=stats_scroll.set)

        # Initiales Rendering der Diagramme planen
        if self.equity_curve_canvas:
            self.equity_curve_canvas.after(50, lambda: self._draw_equity_curve(None))
        if self.monthly_heatmap_canvas:
            self.monthly_heatmap_canvas.after(50, lambda: self._draw_monthly_profit_heatmap(None))
        if self.profit_distribution_canvas:
            self.profit_distribution_canvas.after(50, lambda: self._draw_profit_distribution(None))
        if self.session_heatmap_canvas:
            self.session_heatmap_canvas.after(50, lambda: self._draw_session_heatmap(None))
        if self.pair_distribution_canvas:
            self.pair_distribution_canvas.after(50, lambda: self._draw_pair_distribution(None))
        if self.accuracy_gauge_canvas:
            self.accuracy_gauge_canvas.after(50, lambda: self._draw_accuracy_gauge(None))

    def _build_dashboard_card(self, parent: ttk.Frame, title: str, value, *, icon: Optional[str] = None,
                              subtitle: Optional[str] = None) -> ttk.Frame:
        """Erzeugt eine kompakte Dashboard-Karte mit optionalem Icon."""
        card = ttk.Frame(parent, style='Card.TFrame', padding=(22, 20))
        if icon:
            icon_frame = ttk.Frame(card, style='Card.TFrame')
            icon_frame.pack(anchor='ne')
            ttk.Label(icon_frame, text=icon, style='CardIcon.TLabel').pack(anchor='e')
        ttk.Label(card, text=title, style='CardTitle.TLabel').pack(anchor='w')
        if isinstance(value, tk.Variable):
            ttk.Label(card, textvariable=value, style='MetricValue.TLabel').pack(anchor='w', pady=(8, 0))
        else:
            ttk.Label(card, text=str(value), style='MetricValue.TLabel').pack(anchor='w', pady=(8, 0))
        if subtitle:
            ttk.Label(card, text=subtitle, style='CardSubtitle.TLabel').pack(anchor='w', pady=(6, 0))
        return card

    def _refresh_symbol_exposure(self):
        """Aktualisiert die Anzeige der Symbol-Exposures."""
        if not self.exposure_list_frame:
            return

        for child in self.exposure_list_frame.winfo_children():
            child.destroy()

        exposures = self._get_symbol_exposure()
        if not exposures:
            ttk.Label(
                self.exposure_list_frame,
                text="Noch keine Trades erfasst",
                style='GlassCardSubtitle.TLabel'
            ).pack(anchor='w')
            return

        for symbol, share in exposures:
            row = ttk.Frame(self.exposure_list_frame, style='GlassCard.TFrame')
            row.pack(fill='x', pady=4)
            ttk.Label(row, text=symbol, style='GlassCardSubtitle.TLabel').pack(side='left')
            bar = ttk.Progressbar(row, style='Accent.Horizontal.TProgressbar', maximum=100, value=share)
            bar.pack(side='left', fill='x', expand=True, padx=(12, 8))
            ttk.Label(row, text=f"{share:.0f}%", style='GlassCardSubtitle.TLabel').pack(side='right')

    def _get_symbol_exposure(self) -> List[tuple]:
        """Berechnet die prozentuale Verteilung der offenen Positionen je Symbol."""
        exposures: Dict[str, float] = {}
        for record in self.bot.trade_tracker.trade_records.values():
            lot_size = max(record.lot_size, 0.0)
            exposures[record.symbol] = exposures.get(record.symbol, 0.0) + lot_size

        total_volume = sum(exposures.values())
        if total_volume > 0:
            sorted_exposures = sorted(exposures.items(), key=lambda item: item[1], reverse=True)[:5]
            return [
                (symbol, min(100.0, (volume / total_volume) * 100.0))
                for symbol, volume in sorted_exposures
            ]

        fallback_total = sum(weight for _, weight in self._default_pair_distribution)
        return [
            (symbol, (weight / fallback_total) * 100.0)
            for symbol, weight in self._default_pair_distribution
        ]

    def _interpolate_color(self, start_hex: str, end_hex: str, factor: float) -> str:
        """Interpoliert linear zwischen zwei Hex-Farben."""
        factor = max(0.0, min(1.0, factor))
        start_rgb = tuple(int(start_hex[i:i + 2], 16) for i in (1, 3, 5))
        end_rgb = tuple(int(end_hex[i:i + 2], 16) for i in (1, 3, 5))
        mixed = tuple(int(s + (e - s) * factor) for s, e in zip(start_rgb, end_rgb))
        return '#{0:02x}{1:02x}{2:02x}'.format(*mixed)

    def _draw_equity_curve(self, event):
        """Zeichnet die Equity-Kurve auf dem Canvas."""
        canvas = event.widget if event else self.equity_curve_canvas
        if not canvas:
            return

        data = self._equity_curve_data or self._default_equity_curve
        if not data:
            return

        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 2 or height <= 2:
            return

        canvas.delete('all')
        canvas.create_rectangle(0, 0, width, height, fill=self.theme_colors['surface_alt'], outline='')

        margin = 32
        min_val = min(data)
        max_val = max(data)
        value_range = max(max_val - min_val, 1)

        points: List[float] = []
        step = (width - 2 * margin) / max(len(data) - 1, 1)
        for idx, value in enumerate(data):
            x = margin + idx * step
            normalized = (value - min_val) / value_range
            y = height - margin - normalized * (height - 2 * margin)
            points.extend((x, y))

        if len(points) >= 4:
            canvas.create_line(points, fill=self.theme_colors['accent'], width=3, smooth=True)
            area_points = [margin, height - margin] + points + [points[-2], height - margin]
            canvas.create_polygon(area_points, fill=self.theme_colors['accent_soft'], outline='')

            last_value = data[-1]
            canvas.create_text(
                points[-2],
                points[-1] - 16,
                text=f"{last_value:,.0f}".replace(',', ' '),
                fill=self.theme_colors['text'],
                font=('Segoe UI Semibold', 11)
            )

    def _draw_profit_distribution(self, event):
        """Zeigt die Profitverteilung als Histogramm an."""
        canvas = event.widget if event else self.profit_distribution_canvas
        if not canvas:
            return

        data = self._profit_distribution_data
        if not data:
            return

        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 2 or height <= 2:
            return

        canvas.delete('all')
        canvas.create_rectangle(0, 0, width, height, fill=self.theme_colors['surface_alt'], outline='')

        margin = 30
        bar_area_width = width - 2 * margin
        bar_width = bar_area_width / max(len(data), 1)
        max_count = max(count for _, count in data) or 1

        for idx, (bucket, count) in enumerate(data):
            normalized = count / max_count
            x0 = margin + idx * bar_width + 4
            x1 = x0 + bar_width - 8
            y1 = height - margin
            y0 = y1 - normalized * (height - 2 * margin)
            color = self.theme_colors['success'] if bucket >= 0 else self.theme_colors['danger']
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='')
            canvas.create_text(
                (x0 + x1) / 2,
                y0 - 10,
                text=str(bucket),
                fill=self.theme_colors['subtle_text'],
                font=('Segoe UI', 9)
            )

        canvas.create_line(margin, height - margin, width - margin, height - margin, fill=self.theme_colors['border'])

    def _draw_monthly_profit_heatmap(self, event):
        """Zeichnet eine Monats-Heatmap."""
        canvas = event.widget if event else self.monthly_heatmap_canvas
        if not canvas:
            return

        data = self._monthly_profit_matrix
        if not data:
            return

        rows = len(data)
        cols = len(data[0]) if rows else 0
        if rows == 0 or cols == 0:
            return

        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 2 or height <= 2:
            return

        canvas.delete('all')
        canvas.create_rectangle(0, 0, width, height, fill=self.theme_colors['surface_alt'], outline='')

        min_val = min(min(row) for row in data)
        max_val = max(max(row) for row in data)
        value_range = max(max_val - min_val, 1)

        margin_x = 40
        margin_y = 24
        cell_width = (width - 2 * margin_x) / cols
        cell_height = (height - 2 * margin_y) / rows

        month_labels = ['Jan', 'Feb', 'Mrz', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']

        for row_idx, row in enumerate(data):
            for col_idx, value in enumerate(row):
                intensity = (value - min_val) / value_range if value_range else 0.0
                color = self._interpolate_color('#13213b', self.theme_colors['highlight'], intensity)
                x0 = margin_x + col_idx * cell_width + 2
                y0 = margin_y + row_idx * cell_height + 2
                x1 = x0 + cell_width - 4
                y1 = y0 + cell_height - 4
                canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='')

        for col_idx in range(cols):
            label = month_labels[col_idx % len(month_labels)]
            x = margin_x + col_idx * cell_width + cell_width / 2
            canvas.create_text(x, margin_y - 10, text=label, fill=self.theme_colors['subtle_text'], font=('Segoe UI', 9))

    def _draw_session_heatmap(self, event):
        """Zeichnet eine Heatmap für Wochentage und Sessions."""
        canvas = event.widget if event else self.session_heatmap_canvas
        if not canvas:
            return

        data = self._current_session_heatmap
        if not data:
            return

        rows = len(data)
        cols = len(data[0]) if rows else 0
        if rows == 0 or cols == 0:
            return

        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 2 or height <= 2:
            return

        canvas.delete('all')
        canvas.create_rectangle(0, 0, width, height, fill=self.theme_colors['surface_alt'], outline='')

        max_value = max(max(row) for row in data) or 1
        margin_x = 40
        margin_y = 24
        cell_width = (width - 2 * margin_x) / cols
        cell_height = (height - 2 * margin_y) / rows

        day_labels = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']
        session_labels = ['00-04h', '04-08h', '08-12h', '12-16h', '16-20h', '20-24h']

        for row_idx, row in enumerate(data):
            for col_idx, value in enumerate(row):
                intensity = value / max_value
                color = self._interpolate_color('#1b253a', self.theme_colors['accent'], intensity)
                x0 = margin_x + col_idx * cell_width + 2
                y0 = margin_y + row_idx * cell_height + 2
                x1 = x0 + cell_width - 4
                y1 = y0 + cell_height - 4
                canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='')

        for col_idx in range(cols):
            canvas.create_text(
                margin_x + col_idx * cell_width + cell_width / 2,
                margin_y - 8,
                text=day_labels[col_idx % len(day_labels)],
                fill=self.theme_colors['subtle_text'],
                font=('Segoe UI', 9)
            )

        for row_idx in range(rows):
            canvas.create_text(
                margin_x - 28,
                margin_y + row_idx * cell_height + cell_height / 2,
                text=session_labels[row_idx % len(session_labels)],
                fill=self.theme_colors['subtle_text'],
                font=('Segoe UI', 9)
            )

    def _draw_pair_distribution(self, event):
        """Zeichnet eine einfache Donut-Grafik für die Paar-Verteilung."""
        canvas = event.widget if event else self.pair_distribution_canvas
        if not canvas:
            return

        distribution = self._current_pair_distribution or self._default_pair_distribution
        if not distribution:
            return

        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 2 or height <= 2:
            return

        canvas.delete('all')
        canvas.create_rectangle(0, 0, width, height, fill=self.theme_colors['surface_alt'], outline='')

        total = sum(weight for _, weight in distribution) or 1
        start_angle = 90
        radius_margin = 16
        palette = [self.theme_colors['accent'], self.theme_colors['success'], self.theme_colors['highlight'], '#a855f7', '#f97316']

        for idx, (symbol, weight) in enumerate(distribution):
            extent = (weight / total) * 360
            color = palette[idx % len(palette)]
            canvas.create_arc(
                radius_margin,
                radius_margin,
                width - radius_margin,
                height - radius_margin,
                start=start_angle,
                extent=extent,
                style='pieslice',
                fill=color,
                outline=self.theme_colors['surface_alt']
            )
            start_angle += extent

        canvas.create_oval(
            radius_margin + 40,
            radius_margin + 40,
            width - radius_margin - 40,
            height - radius_margin - 40,
            fill=self.theme_colors['surface_alt'],
            outline=self.theme_colors['surface_alt']
        )

        center_text = ', '.join(symbol for symbol, _ in distribution[:3])
        canvas.create_text(
            width / 2,
            height / 2,
            text=center_text,
            fill=self.theme_colors['subtle_text'],
            font=('Segoe UI', 9),
            width=width - 120
        )

    def _draw_accuracy_gauge(self, event):
        """Zeichnet eine halbkreisförmige Anzeige für die Genauigkeit."""
        canvas = event.widget if event else self.accuracy_gauge_canvas
        if not canvas:
            return

        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 2 or height <= 2:
            return

        ratio = max(0.0, min(1.0, self._current_accuracy_ratio))

        canvas.delete('all')
        canvas.create_rectangle(0, 0, width, height, fill=self.theme_colors['surface_alt'], outline='')

        margin = 24
        start_angle = 135
        max_extent = 270
        canvas.create_arc(
            margin,
            margin,
            width - margin,
            height - margin,
            start=start_angle,
            extent=max_extent,
            style='arc',
            outline=self.theme_colors['border'],
            width=10
        )
        canvas.create_arc(
            margin,
            margin,
            width - margin,
            height - margin,
            start=start_angle,
            extent=max_extent * ratio,
            style='arc',
            outline=self.theme_colors['success'],
            width=12
        )

        canvas.create_text(
            width / 2,
            height / 2,
            text=f"{ratio:.1f}",
            fill=self.theme_colors['text'],
            font=('Segoe UI Semibold', 16)
        )

    def _create_rule_row(
        self,
        container: ttk.Frame,
        *,
        title: str,
        subtitle: str,
        variable: Optional[tk.BooleanVar] = None,
        command=None,
        condition_parts: Optional[List[tuple]] = None,
        action_parts: Optional[List[tuple]] = None
    ) -> None:
        """Hilfsfunktion zum Aufbau einer Automation-Regel."""
        row = ttk.Frame(container, style='Rule.TFrame', padding=(18, 14))
        row.pack(fill='x', pady=(0, 14))

        header = ttk.Frame(row, style='Rule.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text=title, style='RuleTitle.TLabel').pack(side='left')
        if variable is not None:
            ttk.Checkbutton(header, text="Aktiv", variable=variable, command=command, style='Switch.TCheckbutton').pack(side='right')

        ttk.Label(row, text=subtitle, style='RuleSubtitle.TLabel').pack(anchor='w', pady=(8, 12))

        if condition_parts:
            condition_frame = ttk.Frame(row, style='RuleAccent.TFrame', padding=(12, 10))
            condition_frame.pack(fill='x', pady=(0, 6))
            for text, style in condition_parts:
                ttk.Label(condition_frame, text=text, style=style).pack(side='left', padx=(0, 8))

        if action_parts:
            action_frame = ttk.Frame(row, style='RuleAccent.TFrame', padding=(12, 10))
            action_frame.pack(fill='x')
            for text, style in action_parts:
                ttk.Label(action_frame, text=text, style=style).pack(side='left', padx=(0, 8))

    def _refresh_automation_rules_display(self):
        """Aktualisiert die Darstellung der Automation-Regeln."""
        if not self.automation_rules_container:
            return

        for child in self.automation_rules_container.winfo_children():
            child.destroy()

        self._create_rule_row(
            self.automation_rules_container,
            title="Instant Trades",
            subtitle="Führt Premium-Signale ohne Verzögerung aus.",
            variable=self.instant_trading_var,
            command=lambda key='instant_trading_enabled': self._handle_signal_flag_change(key),
            condition_parts=[
                ("IF", 'RulePillNeutral.TLabel'),
                ("Chat", 'RulePillAccent.TLabel'),
                ("IS PREMIUM", 'RulePillNeutral.TLabel'),
                ("AND", 'RulePillNeutral.TLabel'),
                ("Signal", 'RulePillAccent.TLabel'),
                ("IS INSTANT", 'RulePillNeutral.TLabel')
            ],
            action_parts=[
                ("THEN", 'RulePillNeutral.TLabel'),
                ("Execute trade", 'RulePillAccent.TLabel'),
                ("WITH", 'RulePillNeutral.TLabel'),
                ("Risk Fixed", 'RulePillAccent.TLabel')
            ]
        )

        self._create_rule_row(
            self.automation_rules_container,
            title="Zone Monitoring",
            subtitle="Wartet auf Bestätigung innerhalb definierter Kurszonen.",
            variable=self.zone_trading_var,
            command=lambda key='zone_trading_enabled': self._handle_signal_flag_change(key),
            condition_parts=[
                ("IF", 'RulePillNeutral.TLabel'),
                ("Signal", 'RulePillAccent.TLabel'),
                ("IS ZONE", 'RulePillNeutral.TLabel'),
                ("AND", 'RulePillNeutral.TLabel'),
                ("Price", 'RulePillAccent.TLabel'),
                ("REACHES", 'RulePillNeutral.TLabel'),
                ("Zone", 'RulePillAccent.TLabel')
            ],
            action_parts=[
                ("THEN", 'RulePillNeutral.TLabel'),
                ("Queue trade", 'RulePillAccent.TLabel'),
                ("FOR", 'RulePillNeutral.TLabel'),
                ("Manual review", 'RulePillAccent.TLabel')
            ]
        )

        self._create_rule_row(
            self.automation_rules_container,
            title="Bestätigung erforderlich",
            subtitle="Blockiert Ausführung bei niedriger Signalqualität.",
            variable=self.require_confirmation_var,
            command=lambda key='require_confirmation': self._handle_signal_flag_change(key),
            condition_parts=[
                ("IF", 'RulePillNeutral.TLabel'),
                ("Quality", 'RulePillAccent.TLabel'),
                ("<", 'RulePillNeutral.TLabel'),
                ("0.8", 'RulePillAccent.TLabel')
            ],
            action_parts=[
                ("THEN", 'RulePillNeutral.TLabel'),
                ("Require confirmation", 'RulePillAccent.TLabel')
            ]
        )

        self._create_rule_row(
            self.automation_rules_container,
            title="Automatische SL/TP-Erkennung",
            subtitle="Übernimmt Stop-Loss und Take-Profit aus Telegram-Signalen.",
            variable=self.auto_tp_sl_var,
            command=lambda key='auto_tp_sl': self._handle_signal_flag_change(key),
            condition_parts=[
                ("IF", 'RulePillNeutral.TLabel'),
                ("Message", 'RulePillAccent.TLabel'),
                ("CONTAINS", 'RulePillNeutral.TLabel'),
                ("SL/TP", 'RulePillAccent.TLabel')
            ],
            action_parts=[
                ("THEN", 'RulePillNeutral.TLabel'),
                ("Parse levels", 'RulePillAccent.TLabel'),
                ("AND", 'RulePillNeutral.TLabel'),
                ("Apply automatically", 'RulePillAccent.TLabel')
            ]
        )

    def _on_stats_tree_select(self, _event=None):
        """Aktualisiert die Detail-Ansicht beim Wechsel der Quelle."""
        if not hasattr(self, 'stats_tree'):
            return

        selection = self.stats_tree.selection()
        if not selection:
            self._update_chat_detail_card(None)
            return

        item = selection[0]
        values = self.stats_tree.item(item).get('values')
        chat_name = values[0] if values else None
        self._update_chat_detail_card(chat_name)

    def _update_chat_detail_card(self, chat_name: Optional[str]):
        """Setzt die Detailmetriken auf Basis eines ausgewählten Chats."""
        if not chat_name:
            self.chat_detail_title_var.set("Chat-Statistiken – Auswahl")
            self.detail_winrate_var.set("–")
            self.detail_rr_var.set("–")
            self.detail_signals_var.set("0")
            self.detail_profit_var.set("0 €")
            self._current_session_heatmap = [row[:] for row in self._default_session_heatmap]
            self._current_pair_distribution = list(self._default_pair_distribution)
            self._current_accuracy_ratio = 0.7
        else:
            self.chat_detail_title_var.set(f"Chat-Statistiken – {chat_name}")
            trades = self.bot.trade_tracker.get_trades_by_source(chat_name)
            total_trades = len(trades)
            wins = sum(1 for trade in trades if trade.profit_loss > 0)
            total_profit = sum(trade.profit_loss for trade in trades)
            win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
            avg_gain = sum(trade.profit_loss for trade in trades if trade.profit_loss > 0) / max(wins, 1)
            losses = total_trades - wins
            avg_loss = abs(sum(trade.profit_loss for trade in trades if trade.profit_loss < 0) / max(losses, 1))
            rr = avg_gain / avg_loss if avg_loss else avg_gain if avg_gain else 0.0

            self.detail_winrate_var.set(f"{win_rate:.1f}%")
            self.detail_rr_var.set(f"{rr:.2f}")
            self.detail_signals_var.set(str(total_trades))
            self.detail_profit_var.set(f"{self._format_compact_number(total_profit)} €")

            self._current_session_heatmap = self._calculate_session_heatmap(trades)

            distribution: Dict[str, float] = {}
            for trade in trades:
                distribution[trade.symbol] = distribution.get(trade.symbol, 0.0) + 1.0
            total_distribution = sum(distribution.values()) or 1.0
            self._current_pair_distribution = [
                (symbol, weight / total_distribution)
                for symbol, weight in sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:5]
            ] or list(self._default_pair_distribution)

            self._current_accuracy_ratio = win_rate / 100.0 if total_trades else 0.7

        if self.session_heatmap_canvas:
            self._draw_session_heatmap(None)
        if self.pair_distribution_canvas:
            self._draw_pair_distribution(None)
        if self.accuracy_gauge_canvas:
            self._draw_accuracy_gauge(None)

        if self.chat_detail_risk_var:
            self.chat_detail_risk_var.set(f"{self.risk_percent_var.get():.1f}% Risiko")
        if self.chat_detail_latency_var and self.latency_var:
            self.chat_detail_latency_var.set(self.latency_var.get())
        if self.chat_detail_quality_var and self.alert_score_var:
            self.chat_detail_quality_var.set(self.alert_score_var.get())

    def _calculate_session_heatmap(self, trades: List[TradeRecord]) -> List[List[int]]:
        """Aggregiert Trades nach Wochentag und 4-Stunden-Blöcken."""
        matrix = [[0 for _ in range(7)] for _ in range(6)]
        for trade in trades:
            if not trade.timestamp:
                continue
            day_index = trade.timestamp.weekday()
            hour_block = min(5, trade.timestamp.hour // 4)
            matrix[hour_block][day_index] += 1
        return matrix

    def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Berechnet aggregierte Kennzahlen für alle Trades."""
        trades = sorted(self.bot.trade_tracker.trade_records.values(), key=lambda t: t.timestamp)
        profits = [trade.profit_loss for trade in trades]
        total_trades = len(profits)
        if not profits:
            return {
                'total_profit': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'sharpe': 0.0,
                'sortino': 0.0,
                'max_drawdown': 0.0,
                'equity_curve': list(self._default_equity_curve)
            }

        total_profit = sum(profits)
        wins = sum(1 for value in profits if value > 0)
        win_rate = (wins / total_trades) * 100.0

        mean = total_profit / total_trades
        variance = sum((value - mean) ** 2 for value in profits) / total_trades
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        sharpe = (mean / std_dev) if std_dev else 0.0

        downside = [min(0.0, value) for value in profits]
        downside_variance = sum(value ** 2 for value in downside) / total_trades
        sortino = (mean / math.sqrt(downside_variance)) if downside_variance > 0 else 0.0

        equity = 10_000.0
        equity_curve: List[float] = []
        for profit in profits:
            equity += profit
            equity_curve.append(equity)

        max_drawdown = self._calculate_max_drawdown(equity_curve)

        return {
            'total_profit': total_profit,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve
        }

    def _calculate_max_drawdown(self, equity_series: List[float]) -> float:
        """Berechnet den maximalen Drawdown in Prozent."""
        if not equity_series:
            return 0.0

        peak = equity_series[0]
        max_drawdown = 0.0
        for value in equity_series:
            peak = max(peak, value)
            drawdown = (value - peak) / peak if peak else 0.0
            max_drawdown = min(max_drawdown, drawdown)
        return max_drawdown * 100.0

    def _generate_equity_curve_series(self) -> List[float]:
        """Erzeugt eine Beispiel-Kurve für den Startzustand."""
        base = 35_000.0
        series = []
        for idx in range(90):
            wave = math.sin(idx / 7.5) * 1_200
            trend = idx * 120
            series.append(base + wave + trend)
        return series

    def _generate_profit_distribution(self) -> List[tuple]:
        """Erzeugt Beispielwerte für die Profitverteilung."""
        buckets = [-600, -400, -200, 0, 200, 400, 600]
        counts = [6, 12, 28, 40, 52, 34, 18]
        return list(zip(buckets, counts))

    def _generate_monthly_profit_matrix(self) -> List[List[float]]:
        """Erzeugt eine 4x12 Heatmap mit Beispielzahlen."""
        matrix: List[List[float]] = []
        for row in range(4):
            row_values = []
            for col in range(12):
                value = 50 * math.sin((col + row) / 2.3) + row * 30 + col * 12
                row_values.append(value)
            matrix.append(row_values)
        return matrix

    def _generate_session_heatmap_matrix(self) -> List[List[int]]:
        """Erzeugt eine 6x7 Heatmap mit Beispielaktivität."""
        matrix: List[List[int]] = []
        for row in range(6):
            row_values = []
            for col in range(7):
                value = max(0, int(4 + 3 * math.sin((row + col) / 1.8)))
                row_values.append(value)
            matrix.append(row_values)
        return matrix

    def _format_compact_number(self, value: float) -> str:
        """Formatiert Zahlen mit schmalen Leerzeichen und Vorzeichen."""
        formatted = f"{value:+,.0f}" if abs(value) >= 1 else f"{value:+.2f}"
        return formatted.replace(',', ' ')

    def _update_dashboard_metrics(self, total_sources: int, active_sources: int, total_signals: int):
        """Aktualisiert Kennzahlen im Risk-Monitoring."""
        if self.chat_total_var:
            self.chat_total_var.set(str(total_sources))
        if self.chat_active_var:
            self.chat_active_var.set(str(active_sources))
        if self.chat_signal_sum_var:
            self.chat_signal_sum_var.set(str(total_signals))
        if self.chat_summary_var:
            self.chat_summary_var.set(f"{total_sources} Quellen • {active_sources} aktiv")
        if self.chat_last_sync_var:
            self.chat_last_sync_var.set(f"Letzte Synchronisierung: {datetime.now().strftime('%H:%M')} Uhr")

        open_signals = [trade for trade in self.bot.trade_tracker.trade_records.values() if trade.status == 'open']
        open_count = len(open_signals)
        utilisation = min(100.0, open_count * 12.5)
        if self.open_signal_progress_var:
            self.open_signal_progress_var.set(utilisation)
        if self.open_signal_percentage_var:
            self.open_signal_percentage_var.set(f"{utilisation:.1f}% Auslastung")
        if self.open_signal_active_var:
            text = f"{open_count} aktive Signale" if open_count else "Keine offenen Signale"
            self.open_signal_active_var.set(text)

        compliance_alerts = sum(1 for trade in self.bot.trade_tracker.trade_records.values() if trade.profit_loss < -50)
        if self.dashboard_compliance_var:
            self.dashboard_compliance_var.set(f"{compliance_alerts} Compliance Alerts")

        self._update_status_badges()

    def _update_status_badges(self):
        """Synchronisiert Statusanzeigen für Hero und Dashboard."""
        if self.hero_status_var:
            if self.bot.is_running:
                status = "RUNNING" if self.bot.demo_mode else "LIVE"
            else:
                status = "SAFE"
            self.hero_status_var.set(status)

        if self.dashboard_state_var:
            if self.bot.is_running and not self.bot.demo_mode:
                self.dashboard_state_var.set("LIVE MODE")
            elif self.bot.is_running:
                self.dashboard_state_var.set("SAFE")
            else:
                self.dashboard_state_var.set("STANDBY")

        if self.dashboard_alert_var:
            compliance_alerts = sum(1 for trade in self.bot.trade_tracker.trade_records.values() if trade.profit_loss < -50)
            if compliance_alerts:
                self.dashboard_alert_var.set(f"{compliance_alerts} kritische Warnungen")
            else:
                self.dashboard_alert_var.set("Keine aktiven Warnungen")
    def clear_log(self):
        """Log-Anzeige leeren."""
        if hasattr(self, 'log_text'):
            self.log_text.delete('1.0', 'end')
            self.log_text.insert('end', "--- Log gelöscht ---\n")

    def export_chat_config(self):
        """Chat-Konfiguration sichern."""
        try:
            self.bot.chat_manager.save_config()
            messagebox.showinfo(
                "Export abgeschlossen",
                "Die Chat-Konfiguration wurde unter 'chat_config.json' gespeichert."
            )
        except Exception as exc:
            messagebox.showerror("Fehler", f"Konfiguration konnte nicht gespeichert werden: {exc}")

    def export_statistics(self):
        """Aktuelle Statistikdaten exportieren."""
        try:
            export_data = []
            for chat_source in self.bot.chat_manager.chat_sources.values():
                stats = self.bot.trade_tracker.get_source_statistics(chat_source.chat_name)
                export_data.append({
                    'chat_name': chat_source.chat_name,
                    'total_trades': stats['total_trades'],
                    'win_rate': stats['win_rate'],
                    'total_profit': stats['total_profit'],
                    'last_trade': stats['last_trade'].isoformat() if stats['last_trade'] else None
                })

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'statistics_{timestamp}.json'
            with open(filename, 'w', encoding='utf-8') as export_file:
                json.dump(export_data, export_file, indent=2, ensure_ascii=False)

            messagebox.showinfo("Export abgeschlossen", f"Statistiken wurden in '{filename}' gespeichert.")
        except Exception as exc:
            messagebox.showerror("Fehler", f"Statistiken konnten nicht exportiert werden: {exc}")

    def load_chats(self):
        """Chats laden (async wrapper)"""

        def run_async():
            try:
                future = self.bot.submit_coroutine(self.bot.load_all_chats())
                chats = future.result()
                self.root.after(0, lambda: self.update_chat_list(chats))
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"Fehler beim Laden: {e}"))

        threading.Thread(target=run_async, daemon=True).start()

    def update_chat_list(self, chats_data):
        """Chat-Liste in GUI aktualisieren"""
        for item in self.chats_tree.get_children():
            self.chats_tree.delete(item)

        total_signals = 0
        active_count = 0
        for chat in chats_data:
            chat_source = self.bot.chat_manager.get_chat_info(chat['id'])
            is_monitored = "Ja" if chat_source and chat_source.enabled else "Nein"
            signal_count = chat_source.signal_count if chat_source else 0
            total_signals += signal_count
            if chat_source and chat_source.enabled:
                active_count += 1

            self.chats_tree.insert('', 'end', values=(
                chat['name'],
                chat['id'],
                chat['type'],
                chat['participants'],
                is_monitored,
                signal_count
            ))

        self.status_label.config(text=f"Chats geladen: {len(chats_data)}")
        self._update_dashboard_metrics(len(chats_data), active_count, total_signals)
        self._refresh_symbol_exposure()

    def enable_monitoring(self):
        """Überwachung für ausgewählte Chats aktivieren"""
        selection = self.chats_tree.selection()
        if not selection:
            messagebox.showwarning("Keine Auswahl", "Bitte wählen Sie Chats aus.")
            return

        for item in selection:
            values = self.chats_tree.item(item)['values']
            chat_id = int(values[1])
            chat_name = values[0]
            chat_type = values[2]

            self.bot.chat_manager.add_chat_source(chat_id, chat_name, chat_type, True)

            new_values = list(values)
            new_values[4] = "Ja"
            self.chats_tree.item(item, values=new_values)

        messagebox.showinfo("Erfolg", f"{len(selection)} Chat(s) aktiviert")

    def disable_monitoring(self):
        """Überwachung deaktivieren"""
        selection = self.chats_tree.selection()
        if not selection:
            messagebox.showwarning("Keine Auswahl", "Bitte wählen Sie Chats aus.")
            return

        for item in selection:
            values = self.chats_tree.item(item)['values']
            chat_id = int(values[1])

            chat_source = self.bot.chat_manager.get_chat_info(chat_id)
            if chat_source:
                chat_source.enabled = False

            new_values = list(values)
            new_values[4] = "Nein"
            self.chats_tree.item(item, values=new_values)

        self.bot.chat_manager.save_config()
        messagebox.showinfo("Erfolg", f"{len(selection)} Chat(s) deaktiviert")

    def refresh_statistics(self):
        """Statistiken aktualisieren"""
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

        total_sources = len(self.bot.chat_manager.chat_sources)
        active_sources = 0
        total_signals = 0

        for chat_source in self.bot.chat_manager.chat_sources.values():
            stats = self.bot.trade_tracker.get_source_statistics(chat_source.chat_name)
            if chat_source.enabled:
                active_sources += 1
            total_signals += chat_source.signal_count

            last_trade = "Nie"
            if stats['last_trade']:
                last_trade = stats['last_trade'].strftime("%d.%m %H:%M")

            self.stats_tree.insert('', 'end', values=(
                chat_source.chat_name,
                stats['total_trades'],
                f"{stats['win_rate']:.1f}%",
                self._format_compact_number(stats['total_profit']),
                last_trade
            ))

        metrics = self._calculate_portfolio_metrics()
        total_profit = metrics['total_profit']
        win_rate = metrics['win_rate']
        sharpe = metrics['sharpe']
        sortino = metrics['sortino']
        max_drawdown = metrics['max_drawdown']

        self.stats_profit_var.set(f"{self._format_compact_number(total_profit)} €")
        self.stats_win_rate_var.set(f"{win_rate:.1f}%")
        self.stats_sharpe_var.set(f"{sharpe:.2f}")
        self.stats_sortino_var.set(f"{sortino:.2f}")
        self.stats_drawdown_var.set(f"{max_drawdown:.1f}%")

        self._equity_curve_data = metrics['equity_curve'] or list(self._default_equity_curve)
        if self.equity_curve_canvas:
            self._draw_equity_curve(None)
        if self.profit_distribution_canvas:
            self._draw_profit_distribution(None)

        today = datetime.now().date()
        daily_profit = sum(
            trade.profit_loss
            for trade in self.bot.trade_tracker.trade_records.values()
            if trade.timestamp and trade.timestamp.date() == today
        )
        if self.dashboard_daily_loss_var:
            self.dashboard_daily_loss_var.set(f"{self._format_compact_number(daily_profit)} €")
        if self.dashboard_drawdown_var:
            self.dashboard_drawdown_var.set(f"{max_drawdown:.1f}%")

        if self.alert_score_var:
            score = min(0.95, 0.45 + win_rate / 200.0)
            self.alert_score_var.set(f"{score:.2f}")

        self._update_dashboard_metrics(total_sources, active_sources, total_signals)
        self._refresh_symbol_exposure()

        if self.stats_tree.get_children():
            selection = self.stats_tree.selection()
            if not selection:
                first_item = self.stats_tree.get_children()[0]
                self.stats_tree.selection_set(first_item)
            self._on_stats_tree_select()
        else:
            self._update_chat_detail_card(None)

        if hasattr(self, 'statistics_hint'):
            self.statistics_hint.config(
                text=f"Letzte Aktualisierung: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
            )

    def toggle_demo_mode(self):
        """Demo-Modus umschalten"""
        desired_demo_mode = bool(self.demo_var.get())
        if not desired_demo_mode:
            if not self.bot.ensure_mt5_session(enforce_demo_on_fail=False):
                message = self.bot.get_last_mt5_error() or "LIVE-Modus konnte nicht aktiviert werden."
                self.demo_var.set(True)
                self.bot.demo_mode = True
                if hasattr(self, 'trade_status_label'):
                    self.trade_status_label.config(text="Demo-Modus aktiv")
                self.log_message(message)
                try:
                    messagebox.showwarning("LIVE-Modus nicht verfügbar", message)
                except Exception:
                    pass
                return

        self.bot.demo_mode = desired_demo_mode
        mode_text = "Demo-Modus" if self.bot.demo_mode else "LIVE-Modus"
        if hasattr(self, 'trade_status_label'):
            status_text = "Demo-Modus aktiv" if self.bot.demo_mode else "LIVE-Modus aktiv"
            style = 'BadgeSuccess.TLabel' if self.bot.demo_mode else 'BadgeWarning.TLabel'
            self.trade_status_label.config(text=status_text, style=style)

        trading_cfg = self.current_config.setdefault('trading', {})
        previous_value = trading_cfg.get('demo_mode')
        trading_cfg['demo_mode'] = self.bot.demo_mode
        if previous_value != self.bot.demo_mode:
            try:
                self.config_manager.save_config(self.current_config)
            except Exception as exc:
                self.log_message(f"Fehler beim Speichern des Modus: {exc}")

        self.log_message(f"Modus geändert: {mode_text}")
        self._update_status_badges()

    def on_execution_mode_change(self, *_):
        """Ausführungsmodus wechseln und speichern."""
        selected_label = self.execution_mode_var.get() if hasattr(self, 'execution_mode_var') else None
        if not selected_label:
            return

        mode = self.execution_mode_label_to_enum.get(selected_label)
        if not mode:
            self.log_message(f"Unbekannter Ausführungsmodus ausgewählt: {selected_label}")
            return

        if getattr(self.bot, 'execution_mode', ExecutionMode.INSTANT) != mode:
            self.bot.execution_mode = mode

        trading_cfg = self.current_config.setdefault('trading', {})
        previous_value = trading_cfg.get('execution_mode')
        trading_cfg['execution_mode'] = mode.value

        try:
            if previous_value != mode.value:
                self.config_manager.save_config(self.current_config)
        except Exception as exc:
            self.log_message(f"Fehler beim Speichern des Ausführungsmodus: {exc}")
        else:
            if previous_value != mode.value:
                self.log_message(f"Ausführungsmodus geändert zu: {selected_label}")

        self._refresh_automation_rules_display()

    def start_bot(self):
        """Bot starten"""
        if self.bot.is_running or self.bot_starting:
            return

        self.bot_starting = True
        if self.start_button:
            self.start_button.config(state='disabled')

        def run_bot():
            started = False
            try:
                future = self.bot.submit_coroutine(self.bot.start())
                started = future.result()
                if started:
                    self.root.after(0, self.after_bot_started)
                else:
                    self.root.after(0, self.handle_bot_start_failure)
            except Exception as e:
                self.root.after(0, lambda e=e: self.handle_bot_start_exception(e))

        threading.Thread(target=run_bot, daemon=True).start()

    def after_bot_started(self):
        """Aktionen nach erfolgreichem Start"""
        self.status_label.config(text="Bot läuft")
        self.bot_starting = False
        if self.start_button:
            self.start_button.config(state='normal')
        self._update_status_badges()

    def handle_bot_start_failure(self):
        """Fehler beim Starten behandeln"""
        self.log_message(
            "Bot konnte nicht gestartet werden. Bitte prüfen Sie die Telegram-Konfiguration."
        )
        self.after_bot_stopped()

    def handle_bot_start_exception(self, error: Exception):
        """Ausnahme beim Starten behandeln"""
        self.log_message(f"Bot-Start-Fehler: {error}")
        self.after_bot_stopped()

    def stop_bot(self):
        """Bot stoppen"""
        self.bot.is_running = False
        client = self.bot.client
        if client:
            try:
                future = self.bot.submit_coroutine(client.disconnect())
                future.result(timeout=10)
            except Exception:
                pass
        self.after_bot_stopped()

    def after_bot_stopped(self):
        """Aktionen nach dem Stoppen"""
        self.status_label.config(text="Bot gestoppt")
        self.bot_starting = False
        if self.start_button:
            self.start_button.config(state='normal')
        self._update_status_badges()

    def setup_message_processing(self):
        """Message Queue Processing"""
        def process_messages():
            try:
                while True:
                    msg_type, data = self.bot.message_queue.get(block=False)

                    if msg_type == 'LOG':
                        self.log_message(str(data))
                    elif msg_type == 'TRADE_EXECUTED':
                        self.log_message(f"Trade ausgeführt: {data}")
                    elif msg_type == 'AUTH_REQUIRED':
                        info_message = data.get('message') if isinstance(data, dict) else str(data)
                        self.root.after(
                            0,
                            lambda msg=info_message: self.show_auth_required_dialog(msg)
                        )
                    elif msg_type == 'DEMO_MODE_ENFORCED':
                        self._handle_demo_mode_enforced(data)
                    elif msg_type == 'CONFIRM_TRADE':
                        self._handle_trade_confirmation_request(data)

            except queue.Empty:
                pass

            self.root.after(100, process_messages)

        process_messages()

    def _handle_demo_mode_enforced(self, data):
        """Setzt den Demo-Modus in der GUI und informiert den Nutzer."""
        message = ""
        if isinstance(data, dict):
            message = str(data.get('message', '')).strip()
        elif data is not None:
            message = str(data).strip()

        if hasattr(self, 'demo_var'):
            self.demo_var.set(True)
        self.bot.demo_mode = True

        if hasattr(self, 'trade_status_label'):
            self.trade_status_label.config(text="Demo-Modus aktiv")

        trading_cfg = self.current_config.setdefault('trading', {})
        previous_value = trading_cfg.get('demo_mode')
        trading_cfg['demo_mode'] = True
        if previous_value is not True:
            try:
                self.config_manager.save_config(self.current_config)
            except Exception as exc:
                self.log_message(f"Konfiguration konnte nicht gespeichert werden: {exc}")

        if message:
            self.log_message(message)
            try:
                messagebox.showwarning("Live-Modus deaktiviert", message)
            except Exception:
                pass

    def _handle_trade_confirmation_request(self, data):
        """Zeigt einen Bestätigungsdialog für eingehende Trades an."""

        if not isinstance(data, dict):
            return

        future = data.get('future')
        signal = data.get('signal') or {}
        chat_name = data.get('chat_name', 'Unbekannt')
        original_message = data.get('message') or ''

        action = (signal.get('action') or 'TRADE').upper()
        symbol = signal.get('symbol', 'Unbekannt')
        entry_price = signal.get('entry_price')
        stop_loss = signal.get('stop_loss')
        take_profits = signal.get('take_profits') or []

        def _format_price(value) -> str:
            try:
                return f"{float(value):.5f}"
            except (TypeError, ValueError):
                return str(value)

        detail_parts = []
        if action and symbol:
            detail_parts.append(f"{action} {symbol}")
        elif symbol:
            detail_parts.append(str(symbol))
        elif action:
            detail_parts.append(str(action))

        if entry_price is not None:
            detail_parts.append(f"@ {_format_price(entry_price)}")

        detail_text = ' '.join(part for part in detail_parts if part).strip()

        level_lines = []
        if stop_loss is not None:
            level_lines.append(f"SL: {_format_price(stop_loss)}")
        if take_profits:
            formatted_tps = ', '.join(_format_price(tp) for tp in take_profits)
            level_lines.append(f"TPs: {formatted_tps}")

        preview_text = original_message.strip()
        if preview_text and len(preview_text) > 400:
            preview_text = preview_text[:397] + '...'

        prompt_lines = [f"Neues Signal von {chat_name}"]
        if detail_text:
            prompt_lines.append(detail_text)
        if level_lines:
            prompt_lines.extend(level_lines)
        if preview_text:
            prompt_lines.append('')
            prompt_lines.append('Originalnachricht:')
            prompt_lines.append(preview_text)

        prompt = '\n'.join(prompt_lines)

        confirmed = False
        try:
            confirmed = bool(
                messagebox.askyesno(
                    "Trade bestätigen",
                    prompt,
                    parent=self.root
                )
            )
        except Exception as exc:
            self.log_message(f"Fehler beim Anzeigen des Bestätigungsdialogs: {exc}")
            confirmed = False
        finally:
            future_obj = future if isinstance(future, Future) else None
            if future_obj and not future_obj.done():
                future_obj.set_result(bool(confirmed))

    def apply_config(self, config: Dict):
        """Konfiguration auf Bot und GUI anwenden"""
        self.current_config = config
        telegram_cfg = config.get('telegram', {})
        session_name = telegram_cfg.get('session_name', 'trading_session')
        self.bot.update_credentials(
            telegram_cfg.get('api_id'),
            telegram_cfg.get('api_hash'),
            telegram_cfg.get('phone'),
            session_name=session_name
        )

        trading_cfg = self.current_config.setdefault('trading', {})
        demo_mode = bool(trading_cfg.get('demo_mode', True))
        self.bot.demo_mode = demo_mode
        if hasattr(self, 'demo_var'):
            self.demo_var.set(demo_mode)
        if hasattr(self, 'trade_status_label'):
            self.trade_status_label.config(text="Demo-Modus aktiv" if demo_mode else "LIVE-Modus aktiv")

        execution_mode_value = trading_cfg.get('execution_mode', ExecutionMode.INSTANT.value)
        try:
            execution_mode = ExecutionMode(execution_mode_value)
        except ValueError:
            execution_mode = ExecutionMode.INSTANT
        self.bot.execution_mode = execution_mode

        if hasattr(self, 'execution_mode_var'):
            label = self.execution_mode_labels.get(execution_mode, self.execution_mode_labels[ExecutionMode.INSTANT])
            self.execution_mode_var.set(label)

        trading_defaults = self.config_manager.default_config.get('trading', {})
        sanitized_values = {
            'default_lot_size': self._coerce_to_float(
                trading_cfg.get('default_lot_size'),
                trading_defaults.get('default_lot_size', 0.01)
            ),
            'max_spread_pips': self._coerce_to_float(
                trading_cfg.get('max_spread_pips'),
                trading_defaults.get('max_spread_pips', 0.0)
            ),
            'risk_percent': self._coerce_to_float(
                trading_cfg.get('risk_percent'),
                trading_defaults.get('risk_percent', 0.0)
            ),
            'max_trades_per_hour': self._coerce_to_int(
                trading_cfg.get('max_trades_per_hour'),
                trading_defaults.get('max_trades_per_hour', 0)
            )
        }

        trading_cfg.update(sanitized_values)

        self._updating_trading_vars = True
        try:
            self.default_lot_var.set(sanitized_values['default_lot_size'])
            self.max_spread_var.set(sanitized_values['max_spread_pips'])
            self.risk_percent_var.set(sanitized_values['risk_percent'])
            self.max_trades_per_hour_var.set(sanitized_values['max_trades_per_hour'])
        finally:
            self._updating_trading_vars = False

        self.bot.default_lot_size = sanitized_values['default_lot_size']
        self.bot.max_spread_pips = sanitized_values['max_spread_pips']
        self.bot.risk_percent = sanitized_values['risk_percent']
        self.bot.max_trades_per_hour = sanitized_values['max_trades_per_hour']

        mt5_cfg = self.current_config.setdefault('mt5', {})
        login_value = self._coerce_to_str(mt5_cfg.get('login', ""))
        password_value = self._coerce_to_str(mt5_cfg.get('password', ""))
        server_value = self._coerce_to_str(mt5_cfg.get('server', ""))
        path_value = self._coerce_to_str(mt5_cfg.get('path', ""))

        try:
            self.mt5_login_var.set(login_value)
            self.mt5_password_var.set(password_value)
            self.mt5_server_var.set(server_value)
            self.mt5_path_var.set(path_value)
        except tk.TclError:
            pass

        self.bot.update_mt5_credentials(
            login_value.strip() or None,
            password_value or None,
            server_value.strip() or None,
            path_value.strip() or None
        )

        self._refresh_mt5_status_display()

        signal_defaults = self.config_manager.default_config.get('signals', {})
        signals_cfg = self.current_config.setdefault('signals', {})

        self._updating_signal_flags = True
        try:
            for key, var in self._signal_flag_vars.items():
                target_value = self._coerce_to_bool(
                    signals_cfg.get(key, signal_defaults.get(key, True)),
                    signal_defaults.get(key, True)
                )
                var.set(target_value)
                signals_cfg[key] = target_value
                self._apply_signal_flag_to_bot(key, target_value)
        finally:
            self._updating_signal_flags = False

    def log_message(self, message):
        """Log-Nachricht in GUI anzeigen"""
        self.log_text.insert('end', f"{message}\n")
        self.log_text.see('end')

    def _is_valid_mt5_login(self, login: Optional[object]) -> bool:
        """Prüft, ob der übergebene MT5-Login numerisch ist."""
        if login is None:
            return False

        try:
            login_str = str(login).strip()
        except Exception:
            return False

        if not login_str:
            return False

        try:
            int(login_str)
        except (TypeError, ValueError):
            return False

        return True

    def _collect_mt5_form_data(self):
        """Liest die MT5-Formularwerte aus."""
        try:
            login = self.mt5_login_var.get()
            password = self.mt5_password_var.get()
            server = self.mt5_server_var.get()
            path = self.mt5_path_var.get()
        except tk.TclError:
            return "", "", "", ""

        return login.strip(), password, server.strip(), path.strip()

    def _refresh_mt5_status_display(self, message: Optional[str] = None, level: Optional[str] = None):
        """Aktualisiert die Statusanzeige für die MT5-Zugangsdaten."""
        if not self.mt5_status_message_var or not self.mt5_status_label:
            return

        style_map = {
            'info': 'Info.TLabel',
            'success': 'Success.TLabel',
            'warning': 'Warning.TLabel',
            'error': 'Warning.TLabel'
        }

        if not MT5_AVAILABLE:
            if not message:
                message = "MetaTrader5-Python-Modul ist nicht verfügbar."
            level = level or 'warning'
        elif message is None:
            login, password, server, _ = self._collect_mt5_form_data()
            login_present = bool(login)
            password_present = bool(password)
            server_present = bool(server)
            login_valid = self._is_valid_mt5_login(login)
            invalid_login = login_present and not login_valid

            if not (login_present or password_present or server_present):
                message = "Noch keine MT5-Zugangsdaten gespeichert."
                level = 'info'
            elif invalid_login:
                message = self.MT5_INVALID_LOGIN_MESSAGE
                level = 'warning'
            elif not (login_present and password_present and server_present):
                message = "MT5-Zugangsdaten unvollständig. Bitte Login, Passwort und Server angeben."
                level = 'warning'
            else:
                message = "MT5-Zugangsdaten geladen."
                level = 'success'

        level = level or 'info'
        style = style_map.get(level, 'Info.TLabel')
        self.mt5_status_label.configure(style=style)
        self.mt5_status_message_var.set(message or "")

    def save_mt5_credentials(self, silent: bool = False) -> bool:
        """Speichert MT5-Zugangsdaten in der Konfiguration."""
        login, password, server, path = self._collect_mt5_form_data()

        if login and not self._is_valid_mt5_login(login):
            warning_message = self.MT5_INVALID_LOGIN_MESSAGE
            self.log_message(warning_message)
            self._refresh_mt5_status_display(warning_message, level='warning')
            return False

        mt5_cfg = self.current_config.setdefault('mt5', {})
        mt5_cfg['login'] = login
        mt5_cfg['password'] = password
        mt5_cfg['server'] = server
        mt5_cfg['path'] = path

        self.bot.update_mt5_credentials(
            login or None,
            password or None,
            server or None,
            path or None
        )

        try:
            self.config_manager.save_config(self.current_config)
        except Exception as exc:
            error_message = f"Fehler beim Speichern der MT5-Daten: {exc}"
            self.log_message(error_message)
            self._refresh_mt5_status_display(error_message, level='warning')
            return False

        self._refresh_mt5_status_display()
        if not silent:
            self.log_message("MT5-Zugangsdaten aktualisiert.")
        return True

    def test_mt5_connection(self):
        """Testet die Verbindung zu MetaTrader 5."""
        if not MT5_AVAILABLE:
            message = "MetaTrader5-Python-Modul ist nicht verfügbar. Installieren Sie es, um den LIVE-Modus zu nutzen."
            self.log_message(message)
            self._refresh_mt5_status_display(message, level='warning')
            try:
                messagebox.showwarning("MT5 nicht verfügbar", message)
            except Exception:
                pass
            return

        login, password, server, path = self._collect_mt5_form_data()

        if login and not self._is_valid_mt5_login(login):
            warning_message = self.MT5_INVALID_LOGIN_MESSAGE
            self.log_message(warning_message)
            self._refresh_mt5_status_display(warning_message, level='warning')
            try:
                messagebox.showwarning("Ungültiger MT5-Login", warning_message)
            except Exception:
                pass
            return

        if not self.save_mt5_credentials(silent=True):
            return

        if not login or not password or not server:
            message = "Bitte geben Sie Login, Passwort und Server an, bevor Sie die Verbindung testen."
            self.log_message(message)
            self._refresh_mt5_status_display(message, level='warning')
            try:
                messagebox.showwarning("Angaben unvollständig", message)
            except Exception:
                pass
            return

        self.log_message("Teste MT5-Verbindung ...")
        success = self.bot.ensure_mt5_session(enforce_demo_on_fail=False)
        if success:
            message = "MT5-Verbindung erfolgreich aufgebaut."
            self.log_message(message)
            self._refresh_mt5_status_display(message, level='success')
            try:
                messagebox.showinfo("Verbindung erfolgreich", message)
            except Exception:
                pass
        else:
            message = self.bot.get_last_mt5_error() or "MT5-Verbindung konnte nicht hergestellt werden."
            self.log_message(message)
            self._refresh_mt5_status_display(message, level='warning')
            try:
                messagebox.showerror("Verbindung fehlgeschlagen", message)
            except Exception:
                pass

    def browse_mt5_path(self):
        """Dateidialog zum Auswählen des MT5-Terminals öffnen."""
        if not MT5_AVAILABLE:
            message = "MetaTrader5-Python-Modul ist nicht verfügbar."
            self.log_message(message)
            try:
                messagebox.showwarning("MT5 nicht verfügbar", message)
            except Exception:
                pass
            return

        try:
            selected = filedialog.askopenfilename(
                title="MetaTrader-5-Terminal auswählen",
                filetypes=[
                    ("MetaTrader Terminal", "terminal64.exe"),
                    ("Executable", "*.exe"),
                    ("Alle Dateien", "*.*")
                ]
            )
        except Exception as exc:
            self.log_message(f"Fehler beim Öffnen des Dateidialogs: {exc}")
            return

        if not selected:
            return

        try:
            self.mt5_path_var.set(selected)
        except tk.TclError:
            self.log_message("Der ausgewählte Pfad konnte nicht übernommen werden.")
            return

        if self.save_mt5_credentials(silent=True):
            self.log_message("MT5-Terminalpfad aktualisiert.")

    def _add_trading_var_trace(self, key: str, var: tk.Variable, caster):
        """Trace für Trading-Variablen registrieren."""

        def _callback(*_):
            self._handle_trading_var_change(key, var, caster)

        var.trace_add('write', _callback)

    def _handle_trading_var_change(self, key: str, var: tk.Variable, caster):
        """Änderungen an Trading-Variablen anwenden und speichern."""
        if self._updating_trading_vars:
            return

        try:
            current_value = var.get()
        except tk.TclError:
            return

        try:
            new_value = caster(current_value)
        except (TypeError, ValueError):
            return

        if caster is float:
            new_value = round(float(new_value), 4)
            try:
                previous_value = float(self.current_config.setdefault('trading', {}).get(key, new_value))
            except (TypeError, ValueError):
                previous_value = None
            value_changed = previous_value is None or abs(previous_value - new_value) > 1e-6
        else:
            previous_value = self.current_config.setdefault('trading', {}).get(key)
            try:
                value_changed = int(previous_value) != int(new_value)
            except (TypeError, ValueError):
                value_changed = True

        setattr(self.bot, key, new_value)

        if not value_changed:
            return

        trading_cfg = self.current_config.setdefault('trading', {})
        trading_cfg[key] = new_value

        try:
            self.config_manager.save_config(self.current_config)
        except Exception as exc:
            self.log_message(f"Fehler beim Speichern der Trading-Einstellungen: {exc}")

    def _apply_signal_flag_to_bot(self, key: str, value: bool):
        """Aktualisiert die entsprechenden Flags auf dem Bot."""
        value = bool(value)
        if key == 'auto_tp_sl':
            signal_processor = getattr(self.bot, 'signal_processor', None)
            if signal_processor is not None:
                signal_processor.auto_tp_sl = value
        elif key == 'instant_trading_enabled':
            self.bot.instant_trading_enabled = value
        elif key == 'zone_trading_enabled':
            self.bot.zone_trading_enabled = value
        elif key == 'require_confirmation':
            self.bot.require_confirmation = value

    def _handle_signal_flag_change(self, key: str):
        """Callback für Checkbuttons der Signal-Konfiguration."""
        if self._updating_signal_flags:
            return

        var = self._signal_flag_vars.get(key)
        if var is None:
            return

        try:
            value = bool(var.get())
        except tk.TclError:
            return

        signals_cfg = self.current_config.setdefault('signals', {})
        defaults = self.config_manager.default_config.get('signals', {})
        previous_value = self._coerce_to_bool(
            signals_cfg.get(key, defaults.get(key, True)),
            defaults.get(key, True)
        )

        signals_cfg[key] = value
        self._apply_signal_flag_to_bot(key, value)

        if previous_value == value:
            return

        try:
            self.config_manager.save_config(self.current_config)
        except Exception as exc:
            self.log_message(f"Fehler beim Speichern der Signal-Einstellungen: {exc}")
        else:
            label = self._signal_flag_labels.get(key, key)
            state_text = "aktiviert" if value else "deaktiviert"
            self.log_message(f"{label} wurde {state_text}.")

        self._refresh_automation_rules_display()

    def _validate_float_value(self, proposed: str, min_value: str, _setting_key: str) -> bool:
        """Validiert Float-Eingaben für Spinboxen."""
        if proposed is None:
            return False

        proposed = proposed.strip()
        if not proposed:
            return False

        try:
            value = float(proposed)
        except ValueError:
            return False

        try:
            minimum = float(min_value)
        except (TypeError, ValueError):
            minimum = 0.0

        return value >= minimum

    def _validate_int_value(self, proposed: str, min_value: str, _setting_key: str) -> bool:
        """Validiert Integer-Eingaben für Spinboxen."""
        if proposed is None:
            return False

        proposed = proposed.strip()
        if not proposed:
            return False

        try:
            value = int(proposed)
        except ValueError:
            try:
                value = int(float(proposed))
            except (TypeError, ValueError):
                return False

        try:
            minimum = int(float(min_value))
        except (TypeError, ValueError):
            minimum = 0

        return value >= minimum

    def _coerce_to_str(self, value, default: str = "") -> str:
        """Hilfsfunktion zur String-Normalisierung."""
        if value is None:
            return default or ""
        if isinstance(value, str):
            return value
        try:
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value)
        except Exception:
            return default or ""

    def _coerce_to_float(self, value, default: float) -> float:
        """Hilfsfunktion zur sicheren Float-Konvertierung."""
        try:
            if isinstance(value, str):
                value = value.replace(',', '.').strip()
            return float(value)
        except (TypeError, ValueError):
            try:
                if isinstance(default, str):
                    default = default.replace(',', '.').strip()
                return float(default)
            except (TypeError, ValueError):
                return 0.0

    def _coerce_to_int(self, value, default: int) -> int:
        """Hilfsfunktion zur sicheren Integer-Konvertierung."""
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                if isinstance(value, str):
                    value = value.replace(',', '.').strip()
                return int(float(value))
            except (TypeError, ValueError):
                try:
                    return int(default)
                except (TypeError, ValueError):
                    return 0

    def _coerce_to_bool(self, value, default: bool) -> bool:
        """Hilfsfunktion zur Bool-Konvertierung."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {'1', 'true', 'wahr', 'yes', 'ja', 'y', 'on'}:
                return True
            if lowered in {'0', 'false', 'falsch', 'no', 'nein', 'n', 'off'}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return bool(default)

    def show_auth_required_dialog(self, message: str):
        """Dialog anzeigen, wenn ein Login-Code erforderlich ist."""
        if message != self._last_auth_message:
            self.log_message(message)
            self._last_auth_message = message

        if self._auth_dialog_open:
            self._pending_auth_message = message
            return

        self._auth_dialog_open = True
        try:
            dialog = AuthCodeDialog(self.root, message)
            code = getattr(dialog, 'code', None)
        finally:
            self._auth_dialog_open = False

        if code:
            self.log_message("Login-Code erhalten. Authentifizierung wird geprüft...")
            self.verify_login_code(code)
        else:
            self.log_message("Login-Code-Eingabe abgebrochen oder ohne Eingabe geschlossen.")

        pending_message = self._pending_auth_message
        self._pending_auth_message = None
        self._last_auth_message = None

        if pending_message:
            self.root.after(0, lambda msg=pending_message: self.show_auth_required_dialog(msg))

    def verify_login_code(self, code: str):
        """Login-Code asynchron prüfen."""

        def run_verification():
            try:
                future = self.bot.submit_coroutine(self.bot.complete_login_with_code(code))
                result = future.result()
                self.root.after(0, lambda res=result: self.handle_login_code_result(res))
            except Exception as e:
                self.root.after(0, lambda err=e: self.handle_login_code_exception(err))

        threading.Thread(target=run_verification, daemon=True).start()

    def handle_login_code_result(self, result: Dict):
        """Ergebnis der Login-Code-Prüfung verarbeiten."""
        if isinstance(result, dict) and result.get('success'):
            self.status_label.config(text="Telegram-Login erfolgreich. Bot wird gestartet...")
            self.log_message("Telegram-Login erfolgreich. Bot wird erneut gestartet.")
            self.start_bot()
        elif isinstance(result, dict) and result.get('require_password'):
            message = result.get('message') if isinstance(result, dict) else None
            messagebox.showerror(
                "Telegram 2FA erforderlich",
                message or (
                    "Telegram erfordert zusätzlich ein Passwort (2FA). "
                    "Bitte geben Sie das Passwort in der Telegram-App ein."
                )
            )

    def handle_login_code_exception(self, error: Exception):
        """Fehler bei der Login-Code-Prüfung behandeln."""
        self.log_message(f"Fehler bei der Telegram-Authentifizierung: {error}")
        messagebox.showerror(
            "Telegram-Login fehlgeschlagen",
            f"Fehler bei der Verarbeitung des Login-Codes: {error}"
        )

    def run(self):
        """GUI starten"""
        self.root.mainloop()


# ==================== SETUP ASSISTANT & STARTUP ====================

class SetupAssistant:
    """Setup-Assistent für erste Konfiguration"""

    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.window = None
        self.config_saved = False
        self.config_manager = ConfigManager()
        self.current_config = self.config_manager.load_config()

    def show_setup_dialog(self):
        """Setup-Dialog anzeigen"""
        # --- Window setup & theme ---
        self.window = tk.Toplevel(self.parent)
        self.window.title("Welcome · Telegram Copier")
        self.window.geometry("760x620")
        self.window.configure(bg="#050B16")
        self.window.resizable(False, False)

        background = "#050B16"
        surface = "#0B1624"
        neon_blue = "#38BDF8"
        neon_green = "#34D399"
        subdued = "#94A3B8"

        style = ttk.Style(self.window)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        style.configure("DarkRoot.TFrame", background=background)
        style.configure("Dark.TFrame", background=background)
        style.configure("Title.TLabel", background=background, foreground="#F8FAFC",
                        font=("Segoe UI Semibold", 20))
        style.configure("Subtitle.TLabel", background=background, foreground=neon_blue,
                        font=("Segoe UI", 12))
        style.configure("Dark.TLabel", background=background, foreground="#CBD5F5",
                        font=("Segoe UI", 10))
        style.configure("Accent.TLabel", background=background, foreground=neon_blue,
                        font=("Segoe UI Semibold", 10))
        style.configure("Card.TFrame", background=surface)
        style.configure("CardTitle.TLabel", background=surface, foreground="#F8FAFC",
                        font=("Segoe UI Semibold", 13))
        style.configure("Card.TLabel", background=surface, foreground="#E2E8F0",
                        font=("Segoe UI", 10))
        style.configure("MutedCard.TLabel", background=surface, foreground=subdued,
                        font=("Segoe UI", 9))
        style.configure("Dark.TCheckbutton", background=surface, foreground="#E2E8F0",
                        font=("Segoe UI", 10))
        style.map("Dark.TCheckbutton",
                  background=[('active', surface)],
                  foreground=[('disabled', '#475569')])
        style.configure("Dark.TButton", background=background, foreground="#E2E8F0",
                        font=("Segoe UI", 10))
        style.map("Dark.TButton",
                  background=[('active', '#0B182A')],
                  foreground=[('disabled', '#475569')])
        entry_style = "Dark.TEntry"
        style.configure(entry_style,
                        foreground="#F8FAFC",
                        fieldbackground="#0F1E33",
                        background="#0F1E33",
                        bordercolor="#1E293B",
                        insertcolor="#F8FAFC")
        style.map(entry_style,
                  fieldbackground=[('focus', '#132542')],
                  bordercolor=[('focus', neon_blue)],
                  foreground=[('disabled', '#64748B')])

        # --- Helper utilities for drawing custom shapes ---
        def hex_to_rgb(value: str) -> tuple[int, int, int]:
            value = value.lstrip('#')
            return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))

        def blend(color_a: str, color_b: str, factor: float) -> str:
            ra, ga, ba = hex_to_rgb(color_a)
            rb, gb, bb = hex_to_rgb(color_b)
            blended = (
                int(ra + (rb - ra) * factor),
                int(ga + (gb - ga) * factor),
                int(ba + (bb - ba) * factor)
            )
            return '#{:02x}{:02x}{:02x}'.format(*blended)

        def create_rounded_rect(canvas: tk.Canvas, x1: int, y1: int, x2: int, y2: int,
                                radius: int = 20, **kwargs) -> int:
            if radius <= 0:
                return canvas.create_rectangle(x1, y1, x2, y2, **kwargs)
            points = [
                x1 + radius, y1,
                x2 - radius, y1,
                x2, y1,
                x2, y1 + radius,
                x2, y2 - radius,
                x2, y2,
                x2 - radius, y2,
                x1 + radius, y2,
                x1, y2,
                x1, y2 - radius,
                x1, y1 + radius,
                x1, y1
            ]
            return canvas.create_polygon(points, smooth=True, **kwargs)

        # --- Layout containers ---
        container = ttk.Frame(self.window, style="DarkRoot.TFrame", padding=30)
        container.pack(fill='both', expand=True)

        header = ttk.Frame(container, style="Dark.TFrame")
        header.pack(fill='x')

        ttk.Label(header, text="Multi-Chat Trading Bot", style="Title.TLabel").pack(anchor='w')
        ttk.Label(
            header,
            text="Enter your Telegram credentials to power the copier bot.",
            style="Subtitle.TLabel"
        ).pack(anchor='w', pady=(6, 24))

        # --- Progress indicator ---
        progress_frame = tk.Frame(container, bg=background)
        progress_frame.pack(fill='x', pady=(0, 30))

        steps = [
            ("Account", "Create Telegram app"),
            ("API", "Add credentials"),
            ("Verify", "Confirm login"),
            ("Launch", "Start bot")
        ]
        active_index = 1  # Highlight the credential input stage

        progress_canvas = tk.Canvas(progress_frame, height=100, bg=background, highlightthickness=0)
        progress_canvas.pack(fill='x', expand=True)

        def draw_progress(_: Optional[object] = None) -> None:
            width = progress_canvas.winfo_width()
            progress_canvas.delete('all')
            if width <= 0:
                return

            margin = 40
            radius = 20
            y = 32
            spacing = 0
            if len(steps) > 1:
                spacing = (width - 2 * margin) / (len(steps) - 1)

            for idx, (title, caption) in enumerate(steps):
                x = margin + spacing * idx
                if idx < len(steps) - 1:
                    next_x = margin + spacing * (idx + 1)
                    line_color = '#1F2A3A'
                    if idx < active_index:
                        line_color = neon_green
                    elif idx == active_index:
                        line_color = neon_blue
                    progress_canvas.create_line(
                        x + radius, y,
                        next_x - radius, y,
                        fill=line_color,
                        width=4,
                        capstyle=tk.ROUND
                    )

                fill_color = '#1F2A3A'
                outline_color = '#1F2A3A'
                text_color = subdued
                caption_color = '#475569'
                if idx < active_index:
                    fill_color = neon_green
                    outline_color = neon_green
                    text_color = background
                    caption_color = '#A7F3D0'
                elif idx == active_index:
                    fill_color = neon_blue
                    outline_color = neon_blue
                    text_color = background
                    caption_color = '#BAE6FD'

                progress_canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=fill_color,
                    outline=outline_color,
                    width=2
                )
                progress_canvas.create_text(
                    x,
                    y,
                    text=str(idx + 1),
                    fill=text_color,
                    font=("Segoe UI Semibold", 12)
                )
                progress_canvas.create_text(
                    x,
                    y + radius + 14,
                    text=title,
                    fill='#E2E8F0' if idx <= active_index else subdued,
                    font=("Segoe UI Semibold", 11)
                )
                progress_canvas.create_text(
                    x,
                    y + radius + 34,
                    text=caption,
                    fill=caption_color,
                    font=("Segoe UI", 9)
                )

        progress_canvas.bind('<Configure>', draw_progress)
        draw_progress()

        # --- Credential card ---
        card_wrapper = tk.Frame(container, bg=background)
        card_wrapper.pack(fill='x')

        card_canvas = tk.Canvas(card_wrapper, bg=background, highlightthickness=0)
        card_canvas.pack(fill='x', expand=True)

        card_frame = tk.Frame(card_canvas, bg=surface)
        card_window = card_canvas.create_window(0, 0, anchor='n', window=card_frame)

        def layout_card(_: Optional[object] = None) -> None:
            card_canvas.update_idletasks()
            width = card_canvas.winfo_width()
            if width <= 200:
                return
            desired_height = card_frame.winfo_reqheight() + 40
            if desired_height < 200:
                desired_height = 200
            card_canvas.config(height=desired_height)
            card_canvas.delete('card-bg')
            create_rounded_rect(
                card_canvas,
                10,
                10,
                width - 10,
                desired_height - 10,
                radius=24,
                fill=surface,
                outline="#1E293B",
                width=2,
                tags=('card-bg',)
            )
            inner_width = width - 80
            card_canvas.coords(card_window, width / 2, 30)
            card_canvas.itemconfigure(card_window, anchor='n')
            card_canvas.itemconfigure(card_window, width=inner_width)

        card_canvas.bind('<Configure>', layout_card)
        layout_card()

        ttk.Label(card_frame, text="Connect to Telegram", style="CardTitle.TLabel").pack(anchor='w')
        ttk.Label(
            card_frame,
            text="Grab your API keys from my.telegram.org and drop them in below.",
            style="MutedCard.TLabel",
            wraplength=580,
            justify='left'
        ).pack(anchor='w', pady=(6, 18))

        telegram_cfg = self.current_config.get('telegram', {})

        form_inner = ttk.Frame(card_frame, style="Card.TFrame")
        form_inner.pack(fill='x')

        ttk.Label(form_inner, text="API ID", style="Card.TLabel").pack(anchor='w')
        self.setup_api_id = tk.StringVar(value=str(telegram_cfg.get('api_id', "")))
        ttk.Entry(form_inner, textvariable=self.setup_api_id, style=entry_style).pack(fill='x', pady=(4, 14))

        ttk.Label(form_inner, text="API Hash", style="Card.TLabel").pack(anchor='w')
        self.setup_api_hash = tk.StringVar(value=str(telegram_cfg.get('api_hash', "")))
        ttk.Entry(form_inner, textvariable=self.setup_api_hash, style=entry_style).pack(fill='x', pady=(4, 14))

        ttk.Label(
            form_inner,
            text="Telefonnummer (inkl. Ländercode, z. B. +49…)",
            style="Card.TLabel"
        ).pack(anchor='w')
        self.setup_phone = tk.StringVar(value=str(telegram_cfg.get('phone', "")))
        ttk.Entry(form_inner, textvariable=self.setup_phone, style=entry_style).pack(fill='x', pady=(4, 10))

        self.prompt_credentials = tk.BooleanVar(
            value=bool(telegram_cfg.get('prompt_credentials_on_start', False))
        )
        ttk.Checkbutton(
            form_inner,
            text="API-Zugangsdaten bei jedem Start erneut abfragen",
            style="Dark.TCheckbutton",
            variable=self.prompt_credentials
        ).pack(anchor='w', pady=(6, 0))

        # --- Remaining steps list ---
        steps_frame = ttk.Frame(container, style="Dark.TFrame")
        steps_frame.pack(fill='x', pady=(30, 10))

        ttk.Label(steps_frame, text="Next up", style="Accent.TLabel").pack(anchor='w')
        remaining_steps = [
            "Schritt 3 – Telegram sendet dir einen Code. Gib ihn später im Login-Dialog ein.",
            "Schritt 4 – Wähle deine Signal-Channels im Hauptfenster aus.",
            "Schritt 5 – Aktiviere den Kopier-Modus, wenn alles bereit ist."
        ]
        for step_text in remaining_steps:
            ttk.Label(
                steps_frame,
                text=f"• {step_text}",
                style="Dark.TLabel",
                wraplength=640,
                justify='left'
            ).pack(anchor='w', pady=2)

        # --- Action buttons ---
        button_row = tk.Frame(container, bg=background)
        button_row.pack(fill='x', pady=(30, 0))

        ttk.Button(button_row, text="Abbrechen", command=self.cancel_setup, style="Dark.TButton").pack(
            side='right'
        )

        start_canvas = tk.Canvas(button_row, width=220, height=56, bg=background, highlightthickness=0,
                                 cursor='hand2')
        start_canvas.pack(side='right', padx=(0, 20))
        start_canvas.configure(takefocus=1)

        def draw_start_button(hover: bool = False) -> None:
            start_canvas.delete('all')
            width = int(start_canvas.winfo_width())
            height = int(start_canvas.winfo_height())
            if width <= 0 or height <= 0:
                return
            color_left = neon_blue
            color_right = neon_green
            if hover:
                color_left = blend(neon_blue, '#FFFFFF', 0.25)
                color_right = blend(neon_green, '#FFFFFF', 0.25)

            for x in range(width):
                ratio = 0 if width == 1 else x / (width - 1)
                color = blend(color_left, color_right, ratio)
                start_canvas.create_line(
                    x,
                    height / 2,
                    x,
                    height / 2,
                    fill=color,
                    width=height - 6,
                    capstyle=tk.ROUND
                )

            start_canvas.create_text(
                width / 2,
                height / 2,
                text="START BOT",
                font=("Segoe UI Semibold", 13),
                fill="#041225"
            )

        start_canvas.bind('<Configure>', lambda _event: draw_start_button())
        draw_start_button()

        start_canvas.bind('<Enter>', lambda _event: draw_start_button(True))
        start_canvas.bind('<Leave>', lambda _event: draw_start_button(False))
        start_canvas.bind('<Button-1>', lambda _event: self.start_bot())
        start_canvas.bind('<Return>', lambda _event: self.start_bot())
        start_canvas.bind('<space>', lambda _event: self.start_bot())
        start_canvas.focus_set()
        self.window.bind('<Return>', lambda _event: self.start_bot())

    def start_bot(self):
        """Konfiguration testen/speichern"""
        api_id = self.setup_api_id.get().strip()
        api_hash = self.setup_api_hash.get().strip()
        phone = self.setup_phone.get().strip()

        if not all([api_id, api_hash, phone]):
            messagebox.showerror("Fehler", "Bitte füllen Sie alle Felder aus.")
            return

        try:
            config = self.config_manager.load_config()
            telegram_cfg = config.setdefault('telegram', {})
            telegram_cfg.update({
                'api_id': api_id,
                'api_hash': api_hash,
                'phone': phone,
                'session_name': telegram_cfg.get('session_name', 'trading_session'),
                'prompt_credentials_on_start': self.prompt_credentials.get()
            })

            if 'trading' not in config:
                config['trading'] = self.config_manager.default_config.get('trading', {})
            if 'signals' not in config:
                config['signals'] = self.config_manager.default_config.get('signals', {})

            self.config_manager.save_config(config)
            messagebox.showinfo("Erfolg", "Konfiguration gespeichert. Bitte starten Sie den Bot.")
            self.config_saved = True
            self.window.destroy()
            if self.parent:
                self.parent.quit()

        except Exception as e:
            messagebox.showerror("Fehler", f"Konfigurationsfehler: {e}")

    def cancel_setup(self):
        """Setup abbrechen"""
        if messagebox.askyesno("Bestätigung", "Setup wirklich abbrechen?"):
            self.window.destroy()
            if self.parent:
                self.parent.quit()


def check_first_run() -> bool:
    """Prüfen ob es der erste Start ist"""
    config_manager = ConfigManager()
    config = config_manager.load_config()
    telegram_cfg = config.get('telegram', {})

    required_fields = ('api_id', 'api_hash', 'phone')
    return not all(telegram_cfg.get(field) for field in required_fields)


def show_startup_warning() -> bool:
    """Startup-Warnung anzeigen"""
    root = tk.Tk()
    root.withdraw()

    warning_text = (
        "WARNUNG: AUTOMATISIERTES TRADING SYSTEM\n\n"
        "Dieses System kann automatisch Trades ausführen!\n\n"
        "• Hohe finanzielle Verlustrisiken\n"
        "• Nur für erfahrene Trader geeignet\n"
        "• Umfangreiche Tests erforderlich\n"
        "• Demo-Modus wird dringend empfohlen\n\n"
        "Möchten Sie fortfahren?"
    )
    result = messagebox.askyesno("Sicherheitswarnung", warning_text, icon='warning')
    root.destroy()
    return result


def prompt_for_api_credentials(config_manager: ConfigManager, config: Optional[Dict] = None) -> bool:
    """Fragt die Telegram-API-Zugangsdaten ab und speichert sie."""

    config = config or config_manager.load_config()
    telegram_cfg = config.get('telegram', {})

    root = tk.Tk()
    root.withdraw()

    try:
        dialog = ApiCredentialDialog(root, telegram_cfg)
        credentials = getattr(dialog, 'result', None)
    finally:
        root.destroy()

    if not credentials:
        return False

    telegram_cfg.update(credentials)
    config['telegram'] = telegram_cfg
    config_manager.save_config(config)
    return True


# ==================== MAIN ====================


def safe_main():
    try:
        run_onboarding_if_needed()
        _start_ui()
    except Exception as e:
        pathlib.Path("logs").mkdir(exist_ok=True)
        with open("logs/last_startup_error.log", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        print("\n[FATAL] Startfehler:", e)
        print("Details: logs\\last_startup_error.log")
        input("Taste drücken…")


if __name__ == "__main__":
    safe_main()
