# -*- coding: utf-8 -*-
# TelegramCopier_Windows.py
# Windows GUI-App (Tkinter) mit Telethon; MT5 ist optional (nur für Live-Mode)
# Start:  python TelegramCopier_Windows.py

import asyncio
import os
import re
import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List

# ---- optionale Abhängigkeit: MetaTrader5 (nur für Windows verfügbar) ----
try:
    import MetaTrader5 as mt5  # noqa: F401
    MT5_AVAILABLE = True
except Exception:
    MT5_AVAILABLE = False

from telethon import TelegramClient, events
import tkinter as tk
from tkinter import ttk, messagebox

# ==================== THEME CONSTANTS ====================

DARK_BG = "#040915"
PANEL_BG = "#081125"
CARD_BG = "#0F1B33"
CARD_ALT_BG = "#16213F"
ACCENT_PRIMARY = "#16F6A6"
ACCENT_SECONDARY = "#22D3EE"
WARNING_COLOR = "#F97316"
DANGER_COLOR = "#FB7185"
TEXT_PRIMARY = "#F8FAFC"
TEXT_MUTED = "#94A3B8"

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
        self.symbol_mapping = {
            'GOLD': 'XAUUSD',
            'XAU': 'XAUUSD',
            'EURUSD': 'EURUSD',
            'EUR': 'EURUSD',
            'GBPUSD': 'GBPUSD',
            'GBP': 'GBPUSD',
            'USDJPY': 'USDJPY',
            'USD': 'USDJPY'
        }

        self.patterns = {
            'buy_now': r'(?i)\b(gold|xau|eurusd|eur|gbpusd|gbp|usdjpy|usd)\b.*\bbuy\b.*\bnow\b',
            'sell_now': r'(?i)\b(gold|xau|eurusd|eur|gbpusd|gbp|usdjpy|usd)\b.*\bsell\b.*\bnow\b',
            'buy_zone': r'(?i)\b(gold|xau|eurusd|eur|gbpusd|gbp|usdjpy|usd)\b.*\bbuy\b\s+([0-9]+\.?[0-9]*)',
            'sell_zone': r'(?i)\b(gold|xau|eurusd|eur|gbpusd|gbp|usdjpy|usd)\b.*\bsell\b\s+([0-9]+\.?[0-9]*)'
        }

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

        stop_loss = self.extract_stop_loss(message_text)
        take_profits = self.extract_take_profits(message_text)

        # Buy Now
        match = re.search(self.patterns['buy_now'], message_text)
        if match:
            base = match.group(1).upper()
            symbol = self.symbol_mapping.get(base, base)
            return {
                'kind': 'trade',
                'type': 'instant',
                'action': 'BUY',
                'symbol': symbol,
                'source': chat_source.chat_name,
                'execution_mode': ExecutionMode.INSTANT,
                'stop_loss': stop_loss,
                'take_profits': take_profits
            }

        # Sell Now
        match = re.search(self.patterns['sell_now'], message_text)
        if match:
            base = match.group(1).upper()
            symbol = self.symbol_mapping.get(base, base)
            return {
                'kind': 'trade',
                'type': 'instant',
                'action': 'SELL',
                'symbol': symbol,
                'source': chat_source.chat_name,
                'execution_mode': ExecutionMode.INSTANT,
                'stop_loss': stop_loss,
                'take_profits': take_profits
            }

        if stop_loss is not None or take_profits:
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

    def __init__(self, api_id: str, api_hash: str, phone: str):
        # Credentials
        self.api_id = 0
        self.api_hash = ""
        self.phone = ""

        # Components
        self.chat_manager = MultiChatManager()
        self.trade_tracker = TradeTracker()
        self.signal_processor = SignalProcessor()

        # Telegram Client (erst erstellen, wenn gültige Daten vorhanden sind)
        self.client: Optional[TelegramClient] = None
        self.update_credentials(api_id, api_hash, phone)

        # Message Queue für GUI
        self.message_queue: "queue.Queue" = queue.Queue()

        # Status
        self.is_running = False
        self.demo_mode = True  # Immer mit Demo starten!
        self.pending_trade_updates: Dict[int, Dict] = {}

    def update_credentials(self, api_id: str, api_hash: str, phone: str):
        """Telegram-Zugangsdaten aktualisieren und Client vorbereiten"""

        # Vorherigen Client sauber schließen
        if self.client:
            try:
                self.client.disconnect()
            except Exception:
                pass

        self.client = None
        self.api_id = int(api_id) if str(api_id).isdigit() else 0
        self.api_hash = api_hash or ""
        self.phone = phone or ""

        if self.api_id and self.api_hash:
            self.client = TelegramClient('trading_session', self.api_id, self.api_hash)

    def has_active_client(self) -> bool:
        return self.client is not None

    async def start(self):
        """Bot starten"""
        if not self.client:
            self.log("Telegram-Zugangsdaten fehlen. Bitte zuerst verbinden.", "ERROR")
            return

        try:
            await self.client.connect()

            if not await self.client.is_user_authorized():
                await self.client.send_code_request(self.phone)
                # Code-Eingabe sollte über GUI / Telethon auth erfolgen

            @self.client.on(events.NewMessage)
            async def message_handler(event):
                await self.handle_new_message(event)

            self.is_running = True
            self.log("Bot gestartet - Multi-Chat-Modus aktiv")

            # Client im Hintergrund laufen lassen
            asyncio.create_task(self.client.run_until_disconnected())

        except Exception as e:
            self.log(f"Fehler beim Starten: {e}", "ERROR")

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
                    trade_result = await self.execute_signal(signal, chat_source, message_text)
                    if trade_result:
                        pending_info = {
                            'ticket': trade_result['ticket'],
                            'symbol': trade_result['symbol'],
                            'awaiting_sl': signal.get('stop_loss') is None,
                            'awaiting_tp': not signal.get('take_profits'),
                            'timestamp': datetime.now()
                        }
                        if pending_info['awaiting_sl'] or pending_info['awaiting_tp']:
                            self.pending_trade_updates[chat_id] = pending_info
                        else:
                            self.pending_trade_updates.pop(chat_id, None)
                elif kind == 'update':
                    await self.apply_trade_update(chat_source, signal)

        except Exception as e:
            self.log(f"Fehler bei Nachrichtenverarbeitung: {e}", "ERROR")

    async def execute_signal(self, signal: Dict, chat_source: ChatSource, original_message: str):
        """Signal ausführen (Demo oder Live)"""
        try:
            if self.demo_mode or not MT5_AVAILABLE:
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

                # Demo-Trade simulieren
                demo_result = {
                    'ticket': f"DEMO_{int(datetime.now().timestamp())}",
                    'symbol': signal['symbol'],
                    'direction': signal['action'],
                    'price': 1.0850 if 'EUR' in signal['symbol'] else 2660.00,
                    'lot_size': 0.01,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profits': normalized_tps,
                    'status': 'executed',
                    'profit_loss': 0.0
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
            else:
                # TODO: Echten MT5-Trade umsetzen
                self.log("LIVE-Trading ist aktiv, aber noch nicht implementiert.", "WARNING")

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

        self.log(
            f"Trade {record.ticket} aktualisiert: SL={record.stop_loss:.2f} TP={record.take_profit:.2f}"
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
        if not self.client:
            self.log("Keine Telegram-Verbindung. Bitte Zugangsdaten hinterlegen und verbinden.", "ERROR")
            return []

        chats_data = []
        try:
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
                "session_name": "trading_session"
            },
            "trading": {
                "demo_mode": True,
                "default_lot_size": 0.01,
                "max_spread_pips": 3.0,
                "risk_percent": 2.0,
                "max_trades_per_hour": 5
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


class TradingGUI:
    """Moderner Kontrollbereich für den Telegram Trade-Kopierer."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Telegram Copier Control Center")
        self.root.geometry("1280x840")
        self.root.configure(bg=DARK_BG)

        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()

        api_id = str(self.config['telegram'].get('api_id') or "")
        api_hash = self.config['telegram'].get('api_hash', "")
        phone = self.config['telegram'].get('phone', "")

        self.bot = MultiChatTradingBot(api_id or "0", api_hash, phone)
        self.bot.demo_mode = bool(self.config['trading'].get('demo_mode', True))

        self.metric_vars: Dict[str, tk.StringVar] = {}
        self.nav_buttons: Dict[str, ttk.Button] = {}
        self.pages: Dict[str, ttk.Frame] = {}
        self.app_initialized = False

        self.configure_style()
        self.build_start_screen()

        if api_id and api_hash and phone:
            self.show_main_interface(initial=True)
        else:
            self.show_start_screen()

    def configure_style(self):
        style = ttk.Style()
        try:
            style.theme_create(
                "copier_dark",
                parent="clam",
                settings={
                    "TFrame": {"configure": {"background": DARK_BG}},
                    "TLabel": {"configure": {"background": DARK_BG, "foreground": TEXT_PRIMARY}},
                    "TButton": {
                        "configure": {
                            "background": CARD_BG,
                            "foreground": TEXT_PRIMARY,
                            "padding": 10,
                            "relief": "flat",
                            "font": ("Segoe UI", 10)
                        }
                    },
                    "TCheckbutton": {
                        "configure": {
                            "background": DARK_BG,
                            "foreground": TEXT_PRIMARY,
                            "font": ("Segoe UI", 10)
                        }
                    },
                },
            )
        except tk.TclError:
            pass

        try:
            style.theme_use("copier_dark")
        except tk.TclError:
            pass

        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("CardAlt.TFrame", background=CARD_ALT_BG)
        style.configure("CardTitle.TLabel", font=("Segoe UI", 11, "bold"), background=CARD_BG, foreground=TEXT_MUTED)
        style.configure("CardValue.TLabel", font=("Segoe UI", 24, "bold"), background=CARD_BG, foreground=TEXT_PRIMARY)
        style.configure("Accent.TButton", background=ACCENT_PRIMARY, foreground="#00151F", padding=12, font=("Segoe UI", 11, "bold"))
        style.map("Accent.TButton", background=[("active", ACCENT_SECONDARY)], foreground=[("disabled", TEXT_MUTED)])
        style.configure("Nav.TButton", background=PANEL_BG, foreground=TEXT_PRIMARY, padding=12, font=("Segoe UI", 11))
        style.map("Nav.TButton", background=[("active", CARD_ALT_BG)], foreground=[("active", TEXT_PRIMARY)])
        style.configure("NavActive.TButton", background=ACCENT_PRIMARY, foreground="#00151F", padding=12, font=("Segoe UI", 11, "bold"))
        style.configure("Treeview", background=PANEL_BG, foreground=TEXT_PRIMARY, fieldbackground=PANEL_BG, bordercolor=PANEL_BG, rowheight=28)
        style.map("Treeview", background=[("selected", ACCENT_SECONDARY)], foreground=[("selected", DARK_BG)])
        style.configure("Treeview.Heading", background=CARD_BG, foreground=TEXT_PRIMARY, relief="flat", font=("Segoe UI", 11, "bold"))
        style.configure("Dark.TEntry", fieldbackground=CARD_ALT_BG, foreground=TEXT_PRIMARY, background=CARD_ALT_BG, bordercolor=CARD_BG)
        style.map("Dark.TEntry", fieldbackground=[("focus", CARD_BG), ("active", CARD_BG)])
        style.configure("StatusOk.TLabel", background="#123524", foreground=ACCENT_PRIMARY, padding=(12, 6), font=("Segoe UI", 10, "bold"))
        style.configure("StatusWarn.TLabel", background="#3A1F18", foreground=WARNING_COLOR, padding=(12, 6), font=("Segoe UI", 10, "bold"))

    def build_start_screen(self):
        self.start_frame = ttk.Frame(self.root, padding=60)
        self.start_frame.pack(fill='both', expand=True)

        card = ttk.Frame(self.start_frame, style="Card.TFrame", padding=40)
        card.pack(expand=True)

        ttk.Label(card, text="Willkommen beim Telegram Copier", style="CardTitle.TLabel", font=("Segoe UI", 22, "bold"), foreground=TEXT_PRIMARY).pack(anchor='w')
        ttk.Label(
            card,
            text=(
                "Verbinde dich mit deiner Telegram App, indem du deine Zugangsdaten einträgst. "
                "Du findest die Daten unter https://my.telegram.org/"
            ),
            style="CardTitle.TLabel",
            wraplength=520
        ).pack(anchor='w', pady=(10, 20))

        form = ttk.Frame(card, style="Card.TFrame")
        form.pack(fill='x')

        self.api_id_var = tk.StringVar(value=self.config['telegram'].get('api_id', ""))
        self.api_hash_var = tk.StringVar(value=self.config['telegram'].get('api_hash', ""))
        self.phone_var = tk.StringVar(value=self.config['telegram'].get('phone', ""))

        ttk.Label(form, text="Telegram App ID", style="CardTitle.TLabel", foreground=TEXT_PRIMARY).pack(anchor='w')
        ttk.Entry(form, textvariable=self.api_id_var, style="Dark.TEntry", width=40).pack(fill='x', pady=(0, 12))

        ttk.Label(form, text="API Hash", style="CardTitle.TLabel", foreground=TEXT_PRIMARY).pack(anchor='w')
        ttk.Entry(form, textvariable=self.api_hash_var, style="Dark.TEntry", width=40).pack(fill='x', pady=(0, 12))

        ttk.Label(form, text="Telefonnummer (mit + Landesvorwahl)", style="CardTitle.TLabel", foreground=TEXT_PRIMARY).pack(anchor='w')
        ttk.Entry(form, textvariable=self.phone_var, style="Dark.TEntry", width=40).pack(fill='x', pady=(0, 12))

        ttk.Label(
            card,
            text="Diese Angaben werden lokal in trading_config.json gespeichert.",
            style="CardTitle.TLabel"
        ).pack(anchor='w', pady=(10, 20))

        action_row = ttk.Frame(card, style="Card.TFrame")
        action_row.pack(fill='x')
        ttk.Button(action_row, text="Speichern & Oberfläche öffnen", style="Accent.TButton", command=self.handle_start_continue).pack(side='left')
        ttk.Button(action_row, text="Abbrechen", command=self.root.quit).pack(side='left', padx=(10, 0))

    def show_start_screen(self):
        if hasattr(self, 'app_shell') and self.app_shell.winfo_ismapped():
            self.app_shell.pack_forget()
        self.start_frame.pack(fill='both', expand=True)

    def handle_start_continue(self):
        api_id = self.api_id_var.get().strip()
        api_hash = self.api_hash_var.get().strip()
        phone = self.phone_var.get().strip()

        if not api_id:
            messagebox.showerror("Fehlende App ID", "Bitte trage deine Telegram App ID ein.")
            return

        try:
            int(api_id)
        except ValueError:
            messagebox.showerror("Ungültige App ID", "Die App ID darf nur Ziffern enthalten.")
            return

        if not api_hash or not phone:
            messagebox.showerror("Unvollständige Angaben", "Bitte ergänze auch API Hash und Telefonnummer.")
            return

        self.config['telegram']['api_id'] = api_id
        self.config['telegram']['api_hash'] = api_hash
        self.config['telegram']['phone'] = phone
        self.config['trading']['demo_mode'] = bool(self.bot.demo_mode)
        self.config_manager.save_config(self.config)

        self.apply_credentials(api_id, api_hash, phone)
        self.show_main_interface(initial=not self.app_initialized)
        messagebox.showinfo("Gespeichert", "Zugangsdaten gespeichert. Der Kopierer kann nun gestartet werden.")

    def apply_credentials(self, api_id: str, api_hash: str, phone: str):
        self.bot.update_credentials(api_id, api_hash, phone)

    def show_main_interface(self, initial: bool = False):
        if hasattr(self, 'start_frame'):
            self.start_frame.pack_forget()

        if not self.app_initialized:
            self.app_shell = ttk.Frame(self.root, padding=20)
            self.app_shell.pack(fill='both', expand=True)

            self.build_header(self.app_shell)

            body = ttk.Frame(self.app_shell)
            body.pack(fill='both', expand=True, pady=(15, 0))

            self.nav_frame = ttk.Frame(body, width=230, padding=(0, 10))
            self.nav_frame.pack(side='left', fill='y', padx=(0, 18))
            self.nav_frame.pack_propagate(False)

            self.page_container = ttk.Frame(body)
            self.page_container.pack(side='left', fill='both', expand=True)

            nav_items = [
                ("dashboard", "Dashboard"),
                ("chats", "Chats"),
                ("trades", "Trades"),
                ("automation", "Automatisierung"),
                ("performance", "Performance"),
            ]

            self.nav_buttons.clear()
            for key, title in nav_items:
                btn = ttk.Button(
                    self.nav_frame,
                    text=title,
                    style="Nav.TButton",
                    command=lambda k=key: self.show_page(k)
                )
                btn.pack(fill='x', pady=4)
                self.nav_buttons[key] = btn

            self.pages['dashboard'] = self.build_dashboard_page()
            self.pages['chats'] = self.build_chats_page()
            self.pages['trades'] = self.build_trades_page()
            self.pages['automation'] = self.build_automation_page()
            self.pages['performance'] = self.build_performance_page()

            status_bar = ttk.Frame(self.app_shell, padding=(10, 8))
            status_bar.pack(fill='x', pady=(18, 0))
            self.status_var = tk.StringVar(value="Bereit")
            self.status_label = ttk.Label(status_bar, textvariable=self.status_var)
            self.status_label.pack(side='left')

            self.app_initialized = True
            self.setup_message_processing()

        else:
            self.app_shell.pack(fill='both', expand=True)

        self.update_overview_metrics()
        self.show_page('dashboard')

        if initial:
            self.update_status_bar("Konfiguration geladen. Bot bereit.")

    def build_header(self, parent):
        header = ttk.Frame(parent, padding=(10, 0, 10, 20))
        header.pack(fill='x')

        title = ttk.Label(header, text="Signal & Trade Kopierer", font=("Segoe UI", 24, "bold"))
        title.pack(side='left')

        subtitle = ttk.Label(
            header,
            text="Verwalte Telegram Signale, starte Automationen und überwache deine Performance.",
            font=("Segoe UI", 11),
            foreground=TEXT_MUTED
        )
        subtitle.pack(side='left', padx=(15, 0))

        self.connection_status_var = tk.StringVar(value="Getrennt")
        self.connection_badge = ttk.Label(header, textvariable=self.connection_status_var, style="StatusWarn.TLabel")
        self.connection_badge.pack(side='right')

    def build_dashboard_page(self):
        frame = ttk.Frame(self.page_container)

        cards_frame = ttk.Frame(frame)
        cards_frame.pack(fill='x')

        metrics = [
            ("signals", "Signale gesamt"),
            ("active_chats", "Aktive Quellen"),
            ("mode", "Modus"),
            ("last_signal", "Letztes Signal"),
        ]

        for key, title in metrics:
            card = ttk.Frame(cards_frame, style="Card.TFrame", padding=20)
            card.pack(side='left', expand=True, fill='x', padx=(0, 15))

            title_label = ttk.Label(card, text=title, style="CardTitle.TLabel")
            title_label.pack(anchor='w')

            var = tk.StringVar(value="-")
            self.metric_vars[key] = var

            value_label = ttk.Label(card, textvariable=var, style="CardValue.TLabel")
            value_label.pack(anchor='w', pady=(6, 0))

        control_frame = ttk.Frame(frame, padding=(0, 20, 0, 10))
        control_frame.pack(fill='x')
        ttk.Button(control_frame, text="Bot starten", style="Accent.TButton", command=self.start_bot).pack(side='left')
        ttk.Button(control_frame, text="Bot stoppen", command=self.stop_bot).pack(side='left', padx=(10, 0))

        log_container = ttk.Frame(frame)
        log_container.pack(fill='both', expand=True)

        log_title = ttk.Label(
            log_container,
            text="Aktivitätsprotokoll",
            font=("Segoe UI", 12, "bold"),
            foreground=TEXT_MUTED
        )
        log_title.pack(anchor='w', pady=(0, 6))
        log_frame = ttk.Frame(log_container, style="CardAlt.TFrame")
        log_frame.pack(fill='both', expand=True)

        self.log_text = tk.Text(
            log_frame,
            height=18,
            wrap='word',
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief='flat',
            borderwidth=0,
            font=("Consolas", 10)
        )
        self.log_text.pack(side='left', fill='both', expand=True, padx=8, pady=8)

        log_scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        log_scroll.pack(side='right', fill='y')
        self.log_text.configure(yscrollcommand=log_scroll.set)

        return frame

    def build_chats_page(self):
        frame = ttk.Frame(self.page_container)

        intro = ttk.Label(
            frame,
            text="Lade Telegram-Chats, aktiviere Quellen und ordne sie Prioritäten zu.",
            wraplength=700,
            foreground=TEXT_MUTED
        )
        intro.pack(anchor='w', pady=(0, 10))

        controls = ttk.Frame(frame)
        controls.pack(fill='x', pady=(0, 10))

        ttk.Button(controls, text="Chats laden", style="Accent.TButton", command=self.load_chats).pack(side='left')
        ttk.Button(controls, text="Überwachung aktivieren", command=self.enable_monitoring).pack(side='left', padx=(10, 0))
        ttk.Button(controls, text="Überwachung deaktivieren", command=self.disable_monitoring).pack(side='left', padx=(10, 0))

        columns = ('Name', 'ID', 'Typ', 'Teilnehmer', 'Überwacht', 'Signale')
        self.chats_tree = ttk.Treeview(frame, columns=columns, show='headings', height=20)
        for col in columns:
            self.chats_tree.heading(col, text=col)
            self.chats_tree.column(col, anchor='center')
        self.chats_tree.column('Name', anchor='w', width=220)
        self.chats_tree.column('ID', width=140)
        self.chats_tree.column('Typ', width=100)
        self.chats_tree.column('Teilnehmer', width=120)

        self.chats_tree.pack(fill='both', expand=True)

        chat_scroll = ttk.Scrollbar(frame, orient='vertical', command=self.chats_tree.yview)
        chat_scroll.pack(side='right', fill='y')
        self.chats_tree.configure(yscrollcommand=chat_scroll.set)

        return frame

    def build_trades_page(self):
        frame = ttk.Frame(self.page_container)

        ttk.Label(frame, text="Ausgeführte Trades", font=("Segoe UI", 12, "bold"), foreground=TEXT_MUTED).pack(anchor='w')
        trades_frame = ttk.Frame(frame, style="CardAlt.TFrame")
        trades_frame.pack(fill='both', expand=True, pady=(10, 0))

        columns = ('Ticket', 'Symbol', 'Richtung', 'Quelle', 'Preis', 'SL', 'TP', 'Zeit')
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show='headings', height=20)
        for col in columns:
            self.trades_tree.heading(col, text=col)
            self.trades_tree.column(col, anchor='center')
        self.trades_tree.column('Ticket', width=160)
        self.trades_tree.column('Symbol', width=110)
        self.trades_tree.column('Quelle', anchor='w', width=200)

        self.trades_tree.pack(side='left', fill='both', expand=True, padx=8, pady=8)

        trade_scroll = ttk.Scrollbar(trades_frame, orient='vertical', command=self.trades_tree.yview)
        trade_scroll.pack(side='right', fill='y')
        self.trades_tree.configure(yscrollcommand=trade_scroll.set)

        ttk.Button(frame, text="Trades aktualisieren", command=self.refresh_trades_view).pack(anchor='w', pady=(12, 0))

        return frame

    def build_automation_page(self):
        frame = ttk.Frame(self.page_container)

        ttk.Label(
            frame,
            text="Automationsregeln steuern den Umgang mit Signalen und das Risikomanagement.",
            wraplength=720,
            foreground=TEXT_MUTED
        ).pack(anchor='w', pady=(0, 12))

        self.demo_var = tk.BooleanVar(value=self.bot.demo_mode)
        ttk.Checkbutton(frame, text="Demo-Modus aktivieren", variable=self.demo_var, command=self.toggle_demo_mode).pack(anchor='w', pady=4)

        trading_cfg = self.config.get('trading', {})
        signals_cfg = self.config.get('signals', {})

        self.auto_tp_var = tk.BooleanVar(value=signals_cfg.get('auto_tp_sl', True))
        ttk.Checkbutton(frame, text="Take Profit / Stop Loss automatisch übernehmen", variable=self.auto_tp_var).pack(anchor='w', pady=4)

        self.require_confirm_var = tk.BooleanVar(value=signals_cfg.get('require_confirmation', True))
        ttk.Checkbutton(frame, text="Manuelle Bestätigung vor Live-Ausführung", variable=self.require_confirm_var).pack(anchor='w', pady=4)

        ttk.Label(
            frame,
            text="Lot-Größe (Demo)",
            foreground=TEXT_MUTED
        ).pack(anchor='w', pady=(20, 4))
        self.lot_size_var = tk.StringVar(value=str(trading_cfg.get('default_lot_size', 0.01)))
        ttk.Entry(frame, textvariable=self.lot_size_var, style="Dark.TEntry", width=10).pack(anchor='w')

        ttk.Label(
            frame,
            text="Maximale Trades pro Stunde",
            foreground=TEXT_MUTED
        ).pack(anchor='w', pady=(16, 4))
        self.max_trades_var = tk.StringVar(value=str(trading_cfg.get('max_trades_per_hour', 5)))
        ttk.Entry(frame, textvariable=self.max_trades_var, style="Dark.TEntry", width=10).pack(anchor='w')

        ttk.Button(frame, text="Einstellungen speichern", style="Accent.TButton", command=self.save_signal_settings).pack(anchor='w', pady=(24, 0))

        return frame

    def build_performance_page(self):
        frame = ttk.Frame(self.page_container)

        ttk.Label(frame, text="Performance nach Quelle", font=("Segoe UI", 12, "bold"), foreground=TEXT_MUTED).pack(anchor='w')

        stats_frame = ttk.Frame(frame, style="CardAlt.TFrame")
        stats_frame.pack(fill='both', expand=True, pady=(10, 0))

        columns = ('Quelle', 'Trades', 'Gewinnrate', 'Profit', 'Letzter Trade')
        self.stats_tree = ttk.Treeview(stats_frame, columns=columns, show='headings', height=18)
        for col in columns:
            self.stats_tree.heading(col, text=col)
            self.stats_tree.column(col, anchor='center')
        self.stats_tree.column('Quelle', anchor='w', width=220)

        self.stats_tree.pack(side='left', fill='both', expand=True, padx=8, pady=8)

        stats_scroll = ttk.Scrollbar(stats_frame, orient='vertical', command=self.stats_tree.yview)
        stats_scroll.pack(side='right', fill='y')
        self.stats_tree.configure(yscrollcommand=stats_scroll.set)

        ttk.Button(frame, text="Statistiken aktualisieren", command=self.refresh_statistics).pack(anchor='w', pady=(12, 0))

        return frame

    def show_page(self, key: str):
        for frame in self.pages.values():
            frame.pack_forget()

        page = self.pages.get(key)
        if page:
            page.pack(fill='both', expand=True)
        self.update_nav_state(key)

        if key == 'performance':
            self.refresh_statistics()
        elif key == 'trades':
            self.refresh_trades_view()

    def update_nav_state(self, active_key: str):
        for key, button in self.nav_buttons.items():
            style = "NavActive.TButton" if key == active_key else "Nav.TButton"
            button.configure(style=style)

    def update_overview_metrics(self):
        total_signals = sum(source.signal_count for source in self.bot.chat_manager.chat_sources.values())
        active_chats = sum(1 for source in self.bot.chat_manager.chat_sources.values() if source.enabled)
        last_signal_dt = None
        for source in self.bot.chat_manager.chat_sources.values():
            if source.last_signal and (last_signal_dt is None or source.last_signal > last_signal_dt):
                last_signal_dt = source.last_signal

        last_signal_text = last_signal_dt.strftime("%d.%m %H:%M") if last_signal_dt else "Keine Signale"
        mode_text = "Demo" if self.bot.demo_mode or not MT5_AVAILABLE else "Live"

        if self.metric_vars:
            if 'signals' in self.metric_vars:
                self.metric_vars['signals'].set(str(total_signals))
            if 'active_chats' in self.metric_vars:
                self.metric_vars['active_chats'].set(str(active_chats))
            if 'mode' in self.metric_vars:
                self.metric_vars['mode'].set(mode_text)
            if 'last_signal' in self.metric_vars:
                self.metric_vars['last_signal'].set(last_signal_text)

    def append_log(self, message: str):
        if hasattr(self, 'log_text'):
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert('end', f"[{timestamp}] {message}\n")
            self.log_text.see('end')



    def setup_message_processing(self):
        def process_messages():
            try:
                while True:
                    msg_type, data = self.bot.message_queue.get(block=False)
                    if msg_type == 'LOG':
                        self.append_log(str(data))
                    elif msg_type == 'TRADE_EXECUTED':
                        self.handle_trade_event(data)
            except queue.Empty:
                pass
            self.root.after(150, process_messages)

        process_messages()

    def handle_trade_event(self, trade_data):
        self.append_log(f"Trade ausgeführt: {trade_data}")
        self.add_trade_row(trade_data)
        self.update_overview_metrics()

    def add_trade_row(self, trade_data):
        if not hasattr(self, 'trades_tree'):
            return
        values = (
            trade_data.get('ticket', ''),
            trade_data.get('symbol', ''),
            trade_data.get('direction', ''),
            trade_data.get('source', ''),
            f"{trade_data.get('price', 0.0):.2f}",
            f"{trade_data.get('stop_loss', 0.0):.2f}" if trade_data.get('stop_loss') else "-",
            f"{trade_data.get('take_profit', 0.0):.2f}" if trade_data.get('take_profit') else "-",
            datetime.now().strftime("%d.%m %H:%M")
        )
        self.trades_tree.insert('', 0, values=values)
        children = self.trades_tree.get_children()
        if len(children) > 250:
            for item in children[250:]:
                self.trades_tree.delete(item)

    def start_bot(self):
        if not self.bot.has_active_client():
            messagebox.showerror("Keine Zugangsdaten", "Bitte hinterlege eine gültige App ID und API Hash, bevor der Bot gestartet wird.")
            self.update_status_bar("Bot konnte nicht gestartet werden - fehlende Zugangsdaten")
            return

        self.update_status_bar("Bot wird gestartet...")

        def run_bot():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.bot.start())
                self.root.after(0, self.after_bot_started)
                loop.run_forever()
            except Exception as e:
                self.root.after(0, lambda: self.append_log(f"Bot-Start-Fehler: {e}"))
            finally:
                loop.close()

        threading.Thread(target=run_bot, daemon=True).start()

    def after_bot_started(self):
        self.update_status_bar("Bot läuft. Warte auf Signale...")
        if hasattr(self, 'connection_badge'):
            self.connection_status_var.set("Verbunden")
            self.connection_badge.configure(style="StatusOk.TLabel")

    def stop_bot(self):
        self.bot.is_running = False
        try:
            if self.bot.client:
                self.bot.client.disconnect()
        except Exception:
            pass
        self.update_status_bar("Bot gestoppt")
        if hasattr(self, 'connection_badge'):
            self.connection_status_var.set("Getrennt")
            self.connection_badge.configure(style="StatusWarn.TLabel")

    def load_chats(self):
        if not self.bot.has_active_client():
            messagebox.showerror("Keine Verbindung", "Ohne gültige Telegram-Verbindung können keine Chats geladen werden.")
            return

        self.update_status_bar("Chats werden geladen...")

        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                chats = loop.run_until_complete(self.bot.load_all_chats())
                self.root.after(0, lambda: self.update_chat_list(chats))
            except Exception as e:
                self.root.after(0, lambda: self.append_log(f"Fehler beim Laden: {e}"))
            finally:
                loop.close()

        threading.Thread(target=run_async, daemon=True).start()

    def update_chat_list(self, chats_data):
        if not hasattr(self, 'chats_tree'):
            return

        for item in self.chats_tree.get_children():
            self.chats_tree.delete(item)

        for chat in chats_data:
            chat_source = self.bot.chat_manager.get_chat_info(chat['id'])
            is_monitored = "Ja" if chat_source and chat_source.enabled else "Nein"
            signal_count = chat_source.signal_count if chat_source else 0

            self.chats_tree.insert('', 'end', values=(
                chat['name'],
                chat['id'],
                chat['type'],
                chat['participants'],
                is_monitored,
                signal_count
            ))

        self.update_status_bar(f"Chats geladen: {len(chats_data)}")
        self.update_overview_metrics()

    def enable_monitoring(self):
        selection = self.chats_tree.selection()
        if not selection:
            messagebox.showwarning("Keine Auswahl", "Bitte wähle mindestens einen Chat aus.")
            return

        for item in selection:
            values = self.chats_tree.item(item)['values']
            chat_id = int(values[1])
            chat_name = values[0]
            chat_type = values[2]
            self.bot.chat_manager.add_chat_source(chat_id, chat_name, chat_type, True)
            updated_values = list(values)
            updated_values[4] = "Ja"
            self.chats_tree.item(item, values=updated_values)

        self.update_overview_metrics()
        messagebox.showinfo("Aktiviert", f"{len(selection)} Chat(s) aktiviert")

    def disable_monitoring(self):
        selection = self.chats_tree.selection()
        if not selection:
            messagebox.showwarning("Keine Auswahl", "Bitte wähle mindestens einen Chat aus.")
            return

        for item in selection:
            values = self.chats_tree.item(item)['values']
            chat_id = int(values[1])
            chat_source = self.bot.chat_manager.get_chat_info(chat_id)
            if chat_source:
                chat_source.enabled = False
            updated_values = list(values)
            updated_values[4] = "Nein"
            self.chats_tree.item(item, values=updated_values)

        self.bot.chat_manager.save_config()
        self.update_overview_metrics()
        messagebox.showinfo("Deaktiviert", f"{len(selection)} Chat(s) deaktiviert")

    def refresh_statistics(self):
        if not hasattr(self, 'stats_tree'):
            return

        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

        for chat_source in self.bot.chat_manager.chat_sources.values():
            stats = self.bot.trade_tracker.get_source_statistics(chat_source.chat_name)
            last_trade = "Nie"
            if stats['last_trade']:
                last_trade = stats['last_trade'].strftime("%d.%m %H:%M")
            self.stats_tree.insert('', 'end', values=(
                chat_source.chat_name,
                stats['total_trades'],
                f"{stats['win_rate']:.1f}%",
                f"{stats['total_profit']:.2f}",
                last_trade
            ))

    def refresh_trades_view(self):
        if not hasattr(self, 'trades_tree'):
            return

        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)

        trades = sorted(self.bot.trade_tracker.trade_records.values(), key=lambda t: t.timestamp, reverse=True)
        for record in trades:
            values = (
                record.ticket,
                record.symbol,
                record.direction,
                record.source_chat_name,
                f"{record.entry_price:.2f}",
                f"{record.stop_loss:.2f}" if record.stop_loss else "-",
                f"{record.take_profit:.2f}" if record.take_profit else "-",
                record.timestamp.strftime("%d.%m %H:%M")
            )
            self.trades_tree.insert('', 'end', values=values)

    def toggle_demo_mode(self):
        self.bot.demo_mode = self.demo_var.get()
        self.config['trading']['demo_mode'] = bool(self.bot.demo_mode)
        self.config_manager.save_config(self.config)
        self.append_log(f"Modus geändert: {'Demo' if self.bot.demo_mode else 'Live'}")
        self.update_overview_metrics()

    def save_signal_settings(self):
        try:
            lot_size = float(self.lot_size_var.get())
            max_trades = int(float(self.max_trades_var.get()))
        except ValueError:
            messagebox.showerror("Ungültige Eingabe", "Bitte gib gültige Zahlen für Lot-Größe und Max-Trades an.")
            return

        self.config['trading']['default_lot_size'] = lot_size
        self.config['trading']['max_trades_per_hour'] = max_trades
        self.config.setdefault('signals', {})['auto_tp_sl'] = bool(self.auto_tp_var.get())
        self.config['signals']['require_confirmation'] = bool(self.require_confirm_var.get())
        self.config_manager.save_config(self.config)
        self.append_log("Automationseinstellungen gespeichert.")
        messagebox.showinfo("Gespeichert", "Automationseinstellungen wurden übernommen.")

    def update_status_bar(self, message: str):
        if hasattr(self, 'status_var'):
            self.status_var.set(message)

    def run(self):
        self.root.mainloop()


# ==================== STARTUP FLOW ====================


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


# ==================== MAIN ====================

def main():
    """Hauptfunktion für die modernisierte Oberfläche"""
    if not show_startup_warning():
        print("Programm abgebrochen.")
        return

    try:
        app = TradingGUI()
        app.run()
    except Exception as e:
        print(f"Fehler beim Starten der Anwendung: {e}")
        input("Drücken Sie Enter zum Beenden...")


if __name__ == "__main__":
    main()
