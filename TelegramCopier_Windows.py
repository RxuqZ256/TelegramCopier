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
CARD_BG = "#0B1224"
CARD_ALT_BG = "#111C30"
PANEL_BG = "#050A18"
ACCENT_COLOR = "#16F6A6"
ACCENT_COLOR_SECONDARY = "#05C3DD"
WARNING_COLOR = "#FFB400"
DANGER_COLOR = "#FF5E6B"
TEXT_PRIMARY = "#F8FAFC"
TEXT_SECONDARY = "#94A3B8"
TEXT_MUTED = "#64748B"

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
        self.api_id = int(api_id) if str(api_id).isdigit() else 0
        self.api_hash = api_hash
        self.phone = phone

        # Components
        self.chat_manager = MultiChatManager()
        self.trade_tracker = TradeTracker()
        self.signal_processor = SignalProcessor()

        # Telegram Client
        self.client = TelegramClient('trading_session', self.api_id, self.api_hash)

        # Message Queue für GUI
        self.message_queue: "queue.Queue" = queue.Queue()

        # Status
        self.is_running = False
        self.demo_mode = True  # Immer mit Demo starten!
        self.pending_trade_updates: Dict[int, Dict] = {}

    async def start(self):
        """Bot starten"""
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
    """Modernisierte Haupt-GUI im Stil des Risk Control Centers."""

    def __init__(self, config: Dict):
        self.config = config
        telegram_cfg = config.get('telegram', {})
        trading_cfg = config.get('trading', {})

        self.root = tk.Tk()
        self.root.title("Telegram Copier – Control Center")
        self.root.geometry("1320x900")
        self.root.minsize(1180, 820)
        self.root.configure(bg=DARK_BG)

        self.bot = MultiChatTradingBot(
            str(telegram_cfg.get('api_id', "0")),
            telegram_cfg.get('api_hash', ""),
            telegram_cfg.get('phone', "")
        )
        self.bot.demo_mode = bool(trading_cfg.get('demo_mode', True))

        self.metric_vars: Dict[str, tk.StringVar] = {}
        self.chat_metric_vars: Dict[str, tk.StringVar] = {}
        self.performance_vars: Dict[str, tk.StringVar] = {}
        self.selected_chat = tk.StringVar(value="")

        self.configure_theme()
        self.create_widgets()
        self.setup_message_processing()
        self.update_all_views()

    # ------------------------------------------------------------------
    # THEME & LAYOUT
    # ------------------------------------------------------------------
    def configure_theme(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except Exception:
            pass

        self.style.configure('Dark.TFrame', background=DARK_BG)
        self.style.configure('CardBackground.TFrame', background=CARD_BG)
        self.style.configure('CardInner.TFrame', background=CARD_BG)
        self.style.configure('Header.TFrame', background=PANEL_BG)
        self.style.configure('Footer.TFrame', background=PANEL_BG)
        self.style.configure('MetricCard.TFrame', background=CARD_BG)
        self.style.configure('StatusCard.TFrame', background=CARD_BG)

        self.style.configure('Dark.TLabel', background=DARK_BG, foreground=TEXT_PRIMARY, font=('Segoe UI', 11))
        self.style.configure('Title.TLabel', background=PANEL_BG, foreground=TEXT_PRIMARY, font=('Segoe UI Semibold', 22))
        self.style.configure('Subtitle.TLabel', background=PANEL_BG, foreground=TEXT_SECONDARY, font=('Segoe UI', 12))
        self.style.configure('SectionTitle.TLabel', background=CARD_BG, foreground=TEXT_PRIMARY, font=('Segoe UI Semibold', 14))
        self.style.configure('StatusTitle.TLabel', background=CARD_BG, foreground=TEXT_PRIMARY, font=('Segoe UI Semibold', 18))
        self.style.configure('LabelMuted.TLabel', background=CARD_BG, foreground=TEXT_SECONDARY, font=('Segoe UI', 10))
        self.style.configure('LabelHint.TLabel', background=CARD_BG, foreground=TEXT_MUTED, font=('Segoe UI', 10))
        self.style.configure('MetricValue.TLabel', background=CARD_BG, foreground=TEXT_PRIMARY, font=('Segoe UI Semibold', 24))

        self.style.configure('Accent.TButton', background=ACCENT_COLOR, foreground='#00111B', font=('Segoe UI Semibold', 11), borderwidth=0)
        self.style.map('Accent.TButton', background=[('active', '#12d992')])
        self.style.configure('Secondary.TButton', background='#1B243A', foreground=TEXT_SECONDARY, font=('Segoe UI', 11), borderwidth=0)
        self.style.map('Secondary.TButton', background=[('active', '#24304c')])
        self.style.configure('Ghost.TButton', background=CARD_ALT_BG, foreground=TEXT_SECONDARY, font=('Segoe UI', 11), borderwidth=0)
        self.style.map('Ghost.TButton', background=[('active', '#1f2c46')])

        self.style.configure('Switch.TCheckbutton', background=PANEL_BG, foreground=TEXT_SECONDARY, font=('Segoe UI', 11))
        self.style.map('Switch.TCheckbutton', foreground=[('active', TEXT_PRIMARY)])

        self.style.configure('Dark.TNotebook', background=DARK_BG, borderwidth=0)
        self.style.configure('Dark.TNotebook.Tab', background=CARD_BG, foreground=TEXT_SECONDARY, padding=(18, 10))
        self.style.map('Dark.TNotebook.Tab', background=[('selected', CARD_ALT_BG)], foreground=[('selected', TEXT_PRIMARY)])

        self.style.configure('Dark.Treeview', background=CARD_BG, fieldbackground=CARD_BG, foreground=TEXT_PRIMARY, bordercolor=CARD_ALT_BG, rowheight=34)
        self.style.map('Dark.Treeview', background=[('selected', '#1E263A')])
        self.style.configure('Dark.Treeview.Heading', background=CARD_ALT_BG, foreground=TEXT_SECONDARY, font=('Segoe UI Semibold', 10))

        self.style.configure('Accent.Horizontal.TProgressbar', troughcolor=CARD_ALT_BG, bordercolor=CARD_ALT_BG, background=ACCENT_COLOR, lightcolor=ACCENT_COLOR, darkcolor=ACCENT_COLOR)

        self.style.configure('Dark.TCombobox', fieldbackground=CARD_BG, background=CARD_BG, foreground=TEXT_PRIMARY)

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, style='Dark.TFrame', padding=(18, 18, 18, 28))
        self.main_frame.pack(fill='both', expand=True)

        self.create_header(self.main_frame)
        self.create_quick_metrics(self.main_frame)
        self.create_notebook(self.main_frame)
        self.create_log_panel(self.main_frame)
        self.create_status_bar(self.main_frame)

    def create_header(self, parent):
        header = ttk.Frame(parent, style='Header.TFrame', padding=24)
        header.pack(fill='x', pady=(0, 20))

        title_block = ttk.Frame(header, style='Header.TFrame')
        title_block.pack(side='left', anchor='w')

        ttk.Label(title_block, text="Risk Control Center", style='Title.TLabel').pack(anchor='w')
        ttk.Label(title_block, text="Live Monitoring for your Telegram Copier", style='Subtitle.TLabel').pack(anchor='w', pady=(8, 0))

        controls = ttk.Frame(header, style='Header.TFrame')
        controls.pack(side='right', anchor='e')

        self.demo_var = tk.BooleanVar(value=self.bot.demo_mode)
        ttk.Checkbutton(controls, text="Demo-Modus", variable=self.demo_var, command=self.toggle_demo_mode, style='Switch.TCheckbutton').pack(side='left', padx=(0, 18))
        ttk.Button(controls, text="Select Chats", command=self.open_chat_manager, style='Ghost.TButton').pack(side='left', padx=(0, 14))
        ttk.Button(controls, text="Start Bot", command=self.start_bot, style='Accent.TButton').pack(side='left', padx=(0, 12))
        ttk.Button(controls, text="Stop Bot", command=self.stop_bot, style='Secondary.TButton').pack(side='left')

    def create_quick_metrics(self, parent):
        metrics_frame = ttk.Frame(parent, style='Dark.TFrame')
        metrics_frame.pack(fill='x', pady=(0, 24))

        cards = [
            ("Status", "Systemstatus", 'status', ACCENT_COLOR),
            ("Open Signals", "Aktive Telegram-Signale", 'open_signals', ACCENT_COLOR_SECONDARY),
            ("Daily P&L", "Heutiges Ergebnis", 'daily_profit', WARNING_COLOR),
            ("Max Drawdown", "Seit Start", 'drawdown', DANGER_COLOR),
        ]

        for idx, (title, subtitle, key, accent) in enumerate(cards):
            var = tk.StringVar(value="–")
            self.metric_vars[key] = var
            card = ttk.Frame(metrics_frame, style='MetricCard.TFrame', padding=22)
            card.grid(row=0, column=idx, sticky='nsew', padx=(0 if idx == 0 else 18, 0))
            metrics_frame.columnconfigure(idx, weight=1)

            ttk.Label(card, text=title, style='LabelMuted.TLabel').pack(anchor='w')
            ttk.Label(card, textvariable=var, style='MetricValue.TLabel').pack(anchor='w', pady=(6, 4))
            ttk.Label(card, text=subtitle, style='LabelHint.TLabel').pack(anchor='w')
            tk.Frame(card, bg=accent, height=3, bd=0, highlightthickness=0).pack(fill='x', pady=(14, 0))

    def create_notebook(self, parent):
        self.notebook = ttk.Notebook(parent, style='Dark.TNotebook')
        self.notebook.pack(fill='both', expand=True)

        self.dashboard_tab = ttk.Frame(self.notebook, style='CardBackground.TFrame')
        self.chat_stats_tab = ttk.Frame(self.notebook, style='CardBackground.TFrame')
        self.automation_tab = ttk.Frame(self.notebook, style='CardBackground.TFrame')
        self.performance_tab = ttk.Frame(self.notebook, style='CardBackground.TFrame')

        self.notebook.add(self.dashboard_tab, text="Risk Control")
        self.notebook.add(self.chat_stats_tab, text="Chat Analytics")
        self.notebook.add(self.automation_tab, text="Automation")
        self.notebook.add(self.performance_tab, text="Performance")

        self.create_dashboard_tab(self.dashboard_tab)
        self.create_chat_statistics_tab(self.chat_stats_tab)
        self.create_automation_tab(self.automation_tab)
        self.create_performance_tab(self.performance_tab)

    def create_dashboard_tab(self, tab):
        content = ttk.Frame(tab, style='CardBackground.TFrame', padding=30)
        content.pack(fill='both', expand=True)

        status_row = ttk.Frame(content, style='CardBackground.TFrame')
        status_row.pack(fill='x')

        self.safe_status = self._create_status_block(status_row, "SAFE", "Keine Maßnahmen erforderlich", ACCENT_COLOR, padx=(0, 20))
        self.emergency_status = self._create_status_block(status_row, "EMERGENCY FLAT", "Sicherheitsmodus aktiviert", DANGER_COLOR, padx=(0, 0))

        grid = ttk.Frame(content, style='CardBackground.TFrame')
        grid.pack(fill='both', expand=True, pady=(24, 0))

        exposure_card = ttk.Frame(grid, style='CardInner.TFrame', padding=20)
        exposure_card.grid(row=0, column=0, sticky='nsew', padx=(0, 20), pady=(0, 20))
        ttk.Label(exposure_card, text="Exposure by Symbol", style='SectionTitle.TLabel').pack(anchor='w')
        self.exposure_tree = ttk.Treeview(exposure_card, columns=('symbol', 'count', 'volume'), show='headings', style='Dark.Treeview', height=7)
        self.exposure_tree.pack(fill='both', expand=True, pady=(14, 0))
        self.exposure_tree.heading('symbol', text='Symbol')
        self.exposure_tree.heading('count', text='Positionen')
        self.exposure_tree.heading('volume', text='Gesamtvolumen')
        self.exposure_tree.column('symbol', width=120, anchor='w')
        self.exposure_tree.column('count', width=110, anchor='center')
        self.exposure_tree.column('volume', width=140, anchor='e')

        signals_card = ttk.Frame(grid, style='CardInner.TFrame', padding=20)
        signals_card.grid(row=0, column=1, sticky='nsew', pady=(0, 20))
        ttk.Label(signals_card, text="Open Signals", style='SectionTitle.TLabel').pack(anchor='w')
        self.open_signal_progress = ttk.Progressbar(signals_card, style='Accent.Horizontal.TProgressbar', maximum=100)
        self.open_signal_progress.pack(fill='x', pady=(20, 6))
        self.open_signal_summary = ttk.Label(signals_card, text="0 aktive Signale", style='LabelHint.TLabel')
        self.open_signal_summary.pack(anchor='w')

        ttk.Separator(signals_card, orient='horizontal').pack(fill='x', pady=18)
        self.session_limits_var = tk.StringVar(value="Session Limits: 0 / 0")
        ttk.Label(signals_card, textvariable=self.session_limits_var, style='LabelMuted.TLabel').pack(anchor='w')

        table_card = ttk.Frame(content, style='CardInner.TFrame', padding=20)
        table_card.pack(fill='both', expand=True, pady=(12, 0))
        ttk.Label(table_card, text="Open Signals", style='SectionTitle.TLabel').pack(anchor='w')
        columns = ('signal', 'risk', 'heatmap', 'latency', 'hedging', 'leverage', 'value')
        self.open_signals_tree = ttk.Treeview(table_card, columns=columns, show='headings', style='Dark.Treeview', height=8)
        self.open_signals_tree.pack(fill='both', expand=True, pady=(14, 0))
        headings = [
            ('signal', 'Signal'),
            ('risk', 'Risk'),
            ('heatmap', 'Heatmap'),
            ('latency', 'Latency'),
            ('hedging', 'Hedging'),
            ('leverage', 'Leverage'),
            ('value', 'Value'),
        ]
        for key, text in headings:
            self.open_signals_tree.heading(key, text=text)
        self.open_signals_tree.column('signal', width=210, anchor='w')
        self.open_signals_tree.column('risk', width=90, anchor='center')
        self.open_signals_tree.column('heatmap', width=120, anchor='center')
        self.open_signals_tree.column('latency', width=90, anchor='center')
        self.open_signals_tree.column('hedging', width=100, anchor='center')
        self.open_signals_tree.column('leverage', width=90, anchor='center')
        self.open_signals_tree.column('value', width=110, anchor='e')

        scrollbar = ttk.Scrollbar(table_card, orient='vertical', command=self.open_signals_tree.yview)
        self.open_signals_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

        footer = ttk.Frame(content, style='CardBackground.TFrame')
        footer.pack(fill='x', pady=(18, 0))
        self.compliance_var = tk.StringVar(value="0 compliance alerts | Latency to Broker 0 ms")
        ttk.Label(footer, textvariable=self.compliance_var, style='LabelHint.TLabel').pack(anchor='w')

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

    def _create_status_block(self, parent, title: str, subtitle: str, color: str, padx=(0, 0)):
        frame = ttk.Frame(parent, style='StatusCard.TFrame', padding=20)
        frame.pack(side='left', fill='x', expand=True, padx=padx)
        canvas = tk.Canvas(frame, width=42, height=42, bg=CARD_BG, highlightthickness=0, bd=0)
        canvas.pack(anchor='w')
        circle = canvas.create_oval(4, 4, 38, 38, fill=color, outline=color)
        label = ttk.Label(frame, text=title, style='StatusTitle.TLabel')
        label.pack(anchor='w', pady=(10, 2))
        ttk.Label(frame, text=subtitle, style='LabelHint.TLabel').pack(anchor='w')
        return {'frame': frame, 'label': label, 'canvas': canvas, 'circle': circle}

    def create_chat_statistics_tab(self, tab):
        content = ttk.Frame(tab, style='CardBackground.TFrame', padding=30)
        content.pack(fill='both', expand=True)

        top = ttk.Frame(content, style='CardBackground.TFrame')
        top.pack(fill='x')
        ttk.Label(top, text="Chat auswählen", style='LabelMuted.TLabel').pack(side='left', anchor='w')
        self.chat_selector = ttk.Combobox(top, textvariable=self.selected_chat, style='Dark.TCombobox', state='readonly', width=28)
        self.chat_selector.pack(side='left', padx=(14, 0))
        self.chat_selector.bind('<<ComboboxSelected>>', lambda _: self.update_chat_statistics())

        metric_grid = ttk.Frame(content, style='CardBackground.TFrame')
        metric_grid.pack(fill='x', pady=(24, 16))

        metrics = [
            ('winrate', 'Winrate'),
            ('risk_reward', 'R / R'),
            ('signals', 'Signals'),
            ('profit', 'Profit'),
        ]
        for idx, (key, label) in enumerate(metrics):
            var = tk.StringVar(value='–')
            self.chat_metric_vars[key] = var
            card = ttk.Frame(metric_grid, style='MetricCard.TFrame', padding=20)
            card.grid(row=0, column=idx, sticky='nsew', padx=(0 if idx == 0 else 18, 0))
            metric_grid.columnconfigure(idx, weight=1)
            ttk.Label(card, text=label, style='LabelMuted.TLabel').pack(anchor='w')
            ttk.Label(card, textvariable=var, style='MetricValue.TLabel').pack(anchor='w', pady=(6, 4))
            ttk.Label(card, text="Aktualisiert live", style='LabelHint.TLabel').pack(anchor='w')

        chart_row = ttk.Frame(content, style='CardBackground.TFrame')
        chart_row.pack(fill='both', expand=True)

        heatmap_card = ttk.Frame(chart_row, style='CardInner.TFrame', padding=20)
        heatmap_card.grid(row=0, column=0, sticky='nsew', padx=(0, 20))
        ttk.Label(heatmap_card, text="Session Heatmap", style='SectionTitle.TLabel').pack(anchor='w')
        self.heatmap_canvas = tk.Canvas(heatmap_card, width=430, height=220, bg=CARD_BG, highlightthickness=0, bd=0)
        self.heatmap_canvas.pack(fill='both', expand=True, pady=(16, 0))

        stats_card = ttk.Frame(chart_row, style='CardInner.TFrame', padding=20)
        stats_card.grid(row=0, column=1, sticky='nsew')
        ttk.Label(stats_card, text="Pair Distribution", style='SectionTitle.TLabel').pack(anchor='w')
        self.distribution_vars = []
        for pair in ("EUR/USD", "GBP/USD", "USD/JPY"):
            bar_var = tk.StringVar(value="0%")
            frame = ttk.Frame(stats_card, style='CardInner.TFrame')
            frame.pack(fill='x', pady=(12, 0))
            ttk.Label(frame, text=pair, style='LabelMuted.TLabel').pack(anchor='w')
            progress = ttk.Progressbar(frame, style='Accent.Horizontal.TProgressbar', maximum=100)
            progress.pack(fill='x', pady=(6, 2))
            value_label = ttk.Label(frame, textvariable=bar_var, style='LabelHint.TLabel')
            value_label.pack(anchor='e')
            self.distribution_vars.append((progress, bar_var))

        ttk.Separator(stats_card, orient='horizontal').pack(fill='x', pady=18)
        ttk.Label(stats_card, text="Accuracy by Pair", style='SectionTitle.TLabel').pack(anchor='w')
        self.accuracy_vars = {
            'EURUSD': (tk.StringVar(value="0.0%"), 'EUR/USD'),
            'GBPUSD': (tk.StringVar(value="0.0%"), 'GBP/USD'),
        }
        for symbol, (var, label) in self.accuracy_vars.items():
            ttk.Label(stats_card, text=label, style='LabelMuted.TLabel').pack(anchor='w', pady=(12, 0))
            ttk.Label(stats_card, textvariable=var, style='MetricValue.TLabel').pack(anchor='w')

        chart_row.columnconfigure(0, weight=1)
        chart_row.columnconfigure(1, weight=1)

    def create_automation_tab(self, tab):
        content = ttk.Frame(tab, style='CardBackground.TFrame', padding=30)
        content.pack(fill='both', expand=True)

        header = ttk.Frame(content, style='CardBackground.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text="Automation Rules", style='SectionTitle.TLabel').pack(side='left', anchor='w')
        add_rule_btn = ttk.Button(
            header,
            text="Add Rule",
            style='Accent.TButton',
            command=lambda: messagebox.showinfo("Info", "Regel-Editor folgt.")
        )
        add_rule_btn.pack(side='right')

        self.rule_container = ttk.Frame(content, style='CardBackground.TFrame')
        self.rule_container.pack(fill='both', expand=True, pady=(20, 0))

        self.default_rules = [
            "IF Chat is GoldTrading VIP AND Pair is XAUUSD THEN Execute trade with Risk Fixed 0.8%",
            "IF Chat is Premium Forex Signals AND Pair is NOT GBPUSD THEN Mute chat for 12h",
            "IF Slippage > 0.5 THEN Notify via Telegram with Alert Preset High",
        ]

        self.render_rules()

    def render_rules(self):
        for child in self.rule_container.winfo_children():
            child.destroy()

        for rule in self.default_rules:
            card = ttk.Frame(self.rule_container, style='CardInner.TFrame', padding=20)
            card.pack(fill='x', pady=(0, 16))
            ttk.Label(card, text=rule, style='LabelMuted.TLabel', wraplength=820, justify='left').pack(anchor='w')

    def create_performance_tab(self, tab):
        content = ttk.Frame(tab, style='CardBackground.TFrame', padding=30)
        content.pack(fill='both', expand=True)

        metrics_frame = ttk.Frame(content, style='CardBackground.TFrame')
        metrics_frame.pack(fill='x')
        performance_metrics = [
            ('sharpe', 'Sharpe'),
            ('sortino', 'Sortino'),
            ('max_drawdown', 'Max Drawdown'),
            ('win_rate', 'Win Rate'),
            ('profit', 'Profit'),
        ]
        for idx, (key, title) in enumerate(performance_metrics):
            var = tk.StringVar(value='–')
            self.performance_vars[key] = var
            card = ttk.Frame(metrics_frame, style='MetricCard.TFrame', padding=20)
            card.grid(row=0, column=idx, sticky='nsew', padx=(0 if idx == 0 else 18, 0))
            metrics_frame.columnconfigure(idx, weight=1)
            ttk.Label(card, text=title, style='LabelMuted.TLabel').pack(anchor='w')
            ttk.Label(card, textvariable=var, style='MetricValue.TLabel').pack(anchor='w', pady=(6, 4))
            ttk.Label(card, text="Letzte 90 Tage", style='LabelHint.TLabel').pack(anchor='w')

        chart_area = ttk.Frame(content, style='CardBackground.TFrame')
        chart_area.pack(fill='both', expand=True, pady=(24, 0))

        equity_card = ttk.Frame(chart_area, style='CardInner.TFrame', padding=20)
        equity_card.grid(row=0, column=0, sticky='nsew', padx=(0, 20))
        ttk.Label(equity_card, text="Equity Curve", style='SectionTitle.TLabel').pack(anchor='w')
        self.equity_canvas = tk.Canvas(equity_card, height=260, bg=CARD_BG, highlightthickness=0, bd=0)
        self.equity_canvas.pack(fill='both', expand=True, pady=(16, 0))

        distribution_card = ttk.Frame(chart_area, style='CardInner.TFrame', padding=20)
        distribution_card.grid(row=0, column=1, sticky='nsew')
        ttk.Label(distribution_card, text="Profit Distribution", style='SectionTitle.TLabel').pack(anchor='w')
        self.profit_distribution_canvas = tk.Canvas(distribution_card, height=260, bg=CARD_BG, highlightthickness=0, bd=0)
        self.profit_distribution_canvas.pack(fill='both', expand=True, pady=(16, 0))

        chart_area.columnconfigure(0, weight=1)
        chart_area.columnconfigure(1, weight=1)

    def create_log_panel(self, parent):
        panel = ttk.Frame(parent, style='CardBackground.TFrame', padding=20)
        panel.pack(fill='both', expand=False, pady=(24, 0))
        ttk.Label(panel, text="Live Activity Log", style='SectionTitle.TLabel').pack(anchor='w')
        container = ttk.Frame(panel, style='CardBackground.TFrame')
        container.pack(fill='both', expand=True, pady=(12, 0))
        self.log_text = tk.Text(container, height=8, bg=CARD_BG, fg=TEXT_SECONDARY, insertbackground=TEXT_PRIMARY, bd=0, highlightthickness=0, wrap='word')
        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=self.log_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def create_status_bar(self, parent):
        bar = ttk.Frame(parent, style='Footer.TFrame', padding=(20, 12))
        bar.pack(fill='x', pady=(20, 0))
        self.status_label = ttk.Label(bar, text="Bot gestoppt", style='Subtitle.TLabel')
        self.status_label.pack(side='left')
        self.latency_var = tk.StringVar(value="Latency: n/a")
        ttk.Label(bar, textvariable=self.latency_var, style='Subtitle.TLabel').pack(side='right')

    # ------------------------------------------------------------------
    # DATA UPDATES
    # ------------------------------------------------------------------
    def update_all_views(self):
        self.update_dashboard_metrics()
        self.update_chat_statistics()
        self.update_performance_metrics()

    def update_dashboard_metrics(self):
        records = list(self.bot.trade_tracker.trade_records.values())
        records.sort(key=lambda r: r.timestamp)
        total_trades = len(records)
        self.metric_vars['open_signals'].set(str(total_trades))

        today = datetime.now().date()
        daily_profit = sum(record.profit_loss for record in records if record.timestamp.date() == today)
        self.metric_vars['daily_profit'].set(f"{daily_profit:+.2f} €")

        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for record in records:
            cumulative += record.profit_loss
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_drawdown = max(max_drawdown, drawdown)

        drawdown_text = f"-{max_drawdown:.2f}"
        if peak > 0:
            percent = (max_drawdown / peak) * 100
            drawdown_text = f"-{percent:.1f}%"
        self.metric_vars['drawdown'].set(drawdown_text)

        status_text = "SAFE" if max_drawdown < 5 else "ATTENTION"
        self.metric_vars['status'].set(status_text)
        self.safe_status['label'].configure(text=status_text)
        self.emergency_status['label'].configure(text="EMERGENCY FLAT" if max_drawdown > 10 else "STANDBY")
        self.open_signal_progress['value'] = min(total_trades * 12, 100)
        self.open_signal_summary.configure(text=f"{total_trades} aktive Signale")

        max_per_session = 25
        self.session_limits_var.set(f"Session Limits: {total_trades} / {max_per_session}")

        for item in self.exposure_tree.get_children():
            self.exposure_tree.delete(item)
        exposure: Dict[str, Dict[str, float]] = {}
        for record in records:
            info = exposure.setdefault(record.symbol, {'count': 0, 'volume': 0.0})
            info['count'] += 1
            info['volume'] += abs(record.take_profit - record.entry_price) * max(record.lot_size, 0.01)
        for symbol, data in sorted(exposure.items(), key=lambda kv: kv[1]['count'], reverse=True):
            self.exposure_tree.insert('', 'end', values=(symbol, data['count'], f"{data['volume']:.2f}"))

        for item in self.open_signals_tree.get_children():
            self.open_signals_tree.delete(item)
        now = datetime.now()
        for record in reversed(records):
            latency_ms = max(int((now - record.timestamp).total_seconds() * 1000), 0)
            risk_map = {1: 'Low', 2: 'Medium', 3: 'High'}
            risk_text = risk_map.get(record.source_priority, '—')
            hedging = 'Enabled' if record.direction.upper() == 'SELL' else 'Neutral'
            leverage = '1:30'
            value = f"{record.profit_loss:+.2f}"
            heat = (now - record.timestamp).total_seconds() / 60
            heat_text = f"{max(0, 100 - int(heat))}%"
            self.open_signals_tree.insert('', 'end', values=(
                f"{record.source_chat_name} – {record.symbol}",
                risk_text,
                heat_text,
                f"{latency_ms} ms",
                hedging,
                leverage,
                value
            ))

        compliance_alerts = sum(1 for data in exposure.values() if data['count'] > 5)
        broker_latency = min(self.open_signal_progress['value'] * 2, 250)
        latency_text = f"Latency to Broker {int(broker_latency)} ms"
        self.compliance_var.set(f"{compliance_alerts} compliance alerts | {latency_text}")
        self.latency_var.set(f"Latency: {int(broker_latency)} ms")

    def update_chat_statistics(self):
        sources = list(self.bot.chat_manager.chat_sources.values())
        sources.sort(key=lambda src: src.chat_name.lower())
        names = [src.chat_name for src in sources]
        self.chat_selector['values'] = names
        selected_name = self.selected_chat.get()
        if sources and selected_name not in names:
            selected_name = names[0]
            self.selected_chat.set(selected_name)
        if not sources:
            for var in self.chat_metric_vars.values():
                var.set('–')
            self.draw_heatmap([[0] * 6 for _ in range(7)])
            for progress, text_var in self.distribution_vars:
                progress['value'] = 0
                text_var.set('0%')
            for var, _label in self.accuracy_vars.values():
                var.set('0.0%')
            return

        chat = next((src for src in sources if src.chat_name == selected_name), sources[0])
        trades = self.bot.trade_tracker.get_trades_by_source(chat.chat_name)
        stats = self.bot.trade_tracker.get_source_statistics(chat.chat_name)

        winrate = stats['win_rate']
        self.chat_metric_vars['winrate'].set(f"{winrate:.1f}%")
        self.chat_metric_vars['signals'].set(str(stats['total_trades']))
        self.chat_metric_vars['profit'].set(f"{stats['total_profit']:+.2f}")

        positives = [t.profit_loss for t in trades if t.profit_loss > 0]
        negatives = [-t.profit_loss for t in trades if t.profit_loss < 0]
        avg_pos = sum(positives) / len(positives) if positives else 0.0
        avg_neg = sum(negatives) / len(negatives) if negatives else 0.0
        rr = avg_pos / avg_neg if avg_neg else 0.0
        self.chat_metric_vars['risk_reward'].set(f"{rr:.2f}")

        heatmap = [[0 for _ in range(6)] for _ in range(7)]
        for trade in trades:
            day = trade.timestamp.weekday()
            slot = min(int(trade.timestamp.hour / 4), 5)
            heatmap[day][slot] += 1
        self.draw_heatmap(heatmap)

        distribution = {}
        for trade in trades:
            distribution[trade.symbol] = distribution.get(trade.symbol, 0) + 1
        total = sum(distribution.values()) or 1
        pairs = list(distribution.items())
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        for idx, (progress, text_var) in enumerate(self.distribution_vars):
            if idx < len(pairs):
                symbol, count = pairs[idx]
                percent = (count / total) * 100
                progress['value'] = percent
                text_var.set(f"{percent:.1f}% ({symbol})")
            else:
                progress['value'] = 0
                text_var.set('0%')

        for symbol, (var, label) in self.accuracy_vars.items():
            count = distribution.get(symbol, 0)
            ratio = (count / total) if total else 0.0
            var.set(f"{ratio * winrate:.1f}%")

    def draw_heatmap(self, data):
        self.heatmap_canvas.delete('all')
        width = int(self.heatmap_canvas.winfo_width() or 430)
        height = int(self.heatmap_canvas.winfo_height() or 220)
        padding = 20
        cols = len(data[0]) if data else 1
        rows = len(data)
        cell_w = (width - padding * 2) / cols
        cell_h = (height - padding * 2) / rows
        max_value = max((value for row in data for value in row), default=1)
        days = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']
        for r, row in enumerate(data):
            self.heatmap_canvas.create_text(10, padding + r * cell_h + cell_h / 2, text=days[r], anchor='w', fill=TEXT_SECONDARY, font=('Segoe UI', 10))
            for c, value in enumerate(row):
                intensity = value / max_value if max_value else 0
                color = f"#{int(22 + intensity * 60):02x}{int(35 + intensity * 120):02x}{int(90 + intensity * 120):02x}"
                x0 = padding + c * cell_w
                y0 = padding + r * cell_h
                x1 = x0 + cell_w - 6
                y1 = y0 + cell_h - 6
                self.heatmap_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='#0c1628')

    def update_performance_metrics(self):
        records = list(self.bot.trade_tracker.trade_records.values())
        records.sort(key=lambda r: r.timestamp)
        if not records:
            for var in self.performance_vars.values():
                var.set('–')
            self.equity_canvas.delete('all')
            self.profit_distribution_canvas.delete('all')
            return

        profits = [record.profit_loss for record in records]
        total_profit = sum(profits)
        wins = sum(1 for p in profits if p > 0)
        win_rate = (wins / len(profits)) * 100 if profits else 0.0

        mean = total_profit / len(profits)
        variance = sum((p - mean) ** 2 for p in profits) / len(profits) if profits else 0.0
        std_dev = variance ** 0.5
        negatives = [p for p in profits if p < 0]
        downside = sum((p - mean) ** 2 for p in negatives) / len(negatives) if negatives else 0.0
        downside_dev = downside ** 0.5

        sharpe = mean / std_dev if std_dev else 0.0
        sortino = mean / downside_dev if downside_dev else 0.0

        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        equity_curve = []
        for record in records:
            cumulative += record.profit_loss
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_drawdown = max(max_drawdown, drawdown)
            equity_curve.append((record.timestamp, cumulative))

        self.performance_vars['sharpe'].set(f"{sharpe:.2f}")
        self.performance_vars['sortino'].set(f"{sortino:.2f}")
        self.performance_vars['max_drawdown'].set(f"-{max_drawdown:.2f}")
        self.performance_vars['win_rate'].set(f"{win_rate:.1f}%")
        self.performance_vars['profit'].set(f"{total_profit:+.2f}")

        self.draw_equity_curve(equity_curve)
        self.update_profit_distribution(profits)

    def draw_equity_curve(self, equity_points):
        canvas = self.equity_canvas
        canvas.delete('all')
        if not equity_points:
            return
        width = int(canvas.winfo_width() or 600)
        height = int(canvas.winfo_height() or 260)
        padding = 30
        min_equity = min(point[1] for point in equity_points)
        max_equity = max(point[1] for point in equity_points)
        span = max(max_equity - min_equity, 1)
        coords = []
        for idx, (_, value) in enumerate(equity_points):
            x = padding + (width - 2 * padding) * (idx / max(len(equity_points) - 1, 1))
            y = height - padding - ((value - min_equity) / span) * (height - 2 * padding)
            coords.append((x, y))
        for i in range(len(coords) - 1):
            canvas.create_line(*coords[i], *coords[i + 1], fill=ACCENT_COLOR, width=2.2)
        canvas.create_line(padding, height - padding, width - padding, height - padding, fill='#17233c')
        canvas.create_line(padding, padding, padding, height - padding, fill='#17233c')

    def update_profit_distribution(self, profits):
        canvas = self.profit_distribution_canvas
        canvas.delete('all')
        if not profits:
            return
        width = int(canvas.winfo_width() or 600)
        height = int(canvas.winfo_height() or 260)
        padding = 30
        bins = 8
        min_profit = min(profits)
        max_profit = max(profits)
        span = max(max_profit - min_profit, 1)
        histogram = [0 for _ in range(bins)]
        for profit in profits:
            index = int((profit - min_profit) / span * (bins - 1))
            histogram[index] += 1
        max_count = max(histogram) or 1
        bar_width = (width - 2 * padding) / bins
        for idx, count in enumerate(histogram):
            x0 = padding + idx * bar_width
            y1 = height - padding
            height_ratio = count / max_count
            y0 = y1 - (height - 2 * padding) * height_ratio
            canvas.create_rectangle(x0, y0, x0 + bar_width * 0.8, y1, fill=ACCENT_COLOR_SECONDARY, outline='')
        canvas.create_line(padding, height - padding, width - padding, height - padding, fill='#17233c')
        canvas.create_line(padding, padding, padding, height - padding, fill='#17233c')

    # ------------------------------------------------------------------
    # BOT CONTROL
    # ------------------------------------------------------------------
    def toggle_demo_mode(self):
        self.bot.demo_mode = self.demo_var.get()
        mode_text = "Demo-Modus" if self.bot.demo_mode else "LIVE-Modus"
        self.log_message(f"Modus geändert: {mode_text}")

    def start_bot(self):
        self.status_label.configure(text="Bot startet…")

        def run_bot():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.bot.start())
                self.root.after(0, lambda: self.status_label.configure(text="Bot läuft"))
            except Exception as exc:
                self.root.after(0, lambda: self.log_message(f"Bot-Start-Fehler: {exc}"))
            finally:
                try:
                    loop.run_forever()
                except Exception:
                    pass
                loop.close()

        threading.Thread(target=run_bot, daemon=True).start()

    def stop_bot(self):
        self.bot.is_running = False
        try:
            self.bot.client.disconnect()
        except Exception:
            pass
        self.status_label.configure(text="Bot gestoppt")
        self.log_message("Bot wurde gestoppt.")

    def open_chat_manager(self):
        ChatManagerDialog(self.root, self.bot)
        self.update_chat_statistics()

    # ------------------------------------------------------------------
    # LOGGING & MESSAGE PROCESSING
    # ------------------------------------------------------------------
    def setup_message_processing(self):
        def process_messages():
            try:
                while True:
                    msg_type, data = self.bot.message_queue.get(block=False)
                    if msg_type == 'LOG':
                        self.log_message(str(data))
                    elif msg_type == 'TRADE_EXECUTED':
                        self.log_message(f"Trade ausgeführt: {data}")
                        self.update_all_views()
                    elif msg_type == 'TRADE_UPDATED':
                        self.log_message(f"Trade aktualisiert: {data}")
                        self.update_all_views()
            except queue.Empty:
                pass
            self.root.after(150, process_messages)

        process_messages()

    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert('end', f"[{timestamp}] {message}\n")
        self.log_text.see('end')

    def run(self):
        self.root.mainloop()


class ChatManagerDialog:
    """Dialog zur Auswahl und Verwaltung der zu überwachenden Chats."""

    def __init__(self, parent, bot: MultiChatTradingBot):
        self.bot = bot
        self.window = tk.Toplevel(parent)
        self.window.title("Telegram Chats auswählen")
        self.window.geometry("780x560")
        self.window.configure(bg=DARK_BG)
        self.window.transient(parent)
        self.window.grab_set()
        self.window.focus_force()

        container = ttk.Frame(self.window, style='Dark.TFrame', padding=20)
        container.pack(fill='both', expand=True)

        header = ttk.Frame(container, style='Header.TFrame', padding=12)
        header.pack(fill='x', pady=(0, 16))
        ttk.Label(header, text="Select Telegram Chats", style='Title.TLabel').pack(anchor='w')
        ttk.Label(header, text="Aktiviere die Chats, aus denen Signale verarbeitet werden sollen.", style='Subtitle.TLabel').pack(anchor='w', pady=(6, 0))

        controls = ttk.Frame(container, style='Dark.TFrame')
        controls.pack(fill='x', pady=(0, 16))
        ttk.Button(controls, text="Chats laden", style='Accent.TButton', command=self.load_chats).pack(side='left')
        ttk.Button(controls, text="Überwachung aktivieren", style='Ghost.TButton', command=self.enable_monitoring).pack(side='left', padx=(12, 0))
        ttk.Button(controls, text="Überwachung deaktivieren", style='Secondary.TButton', command=self.disable_monitoring).pack(side='left', padx=(12, 0))

        table_card = ttk.Frame(container, style='CardInner.TFrame', padding=16)
        table_card.pack(fill='both', expand=True)
        columns = ('Name', 'ID', 'Typ', 'Teilnehmer', 'Überwacht', 'Signale')
        self.tree = ttk.Treeview(table_card, columns=columns, show='headings', style='Dark.Treeview', height=12)
        for col in columns:
            self.tree.heading(col, text=col)
        self.tree.column('Name', width=220, anchor='w')
        self.tree.column('ID', width=120, anchor='center')
        self.tree.column('Typ', width=80, anchor='center')
        self.tree.column('Teilnehmer', width=110, anchor='center')
        self.tree.column('Überwacht', width=100, anchor='center')
        self.tree.column('Signale', width=100, anchor='center')
        self.tree.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(table_card, orient='vertical', command=self.tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=scrollbar.set)

        footer = ttk.Frame(container, style='Footer.TFrame', padding=(16, 10))
        footer.pack(fill='x', pady=(16, 0))
        self.status_var = tk.StringVar(value="Bereit")
        ttk.Label(footer, textvariable=self.status_var, style='Subtitle.TLabel').pack(side='left')
        ttk.Button(footer, text="Schließen", style='Ghost.TButton', command=self.close).pack(side='right')

        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.wait_visibility()
        self.window.wait_window()

    # ------------------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------------------
    def load_chats(self):
        self.status_var.set("Chats werden geladen…")

        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                chats = loop.run_until_complete(self.bot.load_all_chats())
                self.window.after(0, lambda: self.update_chat_list(chats))
            except Exception as exc:
                self.window.after(0, lambda: self.status_var.set(f"Fehler: {exc}"))
            finally:
                loop.close()

        threading.Thread(target=run_async, daemon=True).start()

    def update_chat_list(self, chats_data):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for chat in chats_data:
            chat_source = self.bot.chat_manager.get_chat_info(chat['id'])
            is_monitored = "Ja" if chat_source and chat_source.enabled else "Nein"
            signal_count = chat_source.signal_count if chat_source else 0
            self.tree.insert('', 'end', values=(
                chat['name'],
                chat['id'],
                chat['type'],
                chat['participants'],
                is_monitored,
                signal_count
            ))

        self.status_var.set(f"{len(chats_data)} Chats geladen")

    def enable_monitoring(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Keine Auswahl", "Bitte wähle mindestens einen Chat aus.")
            return

        for item in selection:
            values = self.tree.item(item)['values']
            chat_id = int(values[1])
            chat_name = values[0]
            chat_type = values[2]
            self.bot.chat_manager.add_chat_source(chat_id, chat_name, chat_type, True)

            updated = list(values)
            updated[4] = "Ja"
            self.tree.item(item, values=updated)

        self.status_var.set(f"{len(selection)} Chat(s) aktiviert")

    def disable_monitoring(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Keine Auswahl", "Bitte wähle mindestens einen Chat aus.")
            return

        for item in selection:
            values = self.tree.item(item)['values']
            chat_id = int(values[1])
            chat_source = self.bot.chat_manager.get_chat_info(chat_id)
            if chat_source:
                chat_source.enabled = False
            updated = list(values)
            updated[4] = "Nein"
            self.tree.item(item, values=updated)

        self.bot.chat_manager.save_config()
        self.status_var.set(f"{len(selection)} Chat(s) deaktiviert")

    def close(self):
        self.window.grab_release()
        self.window.destroy()


# ==================== SETUP ASSISTANT & STARTUP ====================


class OnboardingScreen:
    """Onboarding-Screen im Stil des Telegram-Kopierers."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.result: Optional[Dict] = None
        self.root: Optional[tk.Tk] = None
        self.api_id_var: Optional[tk.StringVar] = None
        self.api_hash_var: Optional[tk.StringVar] = None
        self.phone_var: Optional[tk.StringVar] = None
        self.error_var: Optional[tk.StringVar] = None

    def run(self) -> Optional[Dict]:
        config = self.config_manager.load_config()

        self.root = tk.Tk()
        self.root.title("Telegram Copier – Onboarding")
        self.root.geometry("480x620")
        self.root.configure(bg=DARK_BG)
        self.root.resizable(False, False)

        self._configure_styles()

        self.api_id_var = tk.StringVar(value=str(config['telegram'].get('api_id', "")))
        self.api_hash_var = tk.StringVar(value=config['telegram'].get('api_hash', ""))
        self.phone_var = tk.StringVar(value=config['telegram'].get('phone', ""))
        self.error_var = tk.StringVar(value="")

        self._build_layout()

        self.root.bind('<Return>', lambda _: self.start_bot())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()
        return self.result

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass

        style.configure('Onboarding.Root.TFrame', background=DARK_BG)
        style.configure('Onboarding.Card.TFrame', background=CARD_BG)
        style.configure('Onboarding.Title.TLabel', background=DARK_BG, foreground=TEXT_PRIMARY, font=('Segoe UI Semibold', 24))
        style.configure('Onboarding.Step.TLabel', background=DARK_BG, foreground=TEXT_SECONDARY, font=('Segoe UI', 11))
        style.configure('Onboarding.Section.TLabel', background=CARD_BG, foreground=TEXT_PRIMARY, font=('Segoe UI Semibold', 16))
        style.configure('Onboarding.Text.TLabel', background=CARD_BG, foreground=TEXT_SECONDARY, font=('Segoe UI', 11))
        style.configure('Onboarding.List.TLabel', background=CARD_BG, foreground=TEXT_MUTED, font=('Segoe UI', 10))
        style.configure('Onboarding.Error.TLabel', background=CARD_BG, foreground=DANGER_COLOR, font=('Segoe UI', 10))
        style.configure('Onboarding.TEntry', fieldbackground=CARD_ALT_BG, background=CARD_ALT_BG, foreground=TEXT_PRIMARY)

    def _build_layout(self):
        container = ttk.Frame(self.root, style='Onboarding.Root.TFrame', padding=28)
        container.pack(fill='both', expand=True)

        ttk.Label(container, text="ONBOARDING", style='Onboarding.Title.TLabel').pack(anchor='center', pady=(10, 8))
        self._build_steps(container)

        card = ttk.Frame(container, style='Onboarding.Card.TFrame', padding=24)
        card.pack(fill='both', expand=True)

        ttk.Label(card, text="Connect Telegram", style='Onboarding.Section.TLabel').pack(anchor='w')
        ttk.Label(card, text="API ID", style='Onboarding.Text.TLabel').pack(anchor='w', pady=(18, 4))
        ttk.Entry(card, textvariable=self.api_id_var, style='Onboarding.TEntry').pack(fill='x')

        ttk.Label(card, text="API Hash", style='Onboarding.Text.TLabel').pack(anchor='w', pady=(14, 4))
        ttk.Entry(card, textvariable=self.api_hash_var, style='Onboarding.TEntry', show='•').pack(fill='x')

        ttk.Label(card, text="Telefonnummer", style='Onboarding.Text.TLabel').pack(anchor='w', pady=(14, 4))
        ttk.Entry(card, textvariable=self.phone_var, style='Onboarding.TEntry').pack(fill='x')

        ttk.Label(card, textvariable=self.error_var, style='Onboarding.Error.TLabel').pack(anchor='w', pady=(10, 0))

        checklist = ttk.Frame(card, style='Onboarding.Card.TFrame')
        checklist.pack(anchor='w', pady=(24, 16))
        for text in (
            "1. Connect Telegram",
            "2. Select Chats",
            "3. Set Risk",
            "4. Finish"
        ):
            ttk.Label(checklist, text=text, style='Onboarding.List.TLabel').pack(anchor='w')

        start_button = tk.Button(card, text="START BOT", bg=ACCENT_COLOR, fg="#021019", activebackground='#12d992',
                                 activeforeground='#021019', font=('Segoe UI Semibold', 12), bd=0, relief='flat',
                                 padx=18, pady=10, command=self.start_bot)
        start_button.pack(fill='x', pady=(10, 0))

        cancel_button = tk.Button(card, text="Abbrechen", bg=CARD_ALT_BG, fg=TEXT_SECONDARY,
                                   activebackground='#1f2c46', activeforeground=TEXT_PRIMARY, font=('Segoe UI', 11),
                                   bd=0, relief='flat', padx=18, pady=10, command=self.cancel)
        cancel_button.pack(fill='x', pady=(10, 0))

    def _build_steps(self, parent):
        steps_frame = ttk.Frame(parent, style='Onboarding.Root.TFrame')
        steps_frame.pack(pady=(12, 24))
        for idx in range(1, 5):
            canvas = tk.Canvas(steps_frame, width=42, height=42, bg=DARK_BG, highlightthickness=0, bd=0)
            canvas.pack(side='left')
            fill = ACCENT_COLOR if idx == 1 else CARD_ALT_BG
            outline = fill
            canvas.create_oval(4, 4, 38, 38, fill=fill, outline=outline)
            canvas.create_text(21, 21, text=str(idx), fill='#010b16' if idx == 1 else TEXT_SECONDARY,
                               font=('Segoe UI Semibold', 12))
            if idx < 4:
                bar = tk.Frame(steps_frame, bg=ACCENT_COLOR if idx == 1 else CARD_ALT_BG, width=46, height=4)
                bar.pack(side='left', padx=6, pady=18)

    def start_bot(self):
        api_id = (self.api_id_var.get().strip() if self.api_id_var else "")
        api_hash = (self.api_hash_var.get().strip() if self.api_hash_var else "")
        phone = (self.phone_var.get().strip() if self.phone_var else "")

        if not api_id or not api_hash:
            if self.error_var:
                self.error_var.set("Bitte API ID und API Hash ausfüllen.")
            return

        config = self.config_manager.load_config()
        config['telegram']['api_id'] = api_id
        config['telegram']['api_hash'] = api_hash
        config['telegram']['phone'] = phone
        self.config_manager.save_config(config)
        self.result = config
        if self.root:
            self.root.destroy()

    def cancel(self):
        self.result = None
        if self.root:
            self.root.destroy()

    def _on_close(self):
        self.result = None
        if self.root:
            self.root.destroy()


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
    """Hauptfunktion mit modernem Onboarding."""
    if not show_startup_warning():
        print("Programm abgebrochen.")
        return

    config_manager = ConfigManager()
    onboarding = OnboardingScreen(config_manager)
    config = onboarding.run()

    if not config:
        print("Setup abgebrochen.")
        return

    try:
        app = TradingGUI(config)
        app.run()
    except Exception as e:
        print(f"Fehler beim Starten der Anwendung: {e}")
        input("Drücken Sie Enter zum Beenden...")


if __name__ == "__main__":
    main()
