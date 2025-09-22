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
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Awaitable

# ---- optionale Abhängigkeit: MetaTrader5 (nur für Windows verfügbar) ----
try:
    import MetaTrader5 as mt5
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
    """Vereinfachter Signal-Prozessor"""

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

        self.auto_tp_sl: bool = True

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

        # Buy Zone
        match = re.search(self.patterns['buy_zone'], message_text)
        if match:
            base = match.group(1).upper()
            symbol = self.symbol_mapping.get(base, base)
            entry_price = self._parse_price(match.group(2))
            if entry_price is not None:
                return {
                    'kind': 'trade',
                    'type': 'zone',
                    'action': 'BUY',
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'source': chat_source.chat_name,
                    'execution_mode': ExecutionMode.ZONE_WAIT,
                    'stop_loss': stop_loss,
                    'take_profits': take_profits
                }

        # Sell Zone
        match = re.search(self.patterns['sell_zone'], message_text)
        if match:
            base = match.group(1).upper()
            symbol = self.symbol_mapping.get(base, base)
            entry_price = self._parse_price(match.group(2))
            if entry_price is not None:
                return {
                    'kind': 'trade',
                    'type': 'zone',
                    'action': 'SELL',
                    'symbol': symbol,
                    'entry_price': entry_price,
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
        self._mt5_account_info: Optional[Dict[str, str]] = None

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
        """Signal ausführen und an MetaTrader 5 übergeben."""
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
                    'source': chat_source.chat_name
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
                self.log(
                    (
                        f"Sende Order an MT5: {direction} {symbol} "
                        f"(Volumen {float(lot_size):.2f}, Preis {price:.5f})."
                    ),
                    "INFO"
                )
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
                'source': chat_source.chat_name
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
            self._mt5_account_info = None
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
        self._mt5_account_info = None
        self._last_mt5_error = None

        if self._mt5_initialized:
            try:
                mt5.shutdown()
            except Exception:
                pass
            self._mt5_initialized = False
            self._mt5_account_info = None

    def ensure_mt5_session(self) -> bool:
        """Stellt sicher, dass eine aktive MT5-Session verfügbar ist."""
        self._last_mt5_error = None

        def handle_failure(message: str) -> bool:
            self._mt5_login_ok = False
            self._mt5_account_info = None
            self._last_mt5_error = message
            self.log(message, "ERROR")
            return False

        if not MT5_AVAILABLE:
            return handle_failure("MetaTrader5-Bibliothek nicht verfügbar. Live-Modus deaktiviert.")

        if not self.mt5_login or not self.mt5_password or not self.mt5_server:
            return handle_failure("MT5-Zugangsdaten unvollständig. Bitte prüfen Sie Login, Passwort und Server.")

        initialized_this_call = False
        logged_in_this_call = False

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
            initialized_this_call = True

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
            else:
                logged_in_this_call = True

        if not account_info or account_info.login != self.mt5_login:
            return handle_failure("MT5-Account konnte nicht bestätigt werden.")

        if initialized_this_call:
            self.log("MetaTrader 5 Terminal initialisiert.", "INFO")
        if logged_in_this_call:
            self.log(f"MT5-Login erfolgreich für Konto {self.mt5_login}.", "INFO")

        summary = {
            'login': str(getattr(account_info, 'login', self.mt5_login)),
            'name': str(getattr(account_info, 'name', '') or ''),
            'server': str(getattr(account_info, 'server', self.mt5_server or '') or ''),
            'currency': str(getattr(account_info, 'currency', '') or ''),
            'balance': f"{getattr(account_info, 'balance', 0.0):.2f}",
            'equity': f"{getattr(account_info, 'equity', 0.0):.2f}",
            'leverage': str(getattr(account_info, 'leverage', '') or '')
        }

        self._mt5_account_info = summary
        self._mt5_login_ok = True
        return True

    def get_last_mt5_error(self) -> Optional[str]:
        """Gibt die letzte MT5-Fehlermeldung zurück."""
        return self._last_mt5_error

    def get_mt5_account_summary(self) -> Optional[Dict[str, str]]:
        """Liefert zusammengefasste Informationen zur aktuellen MT5-Session."""
        if not self._mt5_account_info:
            return None
        return dict(self._mt5_account_info)


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
        self.mt5_summary_label: Optional[ttk.Label] = None
        self.mt5_hint_label: Optional[ttk.Label] = None

        # GUI-Komponenten
        self.create_widgets()
        self.setup_message_processing()

        self.apply_config(self.current_config)

    def _configure_styles(self):
        """Globale Styles, Farben und Schriftarten setzen."""
        base_bg = '#eef1f7'
        surface_bg = '#ffffff'
        surface_alt = '#f7f9ff'
        accent_color = '#3f58ff'
        accent_hover = '#2c45e6'
        accent_light = '#e5e9ff'
        text_color = '#1a2233'
        subtle_text = '#626c82'
        border_color = '#d5dbeb'

        self.theme_colors: Dict[str, str] = {
            'base_bg': base_bg,
            'surface_bg': surface_bg,
            'surface_alt': surface_alt,
            'accent': accent_color,
            'accent_hover': accent_hover,
            'accent_light': accent_light,
            'text': text_color,
            'subtle_text': subtle_text,
            'border': border_color,
            'success': '#16a34a',
            'warning': '#f97316'
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
        self.style.configure('Toolbar.TFrame', background=surface_bg)
        self.style.configure('InfoBar.TFrame', background=surface_bg)
        self.style.configure('Metric.TFrame', background=surface_bg, relief='flat', borderwidth=1)
        self.style.configure('Card.TFrame', background=surface_bg, relief='flat')
        self.style.configure('Card.TLabelframe', background=surface_bg, relief='flat', borderwidth=1)
        self.style.configure('Card.TLabelframe.Label', background=surface_bg, foreground=text_color, font=('Segoe UI Semibold', 11))

        self.style.configure('Hero.TFrame', background=accent_color)
        self.style.configure('HeroTitle.TLabel', background=accent_color, foreground='#ffffff', font=('Segoe UI Semibold', 20))
        self.style.configure('HeroSubtitle.TLabel', background=accent_color, foreground='#d8deff', font=('Segoe UI', 11))
        self.style.configure('HeroTag.TLabel', background='#556cff', foreground='#ffffff', font=('Segoe UI Semibold', 10), padding=(12, 4))

        self.style.configure('Statusbar.TFrame', background=surface_bg)
        self.style.configure('Statusbar.TLabel', background=surface_bg, foreground=subtle_text, font=('Segoe UI', 10))

        self.style.configure('TNotebook', background=base_bg, borderwidth=0)
        self.style.configure('TNotebook.Tab', font=('Segoe UI Semibold', 10), padding=(18, 10))
        self.style.map(
            'TNotebook.Tab',
            background=[('selected', surface_bg), ('!selected', base_bg)],
            foreground=[('selected', text_color), ('!selected', subtle_text)]
        )

        self.style.configure('TLabel', background=base_bg, foreground=text_color, font=('Segoe UI', 10))
        self.style.configure('Title.TLabel', background=base_bg, foreground=text_color, font=('Segoe UI Semibold', 18))
        self.style.configure('Subtitle.TLabel', background=base_bg, foreground=subtle_text, font=('Segoe UI', 11))
        self.style.configure('SectionTitle.TLabel', background=base_bg, foreground=text_color, font=('Segoe UI Semibold', 14))
        self.style.configure('Info.TLabel', background=base_bg, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('InfoBar.TLabel', background=surface_bg, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('FieldLabel.TLabel', background=base_bg, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('Warning.TLabel', background=base_bg, foreground=self.theme_colors['warning'], font=('Segoe UI Semibold', 11))
        self.style.configure('MetricTitle.TLabel', background=surface_bg, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('MetricValue.TLabel', background=surface_bg, foreground=text_color, font=('Segoe UI Semibold', 16))
        self.style.configure('InfoBadge.TLabel', background=accent_light, foreground=accent_color, font=('Segoe UI Semibold', 9), padding=(10, 3))

        self.style.configure('TButton', font=('Segoe UI', 10), padding=(14, 8), relief='flat')
        self.style.map('TButton', background=[('active', surface_alt)], relief=[('pressed', 'flat')])
        self.style.configure('Accent.TButton', background=accent_color, foreground='#ffffff')
        self.style.map(
            'Accent.TButton',
            background=[('active', accent_hover), ('disabled', base_bg)],
            foreground=[('disabled', '#b8c3ff')]
        )
        self.style.configure('Toolbar.TButton', background=surface_bg, foreground=text_color, padding=(12, 8))
        self.style.map('Toolbar.TButton', background=[('active', accent_light)], foreground=[('active', accent_color)])
        self.style.configure('Link.TButton', background=base_bg, foreground=accent_color, padding=0)
        self.style.map('Link.TButton', foreground=[('active', accent_hover)])

        self.style.configure('Treeview', background=surface_bg, fieldbackground=surface_bg, foreground=text_color, font=('Segoe UI', 10), rowheight=26, borderwidth=0)
        self.style.configure(
            'Treeview.Heading',
            background=surface_bg,
            foreground=subtle_text,
            font=('Segoe UI Semibold', 10),
            padding=8,
            relief='flat'
        )
        self.style.configure('Dashboard.Treeview', rowheight=26)
        self.style.map('Treeview', background=[('selected', accent_light)], foreground=[('selected', accent_color)])
        self.style.map('Treeview.Heading', background=[('active', accent_light)])

        self.style.configure('TCheckbutton', background=base_bg, foreground=text_color, font=('Segoe UI', 10))
        self.style.configure('Switch.TCheckbutton', background=base_bg, foreground=text_color, font=('Segoe UI', 10, 'bold'))
        self.style.map('Switch.TCheckbutton', foreground=[('selected', accent_color)])

        self.style.configure('TEntry', padding=8)
        self.style.configure('TCombobox', padding=8)

    def create_widgets(self):
        """GUI-Widgets erstellen"""

        # Main Container
        self.main_frame = ttk.Frame(self.root, padding=(24, 24, 24, 18), style='Main.TFrame')
        self.main_frame.pack(fill='both', expand=True)

        # Header
        header_frame = ttk.Frame(self.main_frame, style='Hero.TFrame', padding=(28, 26))
        header_frame.pack(fill='x', pady=(0, 20))

        ttk.Label(
            header_frame,
            text="📊 Multi-Chat Trading Cockpit",
            style='HeroTitle.TLabel'
        ).pack(anchor='w')

        ttk.Label(
            header_frame,
            text="Synchronisiere Signale & verwalte Quellen in Echtzeit",
            style='HeroSubtitle.TLabel'
        ).pack(anchor='w', pady=(6, 0))

        tag_frame = ttk.Frame(header_frame, style='Hero.TFrame')
        tag_frame.pack(anchor='w', pady=(18, 0))
        for tag_text in ("Live-Überwachung", "Mehrere Quellen", "Echtzeit-Sync"):
            ttk.Label(tag_frame, text=tag_text, style='HeroTag.TLabel').pack(side='left', padx=(0, 12))

        ttk.Separator(self.main_frame).pack(fill='x', pady=(0, 16))

        # Notebook-Tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill='both', expand=True)

        self.create_chat_overview_tab()
        self.create_bot_settings_tab()
        self.create_trading_settings_tab()
        self.create_statistics_tab()

        # Status Bar
        self.status_frame = ttk.Frame(self.main_frame, style='Statusbar.TFrame', padding=(18, 12))
        self.status_frame.pack(fill='x', pady=(12, 0))

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
        self.start_button.pack(side='left', padx=(0, 8))
        ttk.Button(button_frame, text="■ Bot stoppen", command=self.stop_bot).pack(side='left')

    def create_chat_overview_tab(self):
        """Tab für die Chat-Übersicht"""
        chat_frame = ttk.Frame(self.notebook, padding=(24, 24, 24, 20), style='Main.TFrame')
        self.notebook.add(chat_frame, text="Chats")

        header = ttk.Frame(chat_frame, style='Main.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text="Meine Chats", style='SectionTitle.TLabel').pack(side='left')
        self.chat_summary_label = ttk.Label(header, text="Keine Chats geladen", style='Info.TLabel')
        self.chat_summary_label.pack(side='right')

        badge_row = ttk.Frame(chat_frame, style='Main.TFrame')
        badge_row.pack(fill='x', pady=(8, 4))
        ttk.Label(badge_row, text="Automatische Synchronisierung aktiv", style='InfoBadge.TLabel').pack(side='left')

        controls_frame = ttk.Frame(chat_frame, style='Toolbar.TFrame', padding=(16, 12))
        controls_frame.pack(fill='x', pady=(18, 14))
        controls_frame.columnconfigure((0, 1, 2, 3), weight=1)

        ttk.Button(
            controls_frame,
            text="🔄 Chats laden",
            command=self.load_chats,
            style='Toolbar.TButton'
        ).grid(row=0, column=0, sticky='w', padx=(0, 12))

        ttk.Button(
            controls_frame,
            text="✅ Überwachung aktivieren",
            command=self.enable_monitoring,
            style='Toolbar.TButton'
        ).grid(row=0, column=1, sticky='w', padx=(0, 12))

        ttk.Button(
            controls_frame,
            text="⏸ Überwachung deaktivieren",
            command=self.disable_monitoring,
            style='Toolbar.TButton'
        ).grid(row=0, column=2, sticky='w', padx=(0, 12))

        ttk.Button(
            controls_frame,
            text="💾 Konfiguration sichern",
            command=self.export_chat_config,
            style='Toolbar.TButton'
        ).grid(row=0, column=3, sticky='w')

        list_frame = ttk.LabelFrame(chat_frame, text="Verfügbare Chats", padding="16", style='Card.TLabelframe')
        list_frame.pack(fill='both', expand=True)

        columns = ('Name', 'ID', 'Typ', 'Teilnehmer', 'Überwacht', 'Signale')
        self.chats_tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show='headings',
            height=18,
            style='Dashboard.Treeview'
        )
        self.chats_tree.pack(side='left', fill='both', expand=True)

        chat_scroll = ttk.Scrollbar(list_frame, orient='vertical', command=self.chats_tree.yview)
        chat_scroll.pack(side='right', fill='y')
        self.chats_tree.configure(yscrollcommand=chat_scroll.set)

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
            'ID': 140,
            'Typ': 110,
            'Teilnehmer': 130,
            'Überwacht': 140,
            'Signale': 140
        }

        for col in columns:
            self.chats_tree.heading(col, text=heading_texts.get(col, col))
            self.chats_tree.column(col, width=column_widths.get(col, 120), anchor='w')

        info_frame = ttk.Frame(chat_frame, style='InfoBar.TFrame', padding=(14, 12))
        info_frame.pack(fill='x', pady=(14, 0))
        ttk.Label(info_frame, text="Hinweis: Aktivierte Chats werden kontinuierlich synchronisiert.", style='InfoBar.TLabel').pack(side='left')
        ttk.Button(
            info_frame,
            text="ℹ Hilfe",
            style='Link.TButton',
            command=lambda: messagebox.showinfo(
                "Information",
                "Markieren Sie Chats und nutzen Sie die Toolbar, um die Überwachung anzupassen."
            )
        ).pack(side='right')

    def create_trading_settings_tab(self):
        """Tab für Handelsparameter und Log"""
        settings_tab = ttk.Frame(self.notebook, padding=(24, 24, 24, 20), style='Main.TFrame')
        self.notebook.add(settings_tab, text="Trading Einstellungen")

        header = ttk.Frame(settings_tab, style='Main.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text="Trading-Konfiguration", style='SectionTitle.TLabel').pack(side='left')

        settings_frame = ttk.Frame(settings_tab, style='Card.TFrame', padding=(20, 18))
        settings_frame.pack(fill='x', pady=(20, 16))
        settings_frame.columnconfigure((0, 1), weight=1)

        ttk.Label(settings_frame, text="Ausführungsmodus:", style='FieldLabel.TLabel').grid(row=0, column=0, sticky='w')
        default_execution_label = self.execution_mode_labels.get(ExecutionMode.INSTANT, "Sofortausführung")
        self.execution_mode_var = tk.StringVar(value=default_execution_label)
        self.execution_mode_combobox = ttk.Combobox(
            settings_frame,
            textvariable=self.execution_mode_var,
            values=list(self.execution_mode_labels.values()),
            state='readonly'
        )
        self.execution_mode_combobox.grid(row=0, column=1, sticky='ew', padx=(8, 0))
        self.execution_mode_combobox.bind('<<ComboboxSelected>>', self.on_execution_mode_change)

        warning_label = ttk.Label(
            settings_frame,
            text="⚠ WARNUNG: Automatisiertes Trading birgt hohe Verlustrisiken!",
            style='Warning.TLabel'
        )
        warning_label.grid(row=1, column=0, columnspan=2, sticky='w', pady=(14, 0))

        signal_settings_frame = ttk.Frame(settings_tab, style='Card.TFrame', padding=(20, 18))
        signal_settings_frame.pack(fill='x', pady=(0, 18))
        signal_settings_frame.columnconfigure((0, 1), weight=1)

        ttk.Label(
            signal_settings_frame,
            text="Signal-Optionen:",
            style='FieldLabel.TLabel'
        ).grid(row=0, column=0, columnspan=2, sticky='w')

        ttk.Checkbutton(
            signal_settings_frame,
            text="Sofort-Trading aktiv",
            variable=self.instant_trading_var,
            command=lambda key='instant_trading_enabled': self._handle_signal_flag_change(key)
        ).grid(row=1, column=0, sticky='w', pady=(10, 4))

        ttk.Checkbutton(
            signal_settings_frame,
            text="Zonen-Trading aktiv",
            variable=self.zone_trading_var,
            command=lambda key='zone_trading_enabled': self._handle_signal_flag_change(key)
        ).grid(row=1, column=1, sticky='w', pady=(10, 4), padx=(12, 0))

        ttk.Checkbutton(
            signal_settings_frame,
            text="Bestätigung vor Ausführung",
            variable=self.require_confirmation_var,
            command=lambda key='require_confirmation': self._handle_signal_flag_change(key)
        ).grid(row=2, column=0, sticky='w', pady=(4, 0))

        ttk.Checkbutton(
            signal_settings_frame,
            text="SL/TP automatisch erkennen",
            variable=self.auto_tp_sl_var,
            command=lambda key='auto_tp_sl': self._handle_signal_flag_change(key)
        ).grid(row=2, column=1, sticky='w', pady=(4, 0), padx=(12, 0))

        numeric_frame = ttk.Frame(settings_tab, style='Card.TFrame', padding=(20, 18))
        numeric_frame.pack(fill='x', pady=(0, 18))
        for col_index in (1, 3):
            numeric_frame.columnconfigure(col_index, weight=1)

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
            row = idx // 2
            column = (idx % 2) * 2
            ttk.Label(
                numeric_frame,
                text=label_text,
                style='FieldLabel.TLabel'
            ).grid(row=row, column=column, sticky='w')

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
            spinbox = ttk.Spinbox(numeric_frame, **spinbox_kwargs)
            spinbox.grid(row=row, column=column + 1, sticky='w', padx=(10, 0), pady=(0, 6))
            numeric_frame.grid_rowconfigure(row, pad=6)

        toolbar = ttk.Frame(settings_tab, style='Toolbar.TFrame', padding=(16, 12))
        toolbar.pack(fill='x', pady=(0, 18))
        ttk.Button(toolbar, text="📥 Signale abrufen", style='Toolbar.TButton', command=self.load_chats).pack(side='left')
        ttk.Button(toolbar, text="🧹 Log leeren", style='Toolbar.TButton', command=self.clear_log).pack(side='left', padx=(10, 0))
        ttk.Button(toolbar, text="📊 Statistiken aktualisieren", style='Toolbar.TButton', command=self.refresh_statistics).pack(side='left', padx=(10, 0))

        metrics_frame = ttk.Frame(settings_tab, style='Main.TFrame')
        metrics_frame.pack(fill='x', pady=(0, 18))
        metrics_frame.columnconfigure((0, 1, 2), weight=1)
        for idx, (title, value) in enumerate([
            ("Aktive Chats", "0"),
            ("Überwachte Signale", "0"),
            ("Heute synchronisiert", "0")
        ]):
            metric = ttk.Frame(metrics_frame, style='Metric.TFrame', padding=(16, 12))
            metric.grid(row=0, column=idx, padx=(0 if idx == 0 else 12, 0), sticky='nsew')
            ttk.Label(metric, text=title, style='MetricTitle.TLabel').pack(anchor='w')
            ttk.Label(metric, text=value, style='MetricValue.TLabel').pack(anchor='w', pady=(4, 0))

        log_frame = ttk.LabelFrame(settings_tab, text="Live-Aktivitätsprotokoll", padding="16", style='Card.TLabelframe')
        log_frame.pack(fill='both', expand=True)

        self.log_text = tk.Text(
            log_frame,
            height=16,
            wrap='word',
            bg=self.theme_colors['surface_bg'],
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

        log_scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        log_scroll.pack(side='right', fill='y')
        self.log_text.configure(yscrollcommand=log_scroll.set, padx=14, pady=12, spacing3=6)

    def create_bot_settings_tab(self):
        """Tab für MT5-Anbindung und Zugangsdaten."""
        mt5_tab = ttk.Frame(self.notebook, padding=(24, 24, 24, 20), style='Main.TFrame')
        self.notebook.add(mt5_tab, text="Bot Einstellungen")

        header = ttk.Frame(mt5_tab, style='Main.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text="MetaTrader 5 Anbindung", style='SectionTitle.TLabel').pack(side='left')
        self.mt5_status_label = ttk.Label(
            header,
            text="",
            style='Info.TLabel'
        )
        self.mt5_status_label.pack(side='right')

        intro_card = ttk.Frame(mt5_tab, style='Card.TFrame', padding=(20, 18))
        intro_card.pack(fill='x', pady=(20, 16))
        ttk.Label(
            intro_card,
            text=(
                "Hinterlegen Sie hier die Zugangsdaten Ihres MetaTrader-5-Kontos, damit der Bot Trades "
                "direkt im Terminal platzieren kann."
            ),
            style='Info.TLabel',
            wraplength=780,
            justify='left'
        ).pack(anchor='w')

        connection_card = ttk.Frame(mt5_tab, style='Card.TFrame', padding=(20, 18))
        connection_card.pack(fill='x', pady=(0, 18))
        connection_card.columnconfigure(1, weight=1)

        ttk.Label(connection_card, text="Login (Kontonummer):", style='FieldLabel.TLabel').grid(
            row=0, column=0, sticky='w'
        )
        ttk.Entry(
            connection_card,
            textvariable=self.mt5_login_var,
        ).grid(row=0, column=1, sticky='ew', padx=(10, 0), pady=(0, 6))

        ttk.Label(connection_card, text="Passwort:", style='FieldLabel.TLabel').grid(
            row=1, column=0, sticky='w'
        )
        ttk.Entry(
            connection_card,
            textvariable=self.mt5_password_var,
            show='•',
        ).grid(row=1, column=1, sticky='ew', padx=(10, 0), pady=(0, 6))

        ttk.Label(connection_card, text="Server:", style='FieldLabel.TLabel').grid(
            row=2, column=0, sticky='w'
        )
        ttk.Entry(
            connection_card,
            textvariable=self.mt5_server_var,
        ).grid(row=2, column=1, sticky='ew', padx=(10, 0), pady=(0, 6))

        ttk.Label(connection_card, text="MT5-Terminal (optional):", style='FieldLabel.TLabel').grid(
            row=3, column=0, sticky='w'
        )
        ttk.Entry(
            connection_card,
            textvariable=self.mt5_path_var,
        ).grid(row=3, column=1, sticky='ew', padx=(10, 0), pady=(0, 6))
        ttk.Button(
            connection_card,
            text="Durchsuchen…",
            command=self.browse_mt5_path
        ).grid(row=3, column=2, sticky='w', padx=(8, 0), pady=(0, 6))

        button_row = ttk.Frame(connection_card, style='Card.TFrame')
        button_row.grid(row=4, column=0, columnspan=3, sticky='w', pady=(14, 0))
        ttk.Button(
            button_row,
            text="Zugangsdaten speichern",
            command=self.save_mt5_credentials
        ).pack(side='left')
        ttk.Button(
            button_row,
            text="Verbindung testen",
            command=self.test_mt5_connection
        ).pack(side='left', padx=(12, 0))

        self.mt5_hint_label = ttk.Label(
            connection_card,
            text="",
            style='Info.TLabel',
            wraplength=780,
            justify='left'
        )
        self.mt5_hint_label.grid(row=5, column=0, columnspan=3, sticky='w', pady=(12, 0))

        status_card = ttk.Frame(mt5_tab, style='Card.TFrame', padding=(20, 18))
        status_card.pack(fill='x', pady=(0, 18))
        ttk.Label(
            status_card,
            text="Verbindungsstatus",
            style='FieldLabel.TLabel'
        ).pack(anchor='w')
        self.mt5_summary_label = ttk.Label(
            status_card,
            text="",
            style='Info.TLabel',
            wraplength=780,
            justify='left'
        )
        self.mt5_summary_label.pack(anchor='w', pady=(8, 0))

        self._refresh_mt5_status_display()

        checklist_card = ttk.Frame(mt5_tab, style='Card.TFrame', padding=(20, 18))
        checklist_card.pack(fill='x')
        ttk.Label(
            checklist_card,
            text="Checkliste vor dem Live-Betrieb:",
            style='FieldLabel.TLabel'
        ).pack(anchor='w')
        checklist = ttk.Frame(checklist_card, style='Card.TFrame')
        checklist.pack(fill='x', pady=(10, 0))
        for idx, hint in enumerate([
            "MetaTrader 5 ist installiert und das Python-Paket `MetaTrader5` ist verfügbar.",
            "Konto-Login, Passwort und Server stammen aus dem gewünschten MT5-Konto.",
            "Das Terminal (`terminal64.exe`) ist optional verknüpft, falls mehrere Installationen existieren.",
            "Die Verbindung wurde erfolgreich getestet."
        ]):
            ttk.Label(
                checklist,
                text=f"• {hint}",
                style='Info.TLabel',
                wraplength=780,
                justify='left'
            ).grid(row=idx, column=0, sticky='w', pady=(0 if idx == 0 else 4, 0))

    def create_statistics_tab(self):
        """Tab für Statistiken des Kopierers"""
        stats_frame = ttk.Frame(self.notebook, padding=(24, 24, 24, 20), style='Main.TFrame')
        self.notebook.add(stats_frame, text="Kopierer Statistiken")

        header = ttk.Frame(stats_frame, style='Main.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text="Performance des Kopierers", style='SectionTitle.TLabel').pack(side='left')
        self.statistics_hint = ttk.Label(header, text="Letzte Aktualisierung: –", style='Info.TLabel')
        self.statistics_hint.pack(side='right')

        kpi_frame = ttk.Frame(stats_frame, style='Main.TFrame')
        kpi_frame.pack(fill='x', pady=(18, 18))
        kpi_frame.columnconfigure((0, 1, 2), weight=1)
        for idx, (title, value) in enumerate([
            ("Gesamtgewinn", "0.00"),
            ("Trefferquote", "0.0%"),
            ("Signale gesamt", "0")
        ]):
            card = ttk.Frame(kpi_frame, style='Metric.TFrame', padding=(16, 12))
            card.grid(row=0, column=idx, padx=(0 if idx == 0 else 12, 0), sticky='nsew')
            ttk.Label(card, text=title, style='MetricTitle.TLabel').pack(anchor='w')
            ttk.Label(card, text=value, style='MetricValue.TLabel').pack(anchor='w', pady=(4, 0))

        source_frame = ttk.LabelFrame(stats_frame, text="Statistiken nach Quelle", padding="16", style='Card.TLabelframe')
        source_frame.pack(fill='both', expand=True)

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

        self.stats_tree = ttk.Treeview(
            source_frame,
            columns=stats_columns,
            show='headings',
            height=12,
            style='Dashboard.Treeview'
        )
        self.stats_tree.pack(fill='both', expand=True)

        for col in stats_columns:
            self.stats_tree.heading(col, text=heading_texts.get(col, col))
            self.stats_tree.column(col, width=column_widths.get(col, 140), anchor='w')

        actions_frame = ttk.Frame(stats_frame, style='Toolbar.TFrame', padding=(12, 10))
        actions_frame.pack(fill='x', pady=(16, 0))
        ttk.Button(actions_frame, text="🔁 Aktualisieren", style='Toolbar.TButton', command=self.refresh_statistics).pack(side='left')
        ttk.Button(actions_frame, text="📤 Export", style='Toolbar.TButton', command=self.export_statistics).pack(side='left', padx=(10, 0))

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

        self.status_label.config(text=f"Chats geladen: {len(chats_data)}")
        if hasattr(self, 'chat_summary_label'):
            active_count = 0
            for chat in chats_data:
                info = self.bot.chat_manager.get_chat_info(chat['id'])
                if info and info.enabled:
                    active_count += 1
            self.chat_summary_label.config(
                text=f"Geladene Quellen: {len(chats_data)} | Aktiv: {active_count}"
            )

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

        if hasattr(self, 'statistics_hint'):
            self.statistics_hint.config(
                text=f"Letzte Aktualisierung: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
            )

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
                    elif msg_type == 'CONFIRM_TRADE':
                        self._handle_trade_confirmation_request(data)

            except queue.Empty:
                pass

            self.root.after(100, process_messages)

        process_messages()

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

    def _format_mt5_summary(self, summary: Dict[str, str]) -> str:
        """Formatiert die MT5-Kontoinformationen für Anzeigezwecke."""
        lines: List[str] = []

        login = summary.get('login')
        if login:
            line = f"Konto: {login}"
            name = summary.get('name')
            if name:
                line = f"{line} – {name}"
            lines.append(line)

        server = summary.get('server')
        if server:
            lines.append(f"Server: {server}")

        balance = summary.get('balance')
        equity = summary.get('equity')
        currency = summary.get('currency')
        if balance:
            balance_text = f"Balance: {balance}"
            if currency:
                balance_text = f"{balance_text} {currency}"
            lines.append(balance_text)
        if equity:
            equity_text = f"Equity: {equity}"
            if currency:
                equity_text = f"{equity_text} {currency}"
            lines.append(equity_text)

        leverage = summary.get('leverage')
        if leverage:
            lines.append(f"Hebel: {leverage}")

        return '\n'.join(lines)

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

    def _apply_mt5_status(
        self,
        status_text: str,
        summary_text: Optional[str] = None,
        *,
        warning: bool = False,
        hint_text: Optional[str] = None,
        hint_warning: Optional[bool] = None
    ):
        """Aktualisiert Status-, Hinweis- und Zusammenfassungsanzeigen für MT5."""
        label_style = 'Warning.TLabel' if warning else 'Info.TLabel'
        if getattr(self, 'mt5_status_label', None):
            try:
                self.mt5_status_label.config(text=status_text, style=label_style)
            except tk.TclError:
                self.mt5_status_label.config(text=status_text)

        summary_value = summary_text if summary_text is not None else ""
        if getattr(self, 'mt5_summary_label', None):
            try:
                self.mt5_summary_label.config(text=summary_value, style=label_style)
            except tk.TclError:
                self.mt5_summary_label.config(text=summary_value)

        if getattr(self, 'mt5_hint_label', None) and hint_text is not None:
            hint_style = 'Warning.TLabel' if (hint_warning if hint_warning is not None else warning) else 'Info.TLabel'
            try:
                self.mt5_hint_label.config(text=hint_text, style=hint_style)
            except tk.TclError:
                self.mt5_hint_label.config(text=hint_text)

    def _refresh_mt5_status_display(self):
        """Setzt den MT5-Status basierend auf den aktuellen Formularwerten."""
        if not getattr(self, 'mt5_status_label', None):
            return

        if not MT5_AVAILABLE:
            summary = (
                "Das MetaTrader5-Python-Modul wurde nicht gefunden. "
                "Installieren Sie MetaTrader 5 inklusive des Python-Pakets, um Orders zu platzieren. "
                "Sie können die Zugangsdaten trotzdem eintragen; die Verbindung wird aktiv, sobald das Modul verfügbar ist."
            )
            hint = "Installieren Sie das Paket z.B. über 'pip install MetaTrader5'."
            self._apply_mt5_status(
                "MT5-Modul nicht installiert",
                summary,
                warning=True,
                hint_text=hint,
                hint_warning=True
            )
            return

        login, password, server, _ = self._collect_mt5_form_data()

        if not login and not password and not server:
            summary = (
                "Keine Zugangsdaten hinterlegt. Tragen Sie Login, Passwort und Server ein und speichern Sie die Daten."
            )
            hint = "Nach dem Speichern können Sie die Verbindung über 'Verbindung testen' prüfen."
            self._apply_mt5_status(
                "Keine Zugangsdaten hinterlegt",
                summary,
                hint_text=hint
            )
            return

        if not login or not password or not server:
            summary = (
                "Bitte füllen Sie Login, Passwort und Server vollständig aus, damit die MT5-Verbindung aufgebaut werden kann."
            )
            hint = "Speichern Sie die Daten nach dem Ausfüllen und testen Sie anschließend die Verbindung."
            self._apply_mt5_status(
                "Angaben unvollständig",
                summary,
                warning=True,
                hint_text=hint,
                hint_warning=True
            )
            return

        summary = (
            f"Zugangsdaten geladen für Konto {login}. Testen Sie die Verbindung, bevor Sie den Bot starten."
        )
        hint = "Mit 'Verbindung testen' überprüfen Sie Terminal und Zugangsdaten unmittelbar."
        self._apply_mt5_status(
            "Zugangsdaten geladen",
            summary,
            hint_text=hint
        )

    def save_mt5_credentials(self, silent: bool = False) -> bool:
        """Speichert MT5-Zugangsdaten in der Konfiguration."""
        login, password, server, path = self._collect_mt5_form_data()

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
            self.log_message(f"Fehler beim Speichern der MT5-Daten: {exc}")
            return False

        if not silent:
            self.log_message("MT5-Zugangsdaten aktualisiert.")
            if MT5_AVAILABLE:
                if login and password and server:
                    summary = (
                        f"Zugangsdaten gespeichert für Konto {login}. "
                        "Führen Sie 'Verbindung testen' aus, um die Anbindung zu prüfen."
                    )
                    hint = "Mit 'Verbindung testen' überprüfen Sie Terminal und Zugangsdaten unmittelbar."
                    self._apply_mt5_status(
                        "Zugangsdaten gespeichert",
                        summary,
                        hint_text=hint
                    )
                else:
                    summary = "Bitte füllen Sie Login, Passwort und Server vollständig aus."
                    hint = "Ohne vollständige Angaben kann keine MT5-Verbindung hergestellt werden."
                    self._apply_mt5_status(
                        "Angaben unvollständig",
                        summary,
                        warning=True,
                        hint_text=hint,
                        hint_warning=True
                    )
            else:
                summary = (
                    "Das MetaTrader5-Python-Modul wurde nicht gefunden. Installieren Sie es, um Orders zu platzieren."
                )
                hint = "Installiere MetaTrader 5 inklusive des Python-Pakets 'MetaTrader5'."
                self._apply_mt5_status(
                    "MT5-Modul nicht installiert",
                    summary,
                    warning=True,
                    hint_text=hint,
                    hint_warning=True
                )
        return True

    def test_mt5_connection(self):
        """Testet die Verbindung zu MetaTrader 5."""
        if not MT5_AVAILABLE:
            message = "MetaTrader5-Python-Modul ist nicht verfügbar. Installieren Sie es, um den LIVE-Modus zu nutzen."
            self.log_message(message)
            self._apply_mt5_status(
                "MT5-Modul nicht installiert",
                message,
                warning=True,
                hint_text="Installieren Sie MetaTrader 5 inklusive des Python-Pakets 'MetaTrader5'.",
                hint_warning=True
            )
            try:
                messagebox.showwarning("MT5 nicht verfügbar", message)
            except Exception:
                pass
            return

        login, password, server, path = self._collect_mt5_form_data()

        if not login or not password or not server:
            message = "Bitte geben Sie Login, Passwort und Server an, bevor Sie die Verbindung testen."
            self.log_message(message)
            self._apply_mt5_status(
                "Angaben unvollständig",
                message,
                warning=True,
                hint_text="Bitte füllen Sie alle Felder aus und speichern Sie die Daten vor dem Test.",
                hint_warning=True
            )
            try:
                messagebox.showwarning("Angaben unvollständig", message)
            except Exception:
                pass
            return

        if not self.save_mt5_credentials(silent=True):
            return

        self.log_message("Teste MT5-Verbindung ...")
        success = self.bot.ensure_mt5_session()
        if success:
            message = "MT5-Verbindung erfolgreich aufgebaut."
            self.log_message(message)
            summary = self.bot.get_mt5_account_summary()
            summary_text = ""
            if summary:
                summary_text = self._format_mt5_summary(summary) or ""
                if summary_text:
                    self.log_message(f"MT5-Kontoübersicht:\n{summary_text}")
            display_text = message
            if summary_text:
                display_text = f"{message}\n\n{summary_text}"
            self._apply_mt5_status(
                "Verbindung aktiv",
                display_text,
                hint_text="Die Verbindung ist aktiv. Der Bot kann Trades direkt an MetaTrader 5 senden."
            )
            try:
                messagebox.showinfo("Verbindung erfolgreich", display_text)
            except Exception:
                pass
        else:
            message = self.bot.get_last_mt5_error() or "MT5-Verbindung konnte nicht hergestellt werden."
            self.log_message(message)
            self._apply_mt5_status(
                "Verbindung fehlgeschlagen",
                message,
                warning=True,
                hint_text="Überprüfen Sie Zugangsdaten, Server und das laufende MT5-Terminal.",
                hint_warning=True
            )
            try:
                messagebox.showerror("Verbindung fehlgeschlagen", message)
            except Exception:
                pass

    def browse_mt5_path(self):
        """Dateidialog zum Auswählen des MT5-Terminals öffnen."""
        module_missing = not MT5_AVAILABLE
        if module_missing:
            message = (
                "MetaTrader5-Python-Modul ist nicht verfügbar. "
                "Sie können den Terminalpfad trotzdem auswählen und speichern; "
                "installieren Sie das Paket anschließend, um die Verbindung zu aktivieren."
            )
            self.log_message(message)
            try:
                messagebox.showwarning("MT5-Modul nicht installiert", message)
            except Exception:
                pass

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

        self.save_mt5_credentials(silent=True)
        self._refresh_mt5_status_display()
        if module_missing:
            self.log_message(
                "MT5-Terminalpfad gespeichert. Installieren Sie das MetaTrader5-Python-Modul, "
                "um die Verbindung testen zu können."
            )
        else:
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
        self.window = tk.Toplevel(self.parent)
        self.window.title("Erste Einrichtung")
        self.window.geometry("600x500")
        self.window.grab_set()  # Modal

        header_frame = ttk.Frame(self.window)
        header_frame.pack(fill='x', padx=20, pady=20)

        ttk.Label(header_frame, text="Multi-Chat Trading Bot Setup",
                  font=('Arial', 16, 'bold')).pack()

        ttk.Label(header_frame,
                  text="Willkommen! Bitte konfigurieren Sie Ihre Telegram-Verbindung.",
                  wraplength=500).pack(pady=(10, 0))

        warning_frame = ttk.LabelFrame(self.window, text="WICHTIGE WARNUNG", padding="15")
        warning_frame.pack(fill='x', padx=20, pady=(0, 20))

        warning_text = (
            "ACHTUNG: Dieses System führt automatische Trades aus!\n\n"
            "• Testen Sie alle Funktionen gründlich mit einem MT5-Demokonto\n"
            "• Überprüfen Sie jede Strategie ausgiebig\n"
            "• Automatisiertes Trading birgt hohe Verlustrisiken\n"
            "• Überwachen Sie das System kontinuierlich\n"
            "• Setzen Sie strikte Risikogrenzen\n\n"
            "Der Autor übernimmt keine Haftung für finanzielle Verluste!"
        )
        ttk.Label(warning_frame, text=warning_text, foreground='red',
                  wraplength=500).pack()

        form_frame = ttk.LabelFrame(self.window, text="Telegram API Konfiguration", padding="15")
        form_frame.pack(fill='x', padx=20, pady=(0, 20))

        instructions = (
            "1. Gehen Sie zu https://my.telegram.org/auth\n"
            "2. Loggen Sie sich ein und erstellen Sie eine neue App\n"
            "3. Kopieren Sie API ID und API Hash hierher"
        )
        ttk.Label(form_frame, text=instructions, wraplength=500).pack(pady=(0, 15))

        telegram_cfg = self.current_config.get('telegram', {})

        ttk.Label(form_frame, text="API ID:").pack(anchor='w')
        self.setup_api_id = tk.StringVar(value=str(telegram_cfg.get('api_id', "")))
        ttk.Entry(form_frame, textvariable=self.setup_api_id, width=40).pack(fill='x', pady=(0, 10))

        ttk.Label(form_frame, text="API Hash:").pack(anchor='w')
        self.setup_api_hash = tk.StringVar(value=str(telegram_cfg.get('api_hash', "")))
        ttk.Entry(form_frame, textvariable=self.setup_api_hash, width=40).pack(fill='x', pady=(0, 10))

        ttk.Label(form_frame, text="Telefonnummer (mit Ländercode, z.B. +49...):").pack(anchor='w')
        self.setup_phone = tk.StringVar(value=str(telegram_cfg.get('phone', "")))
        ttk.Entry(form_frame, textvariable=self.setup_phone, width=40).pack(fill='x', pady=(0, 10))

        self.prompt_credentials = tk.BooleanVar(
            value=bool(telegram_cfg.get('prompt_credentials_on_start', False))
        )
        ttk.Checkbutton(
            form_frame,
            text="API-Zugangsdaten bei jedem Start erneut abfragen",
            variable=self.prompt_credentials
        ).pack(anchor='w', pady=(5, 0))

        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill='x', padx=20, pady=20)

        ttk.Button(button_frame, text="Abbrechen", command=self.cancel_setup).pack(side='right', padx=(10, 0))
        ttk.Button(button_frame, text="Konfiguration speichern", command=self.test_config).pack(side='right')

    def test_config(self):
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

    if telegram_cfg.get('prompt_credentials_on_start'):
        return True

    if not all([
        telegram_cfg.get('api_id'),
        telegram_cfg.get('api_hash'),
        telegram_cfg.get('phone')
    ]):
        return True
    return False


def show_startup_warning() -> bool:
    """Startup-Warnung anzeigen"""
    root = tk.Tk()
    root.withdraw()

    warning_text = (
        "WARNUNG: AUTOMATISIERTES TRADING SYSTEM\n\n"
        "Dieses System kann automatisch Trades ausführen!\n\n"
        "• Hohe finanzielle Verlustrisiken\n"
        "• Nur für erfahrene Trader geeignet\n"
        "• Umfangreiche Tests mit einem MT5-Demokonto erforderlich\n\n"
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

def main():
    """Hauptfunktion mit Setup-Assistent"""
    if not show_startup_warning():
        print("Programm abgebrochen.")
        return

    try:
        config_manager = ConfigManager()

        if check_first_run():
            root = tk.Tk()
            root.withdraw()
            setup = SetupAssistant(root)
            setup.show_setup_dialog()
            root.mainloop()
            root.destroy()

            if not setup.config_saved:
                print("Setup abgebrochen. Anwendung wird beendet.")
                return

        # Konfiguration laden und auf Bot anwenden
        if not prompt_for_api_credentials(config_manager):
            print("Keine API-Zugangsdaten eingegeben. Anwendung wird beendet.")
            return

        cfg = config_manager.load_config()
        app = TradingGUI(cfg)
        app.run()

    except Exception as e:
        print(f"Fehler beim Starten der Anwendung: {e}")
        input("Drücken Sie Enter zum Beenden...")


if __name__ == "__main__":
    main()
