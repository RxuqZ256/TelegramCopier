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
from telethon.errors import (
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    SessionPasswordNeededError
)
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

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

        # Zugangsdaten anwenden (falls vorhanden)
        self.update_credentials(api_id, api_hash, phone, session_name=session_name)

        # Message Queue für GUI
        self.message_queue: "queue.Queue" = queue.Queue()

        # Status
        self.is_running = False
        self.demo_mode = True  # Immer mit Demo starten!
        self.pending_trade_updates: Dict[int, Dict] = {}

    def update_credentials(self, api_id: Optional[str], api_hash: Optional[str], phone: Optional[str],
                           session_name: Optional[str] = None):
        """Telegram Zugangsdaten aktualisieren und Client neu initialisieren"""

        if session_name:
            self.session_name = session_name

        # Vorherige Client-Instanz verwerfen
        if self.client:
            self.client = None

        try:
            self.api_id = int(api_id) if api_id else 0
        except (TypeError, ValueError):
            self.api_id = 0

        self.api_hash = api_hash or ""
        self.phone = phone or ""

        if self.api_id and self.api_hash:
            # Nur erstellen, wenn gültige Daten vorhanden sind
            self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)

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
        if not self.client:
            self.log("Telegram-Konfiguration fehlt. Chats können nicht geladen werden.", "ERROR")
            return chats_data

        try:
            authorized = await self.ensure_authorized(
                request_code=False,
                notify_gui=True,
                message=(
                    "Telegram-Login erforderlich. Chats können erst nach Eingabe des Login-Codes geladen werden."
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
        self.root.minsize(1100, 720)

        # Style & Theme
        self.theme_variant = 'modern_light'
        self.theme_var = tk.StringVar(value='Modern Light')
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use('clam')
        except tk.TclError:
            pass
        self._configure_styles(self.theme_variant)

        # Bot-Instanz (setzt später Config/Setup)
        self.bot = MultiChatTradingBot(None, None, None)
        self.bot_starting = False
        self._auth_dialog_open = False
        self._last_auth_message: Optional[str] = None
        self._pending_auth_message: Optional[str] = None

        # Buttons (werden in create_widgets gesetzt)
        self.start_button: Optional[ttk.Button] = None
        self.theme_selector: Optional[ttk.Combobox] = None

        # GUI-Komponenten
        self.create_widgets()
        self.setup_message_processing()

        self._apply_palette_to_widgets()

        if config:
            self.apply_config(config)

    def _configure_styles(self, variant: str = 'modern_light'):
        """Globale Styles, Farben und Schriftarten setzen."""
        self.palette = self._build_palette(variant)

        base_bg = self.palette['base']
        surface_bg = self.palette['surface']
        surface_alt = self.palette['surface_alt']
        accent_color = self.palette['accent']
        accent_hover = self.palette['accent_hover']
        accent_soft = self.palette['accent_soft']
        accent_text = self.palette['accent_text']
        text_color = self.palette['text']
        subtle_text = self.palette['subtle_text']
        border_color = self.palette['border']
        danger_color = self.palette['danger']
        on_accent = self.palette['on_accent']

        self.root.configure(bg=base_bg)
        self.root.option_add('*Font', 'Segoe UI 10')
        self.root.option_add('*TCombobox*Listbox.font', 'Segoe UI 10')
        self.root.option_add('*TCombobox*Listbox.background', surface_bg)
        self.root.option_add('*TCombobox*Listbox.foreground', text_color)
        self.root.option_add('*TCombobox*Listbox.selectBackground', accent_color)
        self.root.option_add('*TCombobox*Listbox.selectForeground', on_accent)

        self.style.configure('TFrame', background=base_bg)
        self.style.configure('Surface.TFrame', background=surface_bg)
        self.style.configure('Header.TFrame', background=base_bg, padding=(0, 6, 0, 6))
        self.style.configure('Toolbar.TFrame', background=surface_alt, padding=(12, 10))
        self.style.configure('Status.TFrame', background=surface_alt, padding=(12, 10))
        self.style.configure('Metric.TFrame', background=surface_bg, relief='flat')
        self.style.configure('Card.TLabelframe', background=surface_bg, bordercolor=border_color, borderwidth=1, relief='solid')
        self.style.configure(
            'Card.TLabelframe.Label',
            background=surface_bg,
            foreground=subtle_text,
            font=('Segoe UI Semibold', 11)
        )

        self.style.configure('TNotebook', background=base_bg, borderwidth=0, tabmargins=(10, 6, 10, 0))
        self.style.configure(
            'TNotebook.Tab',
            font=('Segoe UI Semibold', 11),
            padding=(20, 10),
            background=surface_alt,
            foreground=subtle_text,
            borderwidth=0
        )
        self.style.map(
            'TNotebook.Tab',
            background=[('selected', surface_bg), ('active', surface_bg)],
            foreground=[('selected', text_color)]
        )

        self.style.configure('TLabel', background=base_bg, foreground=text_color, font=('Segoe UI', 10))
        self.style.configure('Title.TLabel', background=base_bg, foreground=text_color, font=('Segoe UI Semibold', 20))
        self.style.configure('Subtitle.TLabel', background=base_bg, foreground=subtle_text, font=('Segoe UI', 11))
        self.style.configure('SectionTitle.TLabel', background=base_bg, foreground=text_color, font=('Segoe UI Semibold', 14))
        self.style.configure('Info.TLabel', background=base_bg, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('FieldLabel.TLabel', background=base_bg, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('Warning.TLabel', background=base_bg, foreground=danger_color, font=('Segoe UI Semibold', 11))
        self.style.configure('Status.TLabel', background=surface_alt, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('MetricTitle.TLabel', background=surface_bg, foreground=subtle_text, font=('Segoe UI', 10))
        self.style.configure('MetricValue.TLabel', background=surface_bg, foreground=text_color, font=('Segoe UI Semibold', 16))
        self.style.configure('Badge.TLabel', background=accent_color, foreground=on_accent, font=('Segoe UI Semibold', 9), padding=(10, 2))

        self.style.configure('TButton', font=('Segoe UI', 10), padding=6)
        self.style.configure(
            'Accent.TButton',
            background=accent_color,
            foreground=on_accent,
            font=('Segoe UI Semibold', 10),
            padding=(16, 8)
        )
        self.style.map(
            'Accent.TButton',
            background=[('pressed', accent_hover), ('active', accent_hover), ('disabled', surface_alt)],
            foreground=[('disabled', subtle_text), ('!disabled', on_accent)]
        )
        self.style.configure(
            'Ghost.TButton',
            background=surface_alt,
            foreground=text_color,
            padding=(14, 6)
        )
        self.style.map(
            'Ghost.TButton',
            background=[('pressed', surface_bg), ('active', surface_bg)],
            foreground=[('disabled', subtle_text)]
        )
        self.style.configure(
            'Toolbar.TButton',
            background=surface_bg,
            foreground=text_color,
            padding=(14, 6)
        )
        self.style.map(
            'Toolbar.TButton',
            background=[('pressed', accent_soft), ('active', accent_soft)],
            foreground=[('disabled', subtle_text)]
        )
        self.style.configure('Link.TButton', background=base_bg, foreground=accent_color, padding=0)
        self.style.map('Link.TButton', foreground=[('active', accent_hover)])

        self.style.configure(
            'TEntry',
            fieldbackground=surface_bg,
            background=surface_bg,
            foreground=text_color,
            insertcolor=accent_color,
            bordercolor=border_color
        )
        self.style.map(
            'TEntry',
            fieldbackground=[('disabled', surface_alt)],
            foreground=[('disabled', subtle_text)]
        )
        self.style.configure(
            'TCombobox',
            fieldbackground=surface_bg,
            background=surface_bg,
            foreground=text_color,
            bordercolor=border_color
        )
        self.style.map(
            'TCombobox',
            fieldbackground=[('readonly', surface_bg), ('disabled', surface_alt)],
            foreground=[('disabled', subtle_text)]
        )

        self.style.configure(
            'Treeview',
            background=surface_bg,
            fieldbackground=surface_bg,
            foreground=text_color,
            font=('Segoe UI', 10),
            bordercolor=border_color,
            rowheight=26
        )
        self.style.configure(
            'Treeview.Heading',
            background=surface_alt,
            foreground=subtle_text,
            font=('Segoe UI Semibold', 10),
            padding=6,
            relief='flat'
        )
        self.style.map(
            'Treeview.Heading',
            background=[('active', accent_color)],
            foreground=[('active', on_accent)]
        )
        self.style.configure('Dashboard.Treeview', rowheight=26)
        self.style.map(
            'Treeview',
            background=[('selected', accent_soft)],
            foreground=[('selected', accent_text)]
        )

        self.style.configure('Vertical.TScrollbar', background=surface_alt, troughcolor=surface_bg, bordercolor=surface_alt)
        self.style.map('Vertical.TScrollbar', background=[('active', surface_bg)])
        self.style.configure('Horizontal.TScrollbar', background=surface_alt, troughcolor=surface_bg, bordercolor=surface_alt)
        self.style.map('Horizontal.TScrollbar', background=[('active', surface_bg)])

        self.style.configure('TCheckbutton', background=base_bg, foreground=text_color, font=('Segoe UI', 10))
        self.style.configure('Switch.TCheckbutton', background=base_bg, foreground=text_color, font=('Segoe UI', 10, 'bold'))

        self.theme_variant = variant

    def _build_palette(self, variant: str) -> Dict[str, str]:
        """Farbpaletten für helle und dunkle Darstellung."""
        if variant == 'modern_dark':
            return {
                'base': '#0f172a',
                'surface': '#1e293b',
                'surface_alt': '#15213b',
                'accent': '#3b82f6',
                'accent_hover': '#2563eb',
                'accent_soft': '#1d4ed8',
                'accent_text': '#f8fafc',
                'text': '#e2e8f0',
                'subtle_text': '#94a3b8',
                'border': '#1f2a44',
                'danger': '#f87171',
                'on_accent': '#ffffff',
                'log_bg': '#0b1120',
                'log_fg': '#f8fafc'
            }

        return {
            'base': '#f5f7fb',
            'surface': '#ffffff',
            'surface_alt': '#eef1f8',
            'accent': '#2f6bff',
            'accent_hover': '#1c4fd9',
            'accent_soft': '#dbe4ff',
            'accent_text': '#1b2330',
            'text': '#1b2330',
            'subtle_text': '#5c6473',
            'border': '#d7dcea',
            'danger': '#d8334a',
            'on_accent': '#ffffff',
            'log_bg': '#10131a',
            'log_fg': '#f7f9fc'
        }

    def _apply_palette_to_widgets(self):
        """Aktuelle Farbpalette auf bestehende Widgets anwenden."""
        if not hasattr(self, 'palette'):
            return

        self.root.configure(bg=self.palette['base'])

        if hasattr(self, 'main_frame'):
            self.main_frame.configure(style='Surface.TFrame')
        if hasattr(self, 'header_frame'):
            self.header_frame.configure(style='Header.TFrame')
        if hasattr(self, 'status_frame'):
            self.status_frame.configure(style='Status.TFrame')
        if hasattr(self, 'status_label'):
            self.status_label.configure(style='Status.TLabel')
        if hasattr(self, 'chat_summary_label'):
            self.chat_summary_label.configure(style='Info.TLabel')
        if hasattr(self, 'trade_status_label'):
            self.trade_status_label.configure(style='Info.TLabel')
        if hasattr(self, 'statistics_hint'):
            self.statistics_hint.configure(style='Info.TLabel')
        if self.start_button is not None:
            self.start_button.configure(style='Accent.TButton')
        if hasattr(self, 'theme_selector') and self.theme_selector is not None:
            self.theme_selector.configure(style='TCombobox')
        if hasattr(self, 'notebook'):
            self.notebook.configure(style='TNotebook')

        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.configure(
                bg=self.palette['log_bg'],
                fg=self.palette['log_fg'],
                insertbackground=self.palette['log_fg'],
                highlightbackground=self.palette['border'],
                highlightcolor=self.palette['border'],
                highlightthickness=1,
                borderwidth=0,
                relief='flat'
            )

    def on_theme_change(self, _event=None):
        """Theme-Auswahl aus dem Dropdown anwenden."""
        selection = (self.theme_var.get() or '').lower()
        variant = 'modern_dark' if 'dark' in selection or 'dunkel' in selection else 'modern_light'
        if variant != self.theme_variant:
            self._configure_styles(variant)
            self._apply_palette_to_widgets()

    def create_widgets(self):
        """GUI-Widgets erstellen"""

        # Main Container
        self.main_frame = ttk.Frame(self.root, padding=(20, 20, 20, 15), style='Surface.TFrame')
        self.main_frame.pack(fill='both', expand=True)

        # Header
        header_frame = ttk.Frame(self.main_frame, style='Header.TFrame')
        header_frame.pack(fill='x', pady=(0, 15))
        header_frame.columnconfigure(0, weight=1)
        header_frame.columnconfigure(1, weight=0)
        self.header_frame = header_frame

        title_container = ttk.Frame(header_frame, style='Header.TFrame')
        title_container.grid(row=0, column=0, sticky='w')

        ttk.Label(
            title_container,
            text="📊 Multi-Chat Trading Cockpit",
            style='Title.TLabel'
        ).pack(side='left')

        ttk.Label(
            title_container,
            text="Synchronisiere Signale & verwalte Quellen in Echtzeit",
            style='Subtitle.TLabel'
        ).pack(side='left', padx=(18, 0))

        ttk.Label(
            title_container,
            text="Modern UI",
            style='Badge.TLabel'
        ).pack(side='left', padx=(18, 0))

        controls_container = ttk.Frame(header_frame, style='Header.TFrame')
        controls_container.grid(row=0, column=1, sticky='e')

        ttk.Label(
            controls_container,
            text="Darstellung",
            style='Info.TLabel'
        ).pack(side='left', padx=(0, 8))

        self.theme_selector = ttk.Combobox(
            controls_container,
            textvariable=self.theme_var,
            values=['Modern Light', 'Modern Dark'],
            state='readonly',
            width=16
        )
        self.theme_selector.pack(side='left')
        self.theme_selector.bind('<<ComboboxSelected>>', self.on_theme_change)

        # Notebook-Tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill='both', expand=True)

        self.create_chat_management_tab()
        self.create_trading_tab()
        self.create_statistics_tab()

        # Status Bar
        self.status_frame = ttk.Frame(self.main_frame, style='Status.TFrame')
        self.status_frame.pack(fill='x', pady=(10, 0))

        self.status_label = ttk.Label(self.status_frame, text="Bot gestoppt", style='Status.TLabel')
        self.status_label.pack(side='left')

        button_frame = ttk.Frame(self.status_frame, style='Status.TFrame')
        button_frame.pack(side='right')

        self.start_button = ttk.Button(
            button_frame,
            text="▶ Bot starten",
            command=self.start_bot,
            style='Accent.TButton'
        )
        self.start_button.pack(side='left', padx=(0, 8))
        ttk.Button(
            button_frame,
            text="■ Bot stoppen",
            command=self.stop_bot,
            style='Ghost.TButton'
        ).pack(side='left')

    def create_chat_management_tab(self):
        """Chat-Management Tab"""
        chat_frame = ttk.Frame(self.notebook, padding=(20, 20, 20, 15), style='Surface.TFrame')
        self.notebook.add(chat_frame, text="Chat Management")

        header = ttk.Frame(chat_frame, style='Surface.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text="Chat-Quellen", style='SectionTitle.TLabel').pack(side='left')
        self.chat_summary_label = ttk.Label(header, text="Keine Chats geladen", style='Info.TLabel')
        self.chat_summary_label.pack(side='right')

        controls_frame = ttk.Frame(chat_frame, style='Toolbar.TFrame')
        controls_frame.pack(fill='x', pady=(15, 12))
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

        list_frame = ttk.LabelFrame(chat_frame, text="Verfügbare Chats", padding="12", style='Card.TLabelframe')
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

        info_frame = ttk.Frame(chat_frame, style='Surface.TFrame')
        info_frame.pack(fill='x', pady=(12, 0))
        ttk.Label(
            info_frame,
            text="Hinweis: Aktivierte Chats werden kontinuierlich synchronisiert.",
            style='Info.TLabel'
        ).pack(side='left')
        ttk.Button(
            info_frame,
            text="ℹ Hilfe",
            style='Link.TButton',
            command=lambda: messagebox.showinfo(
                "Information",
                "Markieren Sie Chats und nutzen Sie die Toolbar, um die Überwachung anzupassen."
            )
        ).pack(side='right')

    def create_trading_tab(self):
        """Trading Tab"""
        trading_frame = ttk.Frame(self.notebook, padding=(20, 20, 20, 15), style='Surface.TFrame')
        self.notebook.add(trading_frame, text="Trading")

        header = ttk.Frame(trading_frame, style='Surface.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text="Trading-Steuerung", style='SectionTitle.TLabel').pack(side='left')
        self.trade_status_label = ttk.Label(header, text="Demo aktiv", style='Info.TLabel')
        self.trade_status_label.pack(side='right')

        settings_frame = ttk.Frame(trading_frame, style='Surface.TFrame')
        settings_frame.pack(fill='x', pady=(15, 12))
        settings_frame.columnconfigure((0, 1, 2), weight=1)

        self.demo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame,
            text="Demo-Modus (Empfohlen)",
            variable=self.demo_var,
            command=self.toggle_demo_mode,
            style='Switch.TCheckbutton'
        ).grid(row=0, column=0, sticky='w')

        ttk.Label(settings_frame, text="Handelsmodus:", style='FieldLabel.TLabel').grid(row=0, column=1, sticky='w')
        self.execution_mode_var = tk.StringVar(value="Sofortausführung")
        ttk.Combobox(
            settings_frame,
            textvariable=self.execution_mode_var,
            values=["Sofortausführung", "Zone Monitoring", "Ausgeschaltet"],
            state='readonly'
        ).grid(row=0, column=2, sticky='ew', padx=(8, 0))

        warning_label = ttk.Label(
            trading_frame,
            text="⚠ WARNUNG: Automatisiertes Trading birgt hohe Verlustrisiken!",
            style='Warning.TLabel'
        )
        warning_label.pack(fill='x', pady=(0, 15))

        toolbar = ttk.Frame(trading_frame, style='Toolbar.TFrame')
        toolbar.pack(fill='x', pady=(0, 10))
        ttk.Button(toolbar, text="📥 Neue Signale", style='Toolbar.TButton', command=self.load_chats).pack(side='left')
        ttk.Button(toolbar, text="🧹 Log leeren", style='Toolbar.TButton', command=self.clear_log).pack(side='left', padx=(10, 0))
        ttk.Button(toolbar, text="📊 Statistiken", style='Toolbar.TButton', command=self.refresh_statistics).pack(side='left', padx=(10, 0))

        metrics_frame = ttk.Frame(trading_frame, style='Surface.TFrame')
        metrics_frame.pack(fill='x', pady=(0, 15))
        metrics_frame.columnconfigure((0, 1, 2), weight=1)
        for idx, (title, value) in enumerate([
            ("Aktive Signale", "0"),
            ("Offene Trades", "0"),
            ("Heute synchronisiert", "0")
        ]):
            metric = ttk.Frame(metrics_frame, style='Metric.TFrame', padding=(16, 12))
            metric.grid(row=0, column=idx, padx=(0 if idx == 0 else 12, 0), sticky='nsew')
            ttk.Label(metric, text=title, style='MetricTitle.TLabel').pack(anchor='w')
            ttk.Label(metric, text=value, style='MetricValue.TLabel').pack(anchor='w', pady=(4, 0))

        log_frame = ttk.LabelFrame(trading_frame, text="Live Trade Log", padding="12", style='Card.TLabelframe')
        log_frame.pack(fill='both', expand=True)

        self.log_text = tk.Text(
            log_frame,
            height=16,
            wrap='word',
            bg=self.palette['log_bg'],
            fg=self.palette['log_fg'],
            insertbackground=self.palette['log_fg'],
            font=('Consolas', 11),
            borderwidth=0,
            relief='flat',
            highlightthickness=1,
            highlightbackground=self.palette['border'],
            highlightcolor=self.palette['border']
        )
        self.log_text.pack(side='left', fill='both', expand=True)

        log_scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        log_scroll.pack(side='right', fill='y')
        self.log_text.configure(yscrollcommand=log_scroll.set, padx=12, pady=10)

    def create_statistics_tab(self):
        """Statistiken Tab"""
        stats_frame = ttk.Frame(self.notebook, padding=(20, 20, 20, 15), style='Surface.TFrame')
        self.notebook.add(stats_frame, text="Statistiken")

        header = ttk.Frame(stats_frame, style='Surface.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text="Performance & Statistiken", style='SectionTitle.TLabel').pack(side='left')
        self.statistics_hint = ttk.Label(header, text="Letzte Aktualisierung: –", style='Info.TLabel')
        self.statistics_hint.pack(side='right')

        kpi_frame = ttk.Frame(stats_frame, style='Surface.TFrame')
        kpi_frame.pack(fill='x', pady=(15, 15))
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

        source_frame = ttk.LabelFrame(stats_frame, text="Statistiken nach Quelle", padding="12", style='Card.TLabelframe')
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

        actions_frame = ttk.Frame(stats_frame, style='Surface.TFrame')
        actions_frame.pack(fill='x', pady=(12, 0))
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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                chats = loop.run_until_complete(self.bot.load_all_chats())
                self.root.after(0, lambda: self.update_chat_list(chats))
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"Fehler beim Laden: {e}"))
            finally:
                loop.close()

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

    def toggle_demo_mode(self):
        """Demo-Modus umschalten"""
        self.bot.demo_mode = self.demo_var.get()
        mode_text = "Demo-Modus" if self.bot.demo_mode else "LIVE-Modus"
        if hasattr(self, 'trade_status_label'):
            status_text = "Demo aktiv" if self.bot.demo_mode else "LIVE aktiv"
            self.trade_status_label.config(text=status_text)
        self.log_message(f"Modus geändert: {mode_text}")

    def start_bot(self):
        """Bot starten"""
        if self.bot.is_running or self.bot_starting:
            return

        self.bot_starting = True
        if self.start_button:
            self.start_button.config(state='disabled')

        def run_bot():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            started = False
            try:
                started = loop.run_until_complete(self.bot.start())
                if started:
                    self.root.after(0, self.after_bot_started)
                else:
                    self.root.after(0, self.handle_bot_start_failure)
            except Exception as e:
                self.root.after(0, lambda e=e: self.handle_bot_start_exception(e))
            finally:
                if started:
                    # run_until_disconnected läuft in eigenem Task; Loop offen halten:
                    try:
                        loop.run_forever()
                    except Exception:
                        pass
                loop.close()

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
        if self.bot.client:
            try:
                # Client sauber trennen
                self.bot.client.disconnect()
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

            except queue.Empty:
                pass

            self.root.after(100, process_messages)

        process_messages()

    def apply_config(self, config: Dict):
        """Konfiguration auf Bot und GUI anwenden"""
        telegram_cfg = config.get('telegram', {})
        session_name = telegram_cfg.get('session_name', 'trading_session')
        self.bot.update_credentials(
            telegram_cfg.get('api_id'),
            telegram_cfg.get('api_hash'),
            telegram_cfg.get('phone'),
            session_name=session_name
        )

        trading_cfg = config.get('trading', {})
        demo_mode = bool(trading_cfg.get('demo_mode', True))
        self.bot.demo_mode = demo_mode
        if hasattr(self, 'demo_var'):
            self.demo_var.set(demo_mode)
        if hasattr(self, 'trade_status_label'):
            self.trade_status_label.config(text="Demo aktiv" if demo_mode else "LIVE aktiv")

    def log_message(self, message):
        """Log-Nachricht in GUI anzeigen"""
        self.log_text.insert('end', f"{message}\n")
        self.log_text.see('end')

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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.bot.complete_login_with_code(code))
                self.root.after(0, lambda res=result: self.handle_login_code_result(res))
            except Exception as e:
                self.root.after(0, lambda err=e: self.handle_login_code_exception(err))
            finally:
                loop.close()

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
            "• Verwenden Sie IMMER zuerst den Demo-Modus\n"
            "• Testen Sie alle Funktionen gründlich\n"
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
        ttk.Entry(form_frame, textvariable=self.setup_api_hash, width=40, show='*').pack(fill='x', pady=(0, 10))

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
        "• Umfangreiche Tests erforderlich\n"
        "• Demo-Modus wird dringend empfohlen\n\n"
        "Möchten Sie fortfahren?"
    )
    result = messagebox.askyesno("Sicherheitswarnung", warning_text, icon='warning')
    root.destroy()
    return result


# ==================== MAIN ====================

def main():
    """Hauptfunktion mit Setup-Assistent"""
    if not show_startup_warning():
        print("Programm abgebrochen.")
        return

    try:
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
        cfg = ConfigManager().load_config()
        app = TradingGUI(cfg)
        app.run()

    except Exception as e:
        print(f"Fehler beim Starten der Anwendung: {e}")
        input("Drücken Sie Enter zum Beenden...")


if __name__ == "__main__":
    main()
