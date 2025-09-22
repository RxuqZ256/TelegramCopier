# TelegramCopier

Er kopiert Signale von Telegram zu MT5.

### Windows (Ein-Klick-Start)
1) In GitHub Desktop **Fetch/Pull**.
2) Im Explorer `scripts\fix_mt5_numpy.bat` **Doppelklicken**.
   - Erzwingt NumPy 1.26.4 (kompatibel mit MetaTrader5 5.0.45),
   - installiert Anforderungen,
   - startet `TelegramCopier_Windows.py` (Onboarding-Popup erscheint).

### Windows Quickstart
- **Erststart / Setup:** `scripts\fix_mt5_numpy.bat`
- **Normaler Start:** `scripts\start_windows.bat`
- **Setup erzwingen (Onboarding-Popup erneut):** `scripts\force_setup.bat`

## Windows-EXE erstellen

Mit dem Skript `build_windows_exe.bat` kannst du unter Windows eine ausführbare Datei erstellen. Du benötigst dafür eine 64-Bit-Python-Installation (empfohlen wird Python 3.11), weil PyInstaller die Abhängigkeiten direkt aus der installierten Umgebung einsammelt.

1. Lade dieses Repository auf deinen Windows-Rechner herunter und öffne eine Eingabeaufforderung im Projektordner.
2. Führe das Skript `build_windows_exe.bat` per Doppelklick oder mit `build_windows_exe.bat` in der Eingabeaufforderung aus.
3. Das Skript legt automatisch eine virtuelle Umgebung an, installiert die benötigten Pakete (inklusive PyInstaller) und baut anschließend das Projekt.
4. Die fertige Anwendung findest du danach im Ordner `dist\TelegramCopier`. Die Datei `TelegramCopier.exe` kannst du direkt starten oder an einen anderen Ort kopieren.

> Hinweis: Beim ersten Start der EXE werden die Dateien `trading_config.json` und `chat_config.json` im gleichen Verzeichnis erzeugt. Bewahre sie zusammen mit der EXE auf, wenn du das Programm verschieben möchtest.

Wenn du Anpassungen an den PyInstaller-Optionen vornehmen möchtest (z.B. `--onefile` für eine einzelne EXE), kannst du die entsprechende Zeile im Batch-Skript anpassen.

## MT5-Verbindung konfigurieren

1. Starte die Anwendung und öffne den Tab **„Bot Einstellungen“**.
2. Im Abschnitt **„MetaTrader 5 Verbindung“** kannst du Login (Kontonummer), Passwort, Server und optional den Pfad zur `terminal64.exe` deines MetaTrader-5-Terminals hinterlegen.
3. Klicke auf **„Zugangsdaten speichern“**, damit die Informationen in `trading_config.json` abgelegt und vom Bot übernommen werden.
4. Über **„Verbindung testen“** prüfst du sofort, ob die MT5-Schnittstelle erreichbar ist. Bei Erfolg kannst du den Demo-Modus deaktivieren – auch ein MT5-Demokonto lässt sich so nutzen.

> Hinweis: Ohne installiertes MetaTrader5-Python-Modul bleibt der Live-Modus deaktiviert. Installiere MetaTrader 5 inklusive des Python-Pakets `MetaTrader5`, damit die Verbindung funktioniert.

## Manuelle Tests

- 2024-04-XX: In der GUI ein nicht-numerisches Login (z. B. `12abc`) eingegeben → Warnung "MT5-Login ungültig" erscheint, die Zugangsdaten werden nicht gespeichert und der Status bleibt im Warnzustand.
- 2024-04-XX: Gültiges numerisches Login eingegeben, Passwort und Server ergänzt → Zugangsdaten lassen sich speichern, der Status meldet "MT5-Zugangsdaten geladen" und der bisherige Erfolgsfluss bleibt erhalten.
