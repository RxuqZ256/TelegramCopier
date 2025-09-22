# TelegramCopier

Er kopiert Signale von Telegram zu MT5.

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
2. Trage dort Login (Kontonummer), Passwort und Server deines gewünschten MetaTrader-5-Kontos ein. Optional kannst du den Pfad zur `terminal64.exe` des Terminals hinterlegen, falls du mehrere Installationen nutzt.
3. Speichere die Eingaben mit **„Zugangsdaten speichern“**. Die Daten landen in `trading_config.json`, damit der Bot sie beim nächsten Start automatisch verwendet.
4. Öffne dein MetaTrader-5-Terminal, melde dich im gewünschten Konto an und klicke anschließend auf **„Verbindung testen“**. Der Bot prüft damit die Schnittstelle und zeigt bei Erfolg Konto-Informationen (Login, Balance, Währung), damit du siehst, welches Konto angebunden ist.
5. Der Statusbereich im Tab informiert darüber, ob die Zugangsdaten vollständig sind. Nach einem erfolgreichen Test erscheint dort die aktuelle Kontoübersicht, bis du die Daten änderst.
6. Falls das Python-Modul `MetaTrader5` noch fehlt, kannst du die Felder trotzdem ausfüllen und den Terminalpfad auswählen. Installiere das Paket später (z. B. mit `pip install MetaTrader5`) und starte die Anwendung neu, um die Verbindung zu aktivieren.

> Hinweis: Ohne installiertes MetaTrader5-Python-Modul bleibt der Live-Modus deaktiviert. Die Felder kannst du trotzdem ausfüllen; der Status erinnert dich daran, das Paket nachzuinstallieren. Installiere MetaTrader 5 inklusive des Python-Pakets `MetaTrader5`, damit die Verbindung funktioniert.
