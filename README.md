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
