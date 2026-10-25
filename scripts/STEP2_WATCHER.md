# Step2 watcher

Files added:
- [scripts/step2_watcher.sh](scripts/step2_watcher.sh) — watcher that reruns multiple tasks while a PID is alive.
- [scripts/start_step1_and_watcher.sh](scripts/start_step1_and_watcher.sh) — convenience wrapper to start step1 and the watcher.
- [scripts/step2_watcher_example.cfg](scripts/step2_watcher_example.cfg) — example config.

Quick start

1. Make scripts executable:

```bash
chmod +x scripts/step2_watcher.sh scripts/start_step1_and_watcher.sh
```

2. Run the watcher if you already have a step1 PID:

```bash
# assume STEP1_PID is known
./scripts/step2_watcher.sh <STEP1_PID> scripts/step2_watcher_example.cfg
```

3. Or start step1 and watcher together:

```bash
./scripts/start_step1_and_watcher.sh scripts/step2_watcher_example.cfg -- python3 -u my_step1.py
```

Logs

The watcher creates a timestamped directory under `/tmp/` (example: `/tmp/step2_watcher_12345_20260505...`) containing `watcher.log`, `task_N.out`, `task_N.err`, and `task_N.pid` files.

LaunchAgent example

If you want the pipeline started by launchd at login, copy a plist like this to `~/Library/LaunchAgents/com.example.step2_pipeline.plist` and `launchctl load` it. Note: the plist below runs the wrapper directly — adjust paths and arguments as needed.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.example.step2_pipeline</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/you/play/applypilot-alok/scripts/start_step1_and_watcher.sh</string>
    <string>/Users/you/play/applypilot-alok/scripts/step2_watcher_example.cfg</string>
    <string>--</string>
    <string>/usr/bin/python3</string>
    <string>/path/to/my_step1.py</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/step2_pipeline.out</string>
  <key>StandardErrorPath</key>
  <string>/tmp/step2_pipeline.err</string>
</dict>
</plist>
```

Notes

- Use absolute paths for commands in the config to avoid PATH issues.
- The watcher checks the given PID with `kill -0` — if that PID exits the watcher stops.
- If `step1` forks to background and the original PID exits, the watcher will stop; if `step1` is a supervisor that stays running, use the wrapper to ensure the right PID is monitored.
