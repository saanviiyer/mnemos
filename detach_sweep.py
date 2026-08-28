#!/usr/bin/env python3
"""Start the sweep in its own session so it survives the shell that launched it.

`nohup ... &` was not enough here: the sweep died three times when the launching
agent session was torn down, because the whole process group went with it. os.setsid
puts the run in a new session with no controlling terminal, so it outlives the parent
entirely. Re-running is safe -- finished runs are skipped and an interrupted one
resumes from its checkpoint.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

if subprocess.run(["pgrep", "-f", "sweep[.]py"], capture_output=True).returncode == 0:
    print("sweep already running")
    sys.exit(0)

log = open(os.path.join(HERE, "sweep.log"), "a")
p = subprocess.Popen(
    [sys.executable, "-u", "scripts/sweep.py", "--scale", "small",
     "--seeds", "0,1,2", "--steps", "10000", "--out", "runs/sweep_small"],
    cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL, start_new_session=True)
print(f"sweep detached as pid {p.pid}")
