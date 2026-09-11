#!/bin/bash

echo "=== pwd ==="
pwd
echo "=== top-level of / ==="
ls -la /
echo "=== top-level of scratch/cwd ==="
ls -la .
echo "=== relative config/ ==="
ls -la config/ 2>&1
echo "=== absolute /config ==="
ls -la /config 2>&1
echo "=== searching for noise_schedule*.csv ==="
find / -maxdepth 4 -iname "noise_schedule*.csv" 2>/dev/null
find . -maxdepth 4 -iname "noise_schedule*.csv" 2>/dev/null
echo "=== condor env vars ==="
env | grep -i condor
