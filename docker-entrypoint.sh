#!/bin/sh
set -e
mkdir -p /app/data/audio
chown -R 1001:1001 /app/data 2>/dev/null || true
exec gosu nextjs node server.js
