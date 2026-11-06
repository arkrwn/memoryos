#!/bin/bash
# Container health: exits 0 only when /readyz answers 200. The image has bash but no HTTP client,
# so this speaks HTTP over bash's /dev/tcp. A MEMORYOS_BIND on a specific non-loopback address is
# not reachable from here; override the healthcheck in that case.
set -eu

bind=${MEMORYOS_BIND:-0.0.0.0:8080}
exec 3<>"/dev/tcp/127.0.0.1/${bind##*:}"
printf 'GET /readyz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n' >&3
read -r _ status _ <&3
[ "$status" = 200 ]
