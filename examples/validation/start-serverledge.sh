#!/bin/sh
set -eu

binary="$1"
config="$2"

# Compose passes the registry address as an environment variable. Viper reads it from TOML.
case "$LAB_ETCD_ADDRESS" in
  ''|*[!a-zA-Z0-9_.:-]*) echo "LAB_ETCD_ADDRESS deve essere un indirizzo host:porta" >&2; exit 2 ;;
esac

{
  printf 'etcd.address = "%s"\n' "$LAB_ETCD_ADDRESS"
  cat "$config"
} > /tmp/real-system.toml

exec "$binary" /tmp/real-system.toml
