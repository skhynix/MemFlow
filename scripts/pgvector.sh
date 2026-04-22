#!/bin/bash
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Usage:
#   Option 1: Run script
#     $ cd scripts
#     $ ./pgvector.sh
#
#   Option 2: Run manually
#     $ cd scripts
#     $ docker compose -f docker-compose.pgvector.yml down -v  # Clean start
#     $ mkdir -p ../postgres_data
#     $ docker compose -f docker-compose.pgvector.yml up -d
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.pgvector.yml"
DATA_DIR="$SCRIPT_DIR/../postgres_data"

echo "Stopping and removing containers with volumes..."
docker compose -f "$COMPOSE_FILE" down -v

echo "Creating data directory..."
mkdir -p "$DATA_DIR"

echo "Starting postgresql container..."
docker compose -f "$COMPOSE_FILE" up -d

echo "Container status:"
docker compose -f "$COMPOSE_FILE" ps
