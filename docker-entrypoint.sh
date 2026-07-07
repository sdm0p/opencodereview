#!/bin/bash
# =============================================================================
# OpenCodeReview — Docker Entrypoint
# =============================================================================
# Runs as root to ensure the ChromaDB vector store directory is writable by
# the non-root appuser (uid 1000). Then drops privileges and execs the CMD.
#
# This fixes the "unable to open database file" error that occurs when a
# Docker named volume (owned by root) is mounted at /tmp/opencodereview_vectors
# and the app runs as a non-root user.
# =============================================================================

set -e

VECTOR_DIR="/tmp/opencodereview_vectors"

# Ensure the vector store directory exists and is writable by appuser
mkdir -p "$VECTOR_DIR"
chown -R appuser:appuser "$VECTOR_DIR"

# Drop privileges to appuser and execute the main command
exec su -s /bin/sh appuser -c "exec $*"
