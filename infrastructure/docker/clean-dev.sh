#!/bin/bash

echo "⚠️  WARNING: This will remove all containers and volumes!"
read -p "Are you sure? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "🗑️  Removing containers and volumes..."
    docker compose down -v
    echo "✅ Cleanup complete"
fi