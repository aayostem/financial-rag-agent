#!/usr/bin/env python3
"""
Backup Manager for Financial RAG Agent
Handles automated backups of databases and vector stores
"""

import os
import sys
import yaml
import argparse
import subprocess
import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("backup-manager")


class BackupManager:
    def __init__(self, namespace, environment, backup_dir="/backups"):
        self.namespace = namespace
        self.environment = environment
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp for backup files
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def backup_postgresql(self):
        """Backup PostgreSQL database"""
        logger.info("Starting PostgreSQL backup...")

        # Get PostgreSQL credentials from Kubernetes secret
        try:
            # In production, use external secrets or vault
            # For demo purposes, we'll use the values from the chart
            postgres_secret = (
                subprocess.check_output(
                    [
                        "kubectl",
                        "get",
                        "secret",
                        f"financial-rag-agent-{self.environment}-postgresql",
                        "-n",
                        self.namespace,
                        "-o",
                        "jsonpath={.data.postgresql-password}",
                    ]
                )
                .decode()
                .strip()
            )

            # Decode base64 password
            import base64

            postgres_password = base64.b64decode(postgres_secret).decode()

            # Create backup
            backup_file = self.backup_dir / f"postgresql_backup_{self.timestamp}.sql"

            # Use kubectl to exec into the pod and run pg_dump
            subprocess.run(
                [
                    "kubectl",
                    "exec",
                    f"financial-rag-agent-{self.environment}-postgresql-0",
                    "-n",
                    self.namespace,
                    "--",
                    "pg_dump",
                    "-U",
                    "financial_user",
                    "-d",
                    "financial_rag",
                    f"--file=/tmp/backup.sql",
                ],
                check=True,
            )

            # Copy backup file from pod to local
            subprocess.run(
                [
                    "kubectl",
                    "cp",
                    f"{self.namespace}/financial-rag-agent-{self.environment}-postgresql-0:/tmp/backup.sql",
                    str(backup_file),
                ],
                check=True,
            )

            logger.info(f"PostgreSQL backup completed: {backup_file}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"PostgreSQL backup failed: {e}")
            return False

    def backup_redis(self):
        """Backup Redis data"""
        logger.info("Starting Redis backup...")

        try:
            # Execute Redis SAVE command
            subprocess.run(
                [
                    "kubectl",
                    "exec",
                    f"financial-rag-agent-{self.environment}-redis-master-0",
                    "-n",
                    self.namespace,
                    "--",
                    "redis-cli",
                    "SAVE",
                ],
                check=True,
            )

            # Copy dump.rdb from pod
            backup_file = self.backup_dir / f"redis_backup_{self.timestamp}.rdb"
            subprocess.run(
                [
                    "kubectl",
                    "cp",
                    f"{self.namespace}/financial-rag-agent-{self.environment}-redis-master-0:/data/dump.rdb",
                    str(backup_file),
                ],
                check=True,
            )

            logger.info(f"Redis backup completed: {backup_file}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Redis backup failed: {e}")
            return False

    def backup_vector_store(self):
        """Backup ChromaDB vector store"""
        logger.info("Starting Vector Store backup...")

        try:
            backup_file = self.backup_dir / f"chroma_backup_{self.timestamp}.tar.gz"

            # Create tar archive of ChromaDB data
            subprocess.run(
                [
                    "kubectl",
                    "exec",
                    f"financial-rag-agent-{self.environment}-vector-store-0",
                    "-n",
                    self.namespace,
                    "--",
                    "tar",
                    "czf",
                    "/tmp/chroma_backup.tar.gz",
                    "-C",
                    "/chroma/data",
                    ".",
                ],
                check=True,
            )

            # Copy backup from pod
            subprocess.run(
                [
                    "kubectl",
                    "cp",
                    f"{self.namespace}/financial-rag-agent-{self.environment}-vector-store-0:/tmp/chroma_backup.tar.gz",
                    str(backup_file),
                ],
                check=True,
            )

            logger.info(f"Vector store backup completed: {backup_file}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Vector store backup failed: {e}")
            return False

    def cleanup_old_backups(self, retention_days=30):
        """Clean up backups older than retention_days"""
        logger.info(f"Cleaning up backups older than {retention_days} days...")

        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=retention_days)

        for backup_file in self.backup_dir.glob("*_backup_*"):
            if backup_file.is_file():
                file_time = datetime.datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_time < cutoff_time:
                    backup_file.unlink()
                    logger.info(f"Deleted old backup: {backup_file}")

    def run_full_backup(self):
        """Run complete backup of all components"""
        logger.info("Starting full backup...")

        results = {
            "postgresql": self.backup_postgresql(),
            "redis": self.backup_redis(),
            "vector_store": self.backup_vector_store(),
        }

        # Cleanup old backups
        self.cleanup_old_backups()

        # Report status
        successful = sum(results.values())
        total = len(results)

        logger.info(f"Backup completed: {successful}/{total} components backed up successfully")

        return all(results.values())


def main():
    parser = argparse.ArgumentParser(description="Financial RAG Agent Backup Manager")
    parser.add_argument("--namespace", default="financial-rag", help="Kubernetes namespace")
    parser.add_argument(
        "--environment",
        required=True,
        choices=["development", "staging", "production"],
        help="Environment",
    )
    parser.add_argument("--backup-dir", default="/backups", help="Backup directory")
    parser.add_argument(
        "--component",
        choices=["all", "postgresql", "redis", "vector-store"],
        default="all",
        help="Component to backup",
    )

    args = parser.parse_args()

    manager = BackupManager(args.namespace, args.environment, args.backup_dir)

    if args.component == "all":
        success = manager.run_full_backup()
    elif args.component == "postgresql":
        success = manager.backup_postgresql()
    elif args.component == "redis":
        success = manager.backup_redis()
    elif args.component == "vector-store":
        success = manager.backup_vector_store()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
