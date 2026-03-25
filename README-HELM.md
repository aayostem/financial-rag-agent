# Financial RAG Agent - Helm Deployment Guide

## Overview

This Helm chart deploys the complete Financial RAG Agent system to Kubernetes, including all dependencies and monitoring components.

## Prerequisites

- Kubernetes cluster 1.20+
- Helm 3.8+
- kubectl configured to access your cluster
- 10GB+ of storage for persistent volumes

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/financial-rag-agent
cd financial-rag-agent