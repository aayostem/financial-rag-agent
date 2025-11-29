#!/bin/bash

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Running Performance Tests${NC}"

# Run performance tests
echo -e "${YELLOW}1. Running RAG performance tests...${NC}"
pytest tests/performance/test_rag_performance.py -v --tb=short

echo -e "${YELLOW}2. Running API performance tests...${NC}"
pytest tests/performance/test_api_performance.py -v --tb=short

echo -e "${YELLOW}3. Running memory usage tests...${NC}"
pytest tests/performance/test_memory_usage.py -v --tb=short

# Run Locust load tests
echo -e "${YELLOW}4. Starting Locust load tests...${NC}"
echo -e "${YELLOW}   Open http://localhost:8089 to monitor tests${NC}"
locust -f tests/performance/load_tests/locustfile.py --headless -u 100 -r 10 -t 5m --host=http://localhost:8000

echo -e "${GREEN}Performance tests completed!${NC}"