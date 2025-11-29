#!/bin/bash

# create-tests-structure.sh

echo "Creating tests folder structure..."

# Create main tests directory
mkdir -p tests

# Create root test files
touch tests/pytest.ini
touch tests/conftest.py
touch tests/requirements-test.txt

# Create unit tests directory structure
mkdir -p tests/unit/fixtures

# Create unit test files
touch tests/unit/test_agents.py
touch tests/unit/test_rag_engine.py
touch tests/unit/test_data_ingestion.py
touch tests/unit/test_financial_tools.py
touch tests/unit/test_models.py
touch tests/unit/test_api_endpoints.py

# Create unit fixtures files
touch tests/unit/fixtures/test_data.py
touch tests/unit/fixtures/mock_services.py

# Create integration tests directory structure
mkdir -p tests/integration/fixtures

# Create integration test files
touch tests/integration/test_agent_coordination.py
touch tests/integration/test_rag_pipeline.py
touch tests/integration/test_database_integration.py
touch tests/integration/test_external_apis.py

# Create integration fixtures files
touch tests/integration/fixtures/integration_setup.py

# Create performance tests directory structure
mkdir -p tests/performance/load_tests

# Create performance test files
touch tests/performance/test_rag_performance.py
touch tests/performance/test_api_performance.py
touch tests/performance/test_concurrent_requests.py
touch tests/performance/test_memory_usage.py

# Create load test files
touch tests/performance/load_tests/k6_load_test.js
touch tests/performance/load_tests/locustfile.py

# Create e2e tests directory structure
mkdir -p tests/e2e/scenarios

# Create e2e test files
touch tests/e2e/test_full_workflow.py
touch tests/e2e/test_user_scenarios.py

# Create e2e scenario files
touch tests/e2e/scenarios/investment_research.py
touch tests/e2e/scenarios/risk_assessment.py

# Create test utils directory
mkdir -p tests/test_utils

# Create test utils files
touch tests/test_utils/test_helpers.py
touch tests/test_utils/mock_servers.py
touch tests/test_utils/data_generators.py

echo "Tests folder structure created successfully!"
echo "Location: $(pwd)/tests"

