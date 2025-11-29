#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
TEST_TYPE="all"
COVERAGE=false
PARALLEL=false
VERBOSE=false
HTML_REPORT=false

# Function to print usage
usage() {
    echo "Usage: $0 [-t test_type] [-c] [-p] [-v] [-h]"
    echo "  -t  Test type: unit, integration, performance, e2e, all (default: all)"
    echo "  -c  Enable coverage reporting"
    echo "  -p  Run tests in parallel"
    echo "  -v  Verbose output"
    echo "  -r  Generate HTML report"
    echo "  -h  Show this help message"
    exit 1
}

# Parse command line arguments
while getopts "t:cpvrh" opt; do
    case $opt in
        t) TEST_TYPE="$OPTARG" ;;
        c) COVERAGE=true ;;
        p) PARALLEL=true ;;
        v) VERBOSE=true ;;
        r) HTML_REPORT=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Validate test type
case $TEST_TYPE in
    unit|integration|performance|e2e|all) ;;
    *) echo -e "${RED}Error: Test type must be one of: unit, integration, performance, e2e, all${NC}"; exit 1 ;;
esac

echo -e "${GREEN}Running $TEST_TYPE tests...${NC}"

# Build pytest command
PYTEST_CMD="pytest"

# Add markers based on test type
case $TEST_TYPE in
    unit) PYTEST_CMD="$PYTEST_CMD -m unit" ;;
    integration) PYTEST_CMD="$PYTEST_CMD -m integration" ;;
    performance) PYTEST_CMD="$PYTEST_CMD -m performance" ;;
    e2e) PYTEST_CMD="$PYTEST_CMD -m e2e" ;;
    all) PYTEST_CMD="$PYTEST_CMD -m 'not slow'" ;;  # Exclude slow tests by default
esac

# Add options
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=term-missing"
    if [ "$HTML_REPORT" = true ]; then
        PYTEST_CMD="$PYTEST_CMD --cov-report=html:coverage_html"
    fi
fi

if [ "$PARALLEL" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add test path
PYTEST_CMD="$PYTEST_CMD tests/"

# Create test directories
mkdir -p junit
mkdir -p test-reports

echo -e "${YELLOW}Command: $PYTEST_CMD${NC}"

# Execute tests
eval $PYTEST_CMD

TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed. Check the report above.${NC}"
fi

# Generate test report
if [ "$HTML_REPORT" = true ]; then
    echo -e "${YELLOW}Generating HTML test report...${NC}"
    pytest --html=test-reports/report.html --self-contained-html tests/ > /dev/null 2>&1
    echo -e "${GREEN}HTML report generated: test-reports/report.html${NC}"
fi

exit $TEST_RESULT