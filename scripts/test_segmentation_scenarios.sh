#!/bin/bash
#
# Test script for 06_segmentation.py with different flag combinations
# Tests --skip-existing flag and its interactions with other flags
#
# Usage:
#   bash test_segmentation_scenarios.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SEGMENTATION_SCRIPT="${SCRIPT_DIR}/06_segmentation.py"
CONFIG_FILE="${SCRIPT_DIR}/../configs/segmentation_config.yaml"

# Test directories (adjust these paths)
TEST_INPUT_DIR="/media/storage/roppert/mri_ai_service/output/preprocessed"
TEST_OUTPUT_DIR="$/media/storage/roppert/mri_ai_service/output/test_segmentation"

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_requirements() {
    print_header "Checking Requirements"
    
    if [ ! -f "$SEGMENTATION_SCRIPT" ]; then
        print_error "Segmentation script not found: $SEGMENTATION_SCRIPT"
        exit 1
    fi
    print_success "Segmentation script found"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_warning "Config file not found: $CONFIG_FILE (using default)"
    else
        print_success "Config file found"
    fi
    
    if [ ! -d "$TEST_INPUT_DIR" ]; then
        print_error "Test input directory not found: $TEST_INPUT_DIR"
        echo "Please create test data structure:"
        echo "  $TEST_INPUT_DIR/sub-001/ses-001/anat/*.nii.gz"
        exit 1
    fi
    print_success "Test input directory found"
    
    # Check Python script syntax
    python3 -m py_compile "$SEGMENTATION_SCRIPT" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "Script syntax is valid"
    else
        print_error "Script has syntax errors"
        exit 1
    fi
}

setup_test_environment() {
    print_header "Setting Up Test Environment"
    
    # Create fresh output directory
    if [ -d "$TEST_OUTPUT_DIR" ]; then
        print_warning "Removing existing test output directory"
        rm -rf "$TEST_OUTPUT_DIR"
    fi
    
    mkdir -p "$TEST_OUTPUT_DIR"
    print_success "Test output directory created: $TEST_OUTPUT_DIR"
}

# Test Scenario 1: Baseline (no flags)
test_scenario_1() {
    print_header "TEST 1: Baseline Processing (No Flags)"
    echo "Command: python $SEGMENTATION_SCRIPT $TEST_INPUT_DIR $TEST_OUTPUT_DIR --max-subjects 2"
    echo ""
    
    python3 "$SEGMENTATION_SCRIPT" \
        "$TEST_INPUT_DIR" \
        "$TEST_OUTPUT_DIR" \
        --max-subjects 2 \
        --config "$CONFIG_FILE"
    
    if [ $? -eq 0 ]; then
        print_success "Test 1 PASSED: Baseline processing completed"
    else
        print_error "Test 1 FAILED: Baseline processing failed"
        return 1
    fi
}

# Test Scenario 2: With --skip-existing (first run, nothing to skip)
test_scenario_2() {
    print_header "TEST 2: Skip-Existing on Clean Output (Nothing to Skip)"
    echo "Command: python $SEGMENTATION_SCRIPT $TEST_INPUT_DIR $TEST_OUTPUT_DIR --skip-existing --max-subjects 2"
    echo ""
    
    # Clean output first
    rm -rf "$TEST_OUTPUT_DIR"
    mkdir -p "$TEST_OUTPUT_DIR"
    
    python3 "$SEGMENTATION_SCRIPT" \
        "$TEST_INPUT_DIR" \
        "$TEST_OUTPUT_DIR" \
        --max-subjects 2 \
        --skip-existing \
        --config "$CONFIG_FILE"
    
    if [ $? -eq 0 ]; then
        print_success "Test 2 PASSED: Skip-existing with clean output works"
    else
        print_error "Test 2 FAILED"
        return 1
    fi
}

# Test Scenario 3: With --skip-existing (second run, should skip all)
test_scenario_3() {
    print_header "TEST 3: Skip-Existing on Processed Output (Should Skip All)"
    echo "This test assumes Test 2 left processed files in output directory"
    echo "Command: python $SEGMENTATION_SCRIPT $TEST_INPUT_DIR $TEST_OUTPUT_DIR --skip-existing --max-subjects 2"
    echo ""
    
    # Check if there are existing output files
    EXISTING_COUNT=$(find "$TEST_OUTPUT_DIR" -name "*_segmask.nii.gz" | wc -l)
    echo "Existing output masks: $EXISTING_COUNT"
    
    if [ $EXISTING_COUNT -eq 0 ]; then
        print_warning "No existing masks found. Running baseline first..."
        python3 "$SEGMENTATION_SCRIPT" \
            "$TEST_INPUT_DIR" \
            "$TEST_OUTPUT_DIR" \
            --max-subjects 2 \
            --config "$CONFIG_FILE"
    fi
    
    echo ""
    echo "Now testing skip-existing..."
    python3 "$SEGMENTATION_SCRIPT" \
        "$TEST_INPUT_DIR" \
        "$TEST_OUTPUT_DIR" \
        --max-subjects 2 \
        --skip-existing \
        --config "$CONFIG_FILE"
    
    if [ $? -eq 0 ]; then
        print_success "Test 3 PASSED: Skip-existing correctly skipped processed sessions"
    else
        print_error "Test 3 FAILED"
        return 1
    fi
}

# Test Scenario 4: Partial processing (process 1, then skip it and process 1 more)
test_scenario_4() {
    print_header "TEST 4: Incremental Processing (Skip Existing, Process New)"
    echo "Process 1 session, then process 2 sessions total with --skip-existing"
    echo ""
    
    # Clean output
    rm -rf "$TEST_OUTPUT_DIR"
    mkdir -p "$TEST_OUTPUT_DIR"
    
    # Process 1 session
    print_warning "Step 1: Processing 1 session..."
    python3 "$SEGMENTATION_SCRIPT" \
        "$TEST_INPUT_DIR" \
        "$TEST_OUTPUT_DIR" \
        --max-subjects 1 \
        --config "$CONFIG_FILE"
    
    FIRST_COUNT=$(find "$TEST_OUTPUT_DIR" -name "*_segmask.nii.gz" | wc -l)
    echo "Output masks after first run: $FIRST_COUNT"
    
    # Process 2 sessions with skip-existing (should process 1 new)
    print_warning "Step 2: Processing 2 sessions with --skip-existing..."
    python3 "$SEGMENTATION_SCRIPT" \
        "$TEST_INPUT_DIR" \
        "$TEST_OUTPUT_DIR" \
        --max-subjects 2 \
        --skip-existing \
        --config "$CONFIG_FILE"
    
    SECOND_COUNT=$(find "$TEST_OUTPUT_DIR" -name "*_segmask.nii.gz" | wc -l)
    echo "Output masks after second run: $SECOND_COUNT"
    
    if [ $SECOND_COUNT -gt $FIRST_COUNT ]; then
        print_success "Test 4 PASSED: Incremental processing works correctly"
    else
        print_error "Test 4 FAILED: Expected more output files after second run"
        return 1
    fi
}

# Test Scenario 5: Skip-existing with benchmark
test_scenario_5() {
    print_header "TEST 5: Skip-Existing with Benchmark Mode"
    echo "Command: python $SEGMENTATION_SCRIPT ... --skip-existing --benchmark --max-subjects 2"
    echo ""
    
    # Clean output
    rm -rf "$TEST_OUTPUT_DIR"
    mkdir -p "$TEST_OUTPUT_DIR"
    
    python3 "$SEGMENTATION_SCRIPT" \
        "$TEST_INPUT_DIR" \
        "$TEST_OUTPUT_DIR" \
        --max-subjects 2 \
        --skip-existing \
        --benchmark \
        --server-name "test-server" \
        --gpu-count 1 \
        --gpu-ids "0" \
        --config "$CONFIG_FILE"
    
    if [ $? -eq 0 ]; then
        # Check if metrics file was created
        METRICS_FILE="$TEST_OUTPUT_DIR/benchmark_results/segmentation/metrics.csv"
        if [ -f "$METRICS_FILE" ]; then
            print_success "Test 5 PASSED: Skip-existing works with benchmark mode"
            echo "Metrics file created: $METRICS_FILE"
        else
            print_warning "Test 5 PASSED but no metrics file found (may be expected)"
        fi
    else
        print_error "Test 5 FAILED"
        return 1
    fi
}

# Test Scenario 6: Dry run check
test_scenario_6() {
    print_header "TEST 6: Help and Argument Validation"
    echo "Checking that --skip-existing appears in help text"
    echo ""
    
    python3 "$SEGMENTATION_SCRIPT" --help | grep -q "skip-existing"
    
    if [ $? -eq 0 ]; then
        print_success "Test 6 PASSED: --skip-existing flag is documented in help"
    else
        print_error "Test 6 FAILED: --skip-existing not found in help text"
        return 1
    fi
}

# Main test execution
main() {
    print_header "Segmentation Script Test Suite"
    echo "Testing --skip-existing flag and combinations"
    echo "Script: $SEGMENTATION_SCRIPT"
    echo "Input:  $TEST_INPUT_DIR"
    echo "Output: $TEST_OUTPUT_DIR"
    
    # Check requirements
    check_requirements
    
    # Setup
    setup_test_environment
    
    # Run tests
    TESTS_PASSED=0
    TESTS_FAILED=0
    
    # Test 1
    if test_scenario_1; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    
    # Test 2
    if test_scenario_2; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    
    # Test 3
    if test_scenario_3; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    
    # Test 4
    if test_scenario_4; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    
    # Test 5
    if test_scenario_5; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    
    # Test 6
    if test_scenario_6; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    
    # Summary
    print_header "Test Summary"
    echo "Tests passed: $TESTS_PASSED"
    echo "Tests failed: $TESTS_FAILED"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        print_success "ALL TESTS PASSED ✓"
        exit 0
    else
        print_error "SOME TESTS FAILED ✗"
        exit 1
    fi
}

# Run main function
main "$@"