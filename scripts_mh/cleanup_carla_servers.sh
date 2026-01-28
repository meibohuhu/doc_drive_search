#!/bin/bash
# Cleanup script for CARLA servers
# This script kills all running CARLA server processes and processes using CARLA ports

# Configuration
PORT_BASE="${PORT_BASE:-20000}"
TM_PORT_BASE="${TM_PORT_BASE:-30000}"
PORT_RANGE="${PORT_RANGE:-100}"  # Check ports from BASE to BASE+RANGE

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "CARLA Server Cleanup Script"
echo "=========================================="
echo "Port base: ${PORT_BASE}"
echo "TM port base: ${TM_PORT_BASE}"
echo "Port range: ${PORT_RANGE}"
echo ""

# Function to check if a process exists
check_process() {
    local pid=$1
    if kill -0 "${pid}" 2>/dev/null; then
        return 0  # Process exists
    else
        return 1  # Process doesn't exist
    fi
}

# Function to kill a process gracefully, then force kill if needed
kill_process() {
    local pid=$1
    local name=$2
    
    if check_process "${pid}"; then
        echo -e "${YELLOW}Killing ${name} (PID: ${pid})...${NC}"
        kill "${pid}" 2>/dev/null || true
        sleep 2
        
        if check_process "${pid}"; then
            echo -e "${RED}Force killing ${name} (PID: ${pid})...${NC}"
            kill -9 "${pid}" 2>/dev/null || true
            sleep 1
        fi
        
        if check_process "${pid}"; then
            echo -e "${RED}Failed to kill ${name} (PID: ${pid})${NC}"
            return 1
        else
            echo -e "${GREEN}Successfully killed ${name} (PID: ${pid})${NC}"
            return 0
        fi
    else
        echo -e "${YELLOW}${name} (PID: ${pid}) already stopped${NC}"
        return 0
    fi
}

# Track killed processes
KILLED_COUNT=0

# 1. Kill CARLA processes by name
echo "=========================================="
echo "Step 1: Killing CARLA processes by name"
echo "=========================================="

# Find all CARLA-related processes
CARLA_PIDS=$(pgrep -f "CarlaUE4" 2>/dev/null || echo "")
CARLA_PIDS="$CARLA_PIDS $(pgrep -f "carla" 2>/dev/null | grep -v $$ || echo "")"

if [ -n "${CARLA_PIDS}" ]; then
    for pid in ${CARLA_PIDS}; do
        # Skip this script's own PID
        if [ "${pid}" != "$$" ]; then
            # Get process name for display
            proc_name=$(ps -p "${pid}" -o comm= 2>/dev/null || echo "CARLA process")
            if kill_process "${pid}" "${proc_name}"; then
                KILLED_COUNT=$((KILLED_COUNT + 1))
            fi
        fi
    done
else
    echo "No CARLA processes found by name"
fi

echo ""

# 2. Kill processes using CARLA ports
echo "=========================================="
echo "Step 2: Killing processes using CARLA ports"
echo "=========================================="

# Check CARLA RPC ports (PORT_BASE to PORT_BASE+PORT_RANGE)
for port in $(seq ${PORT_BASE} $((PORT_BASE + PORT_RANGE))); do
    # Use lsof to find process using this port
    pid=$(lsof -ti:${port} 2>/dev/null || echo "")
    
    if [ -n "${pid}" ]; then
        proc_name=$(ps -p "${pid}" -o comm= 2>/dev/null || echo "process")
        echo -e "${YELLOW}Found process using port ${port}: ${proc_name} (PID: ${pid})${NC}"
        if kill_process "${pid}" "${proc_name} on port ${port}"; then
            KILLED_COUNT=$((KILLED_COUNT + 1))
        fi
    fi
done

# Check Traffic Manager ports (TM_PORT_BASE to TM_PORT_BASE+PORT_RANGE)
for port in $(seq ${TM_PORT_BASE} $((TM_PORT_BASE + PORT_RANGE))); do
    pid=$(lsof -ti:${port} 2>/dev/null || echo "")
    
    if [ -n "${pid}" ]; then
        proc_name=$(ps -p "${pid}" -o comm= 2>/dev/null || echo "process")
        echo -e "${YELLOW}Found process using TM port ${port}: ${proc_name} (PID: ${pid})${NC}"
        if kill_process "${pid}" "${proc_name} on TM port ${port}"; then
            KILLED_COUNT=$((KILLED_COUNT + 1))
        fi
    fi
done

echo ""

# 3. Final check for any remaining CARLA processes
echo "=========================================="
echo "Step 3: Final check for remaining CARLA processes"
echo "=========================================="

REMAINING_CARLA=$(pgrep -f "CarlaUE4" 2>/dev/null || echo "")
REMAINING_PORTS=0

# Check if any ports are still in use
for port in $(seq ${PORT_BASE} $((PORT_BASE + PORT_RANGE))); do
    if lsof -ti:${port} >/dev/null 2>&1; then
        REMAINING_PORTS=$((REMAINING_PORTS + 1))
    fi
done

for port in $(seq ${TM_PORT_BASE} $((TM_PORT_BASE + PORT_RANGE))); do
    if lsof -ti:${port} >/dev/null 2>&1; then
        REMAINING_PORTS=$((REMAINING_PORTS + 1))
    fi
done

# Summary
echo ""
echo "=========================================="
echo "Cleanup Summary"
echo "=========================================="
echo "Processes killed: ${KILLED_COUNT}"

if [ -n "${REMAINING_CARLA}" ]; then
    echo -e "${RED}⚠️  Warning: ${REMAINING_CARLA} CARLA process(es) still running${NC}"
    echo "Remaining PIDs: ${REMAINING_CARLA}"
else
    echo -e "${GREEN}✅ No CARLA processes remaining${NC}"
fi

if [ ${REMAINING_PORTS} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Warning: ${REMAINING_PORTS} port(s) still in use${NC}"
    echo "You may need to wait a few seconds for ports to be released"
else
    echo -e "${GREEN}✅ All CARLA ports are free${NC}"
fi

echo ""

if [ -z "${REMAINING_CARLA}" ] && [ ${REMAINING_PORTS} -eq 0 ]; then
    echo -e "${GREEN}✅ Cleanup completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Cleanup completed with warnings${NC}"
    exit 1
fi

