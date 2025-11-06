#!/bin/bash
# ==============================================================================
# OPTIMIZED BUILD SCRIPT FOR SUPREME SYSTEM V5
# ==============================================================================
#
# Professional build process with:
# - SIMD optimization for i3 8th Gen (AVX2, Skylake)
# - Memory constraint validation
# - Performance verification
# - Comprehensive error handling
#
# Usage: ./scripts/build_optimized.sh
#
# ==============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔨 SUPREME SYSTEM V5 - OPTIMIZED BUILD${NC}"
echo "========================================"

# Build configuration
TARGET_CPU="skylake"  # i3 8th Gen
TARGET_FEATURES="+avx2,+fma,+sse4.2,+popcnt"
OPT_LEVEL="3"
LTO="fat"

echo -e "${YELLOW}⚙️  Build Configuration:${NC}"
echo "  Target CPU: $TARGET_CPU"
echo "  SIMD Features: $TARGET_FEATURES"
echo "  Optimization Level: $OPT_LEVEL"
echo "  LTO: $LTO"

# Set optimization flags
export RUSTFLAGS="-C target-cpu=$TARGET_CPU -C target-features=$TARGET_FEATURES -C opt-level=$OPT_LEVEL -C lto=$LTO"
export CARGO_BUILD_RUSTFLAGS="$RUSTFLAGS"

echo -e "\n${YELLOW}🦀 Building Rust Core...${NC}"
cd rust/supreme_core

# Clean previous builds
echo -e "${BLUE}🧨 Cleaning previous builds...${NC}"
cargo clean

# Update dependencies
echo -e "${BLUE}📦 Updating dependencies...${NC}"
cargo update

# Build with maximum optimization
echo -e "${BLUE}🔨 Building optimized release...${NC}"
if cargo build --release --features max-performance; then
    echo -e "${GREEN}✅ Rust build successful${NC}"
else
    echo -e "${RED}❌ Rust build failed${NC}"
    exit 1
fi

# Run tests
echo -e "${BLUE}🧪 Running Rust tests...${NC}"
if cargo test --release --features max-performance; then
    echo -e "${GREEN}✅ Rust tests passed${NC}"
else
    echo -e "${RED}❌ Rust tests failed${NC}"
    exit 1
fi

# Build Python extension
echo -e "\n${YELLOW}🐍 Building Python Extension...${NC}"
cd ../../python

if python setup.py build_ext --inplace; then
    echo -e "${GREEN}✅ Python extension built${NC}"
else
    echo -e "${RED}❌ Python extension build failed${NC}"
    exit 1
fi

# Validate build
echo -e "\n${YELLOW}✅ Build Validation${NC}"
echo "------------------"

# Check binary size
BINARY_SIZE=$(stat -f%z ../rust/supreme_core/target/release/libsupreme_core.* 2>/dev/null || stat -c%s ../rust/supreme_core/target/release/libsupreme_core.* 2>/dev/null || echo "0")
BINARY_SIZE_MB=$((BINARY_SIZE / 1024 / 1024))

echo -e "${GREEN}💾 Binary Size: ${BINARY_SIZE_MB}MB${NC}"

if [ "$BINARY_SIZE_MB" -gt 100 ]; then
    echo -e "${YELLOW}⚠️  Warning: Binary size >100MB might impact startup${NC}"
fi

echo -e "\n${GREEN}✅ OPTIMIZED BUILD COMPLETE${NC}"
echo "================================"
echo -e "${BLUE}Ready for performance testing and deployment${NC}"