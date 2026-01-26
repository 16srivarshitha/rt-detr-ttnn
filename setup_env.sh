#!/bin/bash
export TT_METAL_HOME=/root/tt-metal
export PYTHONPATH=/root/tt-metal:$PYTHONPATH
export ARCH_NAME=wormhole_b0
export LD_LIBRARY_PATH=/root/tt-metal/build/lib:$LD_LIBRARY_PATH
export TT_METAL_SKIP_HUGEPAGE_CHECK=1
export TT_METAL_SINGLE_CHIP=1
