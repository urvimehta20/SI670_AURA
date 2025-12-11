#!/usr/bin/env python3
"""Fix Stage 5 SLURM scripts to add import validation."""

import re
from pathlib import Path

validation_block = '''# ============================================================================
# Import Validation: Verify all imports work before expensive training
# ============================================================================

log "=========================================="
log "Validating Stage 5 Imports"
log "=========================================="

VALIDATION_SCRIPT="$ORIG_DIR/src/scripts/validate_stage5_imports.py"
if [ ! -f "$VALIDATION_SCRIPT" ]; then
    log "⚠ WARNING: Import validation script not found: $VALIDATION_SCRIPT"
    log "  Skipping import validation (not recommended)"
else
    log "Running import validation (testing with dummy tensors)..."
    if "$PYTHON_CMD" "$VALIDATION_SCRIPT" 2>&1 | tee -a "$LOG_FILE"; then
        log "✓ Import validation passed"
    else
        VALIDATION_EXIT_CODE=${PIPESTATUS[0]}
        log "✗ ERROR: Import validation failed (exit code: $VALIDATION_EXIT_CODE)"
        log "  Please review the validation output above"
        log "  Training will not proceed until imports are validated"
        log "  This catches import errors before expensive training jobs"
        exit $VALIDATION_EXIT_CODE
    fi
fi
'''

scripts_dir = Path(__file__).parent
stage5_scripts = sorted(scripts_dir.glob("slurm_stage5*.sh"))
# Exclude the coordinator script
stage5_scripts = [s for s in stage5_scripts if s.name != "slurm_stage5_training.sh"]

for script_path in stage5_scripts:
    print(f"Processing {script_path.name}...")
    
    content = script_path.read_text()
    
    # Remove any existing validation blocks (clean up previous attempts)
    content = re.sub(
        r'# ============================================================================\n# Import Validation:.*?\nfi\n',
        '',
        content,
        flags=re.DOTALL
    )
    
    # Find the pattern: cd "$ORIG_DIR" followed by PYTHON_CMD definition
    # Then insert validation block before "# Validate Python script exists"
    pattern = r'(cd "\$ORIG_DIR"[^\n]*\nPYTHON_CMD=.*?\n)'
    
    def insert_validation(match):
        return match.group(1) + '\n' + validation_block + '\n'
    
    content = re.sub(pattern, insert_validation, content, flags=re.MULTILINE)
    
    # If pattern not found, try alternative: find PYTHON_CMD line and insert after it
    if validation_block not in content:
        pattern2 = r'(PYTHON_CMD=\$\(which python \|\| echo "python"\)\n)'
        content = re.sub(pattern2, r'\1\n' + validation_block + '\n', content)
    
    script_path.write_text(content)
    print(f"  ✓ Updated {script_path.name}")

print("\nDone! All Stage 5 scripts updated.")
