# Qlib Workflow Log Checker

A comprehensive Python script to analyze qlib workflow runner execution logs and determine if commands executed successfully.

## Features

- **Automatic Success/Failure Detection**: Analyzes log patterns to determine execution status
- **Performance Metrics Extraction**: Extracts IC scores, RankIC scores, validation scores, and training epochs
- **Multiple Input Methods**: Read from files, stdin, or execute commands directly
- **Detailed Analysis**: Provides comprehensive reports with timestamps, warnings, and progress tracking
- **Exit Code Support**: Returns appropriate exit codes for automation/scripting

## Usage

### 1. Analyze a Log File

```bash
# Basic analysis
python scripts/data_collector/akshare/check_execution_log.py execution.log

# Quiet mode (only status)
python scripts/data_collector/akshare/check_execution_log.py execution.log --quiet

# Verbose mode (detailed progress steps)
python scripts/data_collector/akshare/check_execution_log.py execution.log --verbose
```

### 2. Pipe from Command Output

```bash
# Run command and pipe output to checker
python scripts/data_collector/akshare/qlib_workflow_runner.py \
  --config scripts/data_collector/akshare/workflow_config_shanghai_simple_transformer.yaml \
  --model transformer \
  --start-date 2022-01-01 --end-date 2024-12-31 \
  --symbols 600000 600036 600519 601318 601398 \
  --experiment-name test_run 2>&1 | \
python scripts/data_collector/akshare/check_execution_log.py --stdin
```

### 3. Execute Command and Check (Recommended)

```bash
# Execute the command and automatically analyze its output
python scripts/data_collector/akshare/check_execution_log.py --command \
  "python scripts/data_collector/akshare/qlib_workflow_runner.py \
   --config scripts/data_collector/akshare/workflow_config_shanghai_simple_transformer.yaml \
   --model transformer \
   --start-date 2022-01-01 --end-date 2024-12-31 \
   --symbols 600 601 \
   --experiment-name shanghai_transformer_production_active_stocks"
```

## Success Indicators

The script looks for these success patterns (in order of priority):

1. `🎉.*?training completed successfully!`
2. `🎉.*?workflow finished successfully!`
3. `✅.*?training completed successfully!`
4. `✅.*?workflow finished successfully!`
5. `✅.*?Data preparation completed!`
6. `✅.*?Step 2 completed.*?successfully`
7. `✅.*?Step 1 completed.*?successfully`

## Failure Indicators

Critical failure patterns that indicate execution failure:

1. `❌.*?training failed`
2. `❌.*?Data preparation failed`
3. `❌.*?Step [12] failed`
4. `ERROR.*?training failed`
5. `ERROR.*?workflow failed`
6. `ERROR.*?Dependency check failed`
7. `ModuleNotFoundError.*?PyTorch models are skipped`
8. `ValueError.*?cannot convert float NaN to integer`
9. `ImportError.*?Failed to import qlib`

## Performance Metrics Extracted

- **IC Scores**: Information Coefficient for train/valid/test sets
- **RankIC Scores**: Rank Information Coefficient for train/valid/test sets
- **Overall IC/RankIC**: Overall performance metrics
- **Best Validation Score**: Best validation score achieved during training
- **Total Epochs**: Number of training epochs completed

## Output Status Codes

- **SUCCESS**: Execution completed successfully with no warnings
- **SUCCESS_WITH_WARNINGS**: Execution completed successfully but with warnings
- **FAILED**: Critical failure detected
- **INCOMPLETE**: Execution appears incomplete (progress detected but no completion)
- **UNKNOWN**: Unable to determine status from log content

## Exit Codes

- `0`: Success (command executed successfully AND log analysis shows success)
- `1`: Failure (command failed OR log analysis shows failure)

## Example Output

### Successful Execution
```
================================================================================
🔍 QLIB WORKFLOW EXECUTION LOG ANALYSIS
================================================================================

✅ OVERALL STATUS: SUCCESS_WITH_WARNINGS
📝 SUMMARY: Execution completed successfully: 🎉 Transformer training completed successfully! (with 2 warnings)
⏰ EXECUTION TIME: 2025-09-19 10:45:00.426 → 2025-09-19 10:45:08.456
📄 LOG LINES: 156

📊 PERFORMANCE METRICS:
   IC Scores - Train: 0.2923, Valid: 0.1858, Test: 0.0530
   RankIC Scores - Train: 0.2785, Valid: 0.1915, Test: 0.0579
   Overall IC: 0.0147
   Overall RankIC: 0.0208
   Total Epochs: 41

✅ SUCCESS INDICATORS (2):
   Line 145: ✅ Direct transformer training completed successfully!
   Line 152: 🎉 Transformer training completed successfully!

⚠️  WARNINGS (2):
   Line 23: ModuleNotFoundError. CatBoostModel are skipped
   Line 24: ModuleNotFoundError. XGBModel is skipped
```

### Failed Execution
```
================================================================================
🔍 QLIB WORKFLOW EXECUTION LOG ANALYSIS
================================================================================

❌ OVERALL STATUS: FAILED
📝 SUMMARY: Critical failure detected: ValueError.*?cannot convert float NaN to integer
⏰ EXECUTION TIME: 2025-09-19 10:45:00.426 → 2025-09-19 10:45:04.456
📄 LOG LINES: 87

❌ CRITICAL FAILURES (3):
   Line 78: ERROR    | __main__:_run_direct_training:762 - Direct training failed
   Line 82: ❌ Transformer training failed
   Line 85: ERROR    | __main__:main:966 - Transformer training workflow failed
```

## Integration with Scripts

You can use this checker in shell scripts for automation:

```bash
#!/bin/bash

# Run qlib workflow and check success
if python scripts/data_collector/akshare/check_execution_log.py --command \
   "python scripts/data_collector/akshare/qlib_workflow_runner.py --config ... --experiment-name prod_run" \
   --quiet; then
    echo "✅ Production run completed successfully!"
    # Continue with next steps...
else
    echo "❌ Production run failed!"
    exit 1
fi
```

## Requirements

- Python 3.7+
- Standard library only (no external dependencies)
- Compatible with qlib workflow runner logging format

## Notes

- The script is designed specifically for the qlib workflow runner logging format
- It handles both stdout and stderr output
- Timestamps are extracted when available for execution time tracking
- The script is safe to use in production environments (read-only analysis)
