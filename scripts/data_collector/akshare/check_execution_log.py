#!/usr/bin/env python3
"""
Log Checker for qlib_workflow_runner.py execution
Analyzes log output to determine if the command executed successfully.

Usage:
    python check_execution_log.py <log_file_path>
    python check_execution_log.py --stdin  # Read from stdin
    python check_execution_log.py --command "python scripts/data_collector/akshare/qlib_workflow_runner.py ..."  # Run command and check
"""

import sys
import re
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class QlibWorkflowLogChecker:
    """Analyzes qlib workflow runner logs to determine execution success."""
    
    # Success indicators (in order of preference)
    SUCCESS_PATTERNS = [
        r"🎉.*?training completed successfully!",
        r"🎉.*?workflow finished successfully!",
        r"✅.*?training completed successfully!",
        r"✅.*?workflow finished successfully!",
        r"✅.*?Data preparation completed!",
        r"✅.*?Step 2 completed.*?successfully",
        r"✅.*?Step 1 completed.*?successfully",
    ]
    
    # Critical failure indicators
    CRITICAL_FAILURE_PATTERNS = [
        r"❌.*?training failed",
        r"❌.*?Data preparation failed",
        r"❌.*?Step [12] failed",
        r"ERROR.*?training failed",
        r"ERROR.*?workflow failed",
        r"ERROR.*?Dependency check failed",
        r"ModuleNotFoundError.*?PyTorch models are skipped",
        r"ValueError.*?cannot convert float NaN to integer",
        r"ImportError.*?Failed to import qlib",
    ]
    
    # Warning patterns (non-critical but noteworthy)
    WARNING_PATTERNS = [
        r"WARNING.*?Failed to.*?",
        r"WARN.*?data not found",
        r"warning.*?Failed.*?",
        r"ModuleNotFoundError.*?CatBoostModel.*?skipped",
        r"ModuleNotFoundError.*?XGBModel.*?skipped",
    ]
    
    # Progress indicators
    PROGRESS_PATTERNS = [
        r"🚀.*?Starting.*?workflow",
        r"Step 1:.*?data preparation",
        r"Step 2:.*?training",
        r"✅.*?Qlib initialized successfully",
        r"✅.*?PyTorch.*?successfully",
        r"Creating.*?model",
        r"Training.*?epochs",
        r"IC.*?Train.*?Valid.*?Test",
    ]
    
    # Performance metrics patterns
    METRICS_PATTERNS = [
        r"IC.*?Train.*?(\d+\.\d+).*?Valid.*?(\d+\.\d+).*?Test.*?(\d+\.\d+)",
        r"RankIC.*?Train.*?(\d+\.\d+).*?Valid.*?(\d+\.\d+).*?Test.*?(\d+\.\d+)",
        r"overall_IC=(\d+\.\d+)",
        r"overall_RankIC=(\d+\.\d+)",
        r"Best validation score.*?(-?\d+\.\d+)",
        r"(\d+) epochs.*?early stopping",
    ]

    def __init__(self):
        self.log_lines = []
        self.analysis_results = {}
        
    def analyze_log(self, log_content: str) -> Dict:
        """
        Analyze log content and return comprehensive results.
        
        Args:
            log_content: Raw log content as string
            
        Returns:
            Dictionary with analysis results
        """
        self.log_lines = log_content.strip().split('\n')
        
        # Initialize results
        results = {
            'success': False,
            'status': 'UNKNOWN',
            'summary': '',
            'details': {
                'success_indicators': [],
                'critical_failures': [],
                'warnings': [],
                'progress_steps': [],
                'performance_metrics': {},
                'total_lines': len(self.log_lines),
                'timestamp_range': self._extract_timestamp_range(),
            }
        }
        
        # Analyze patterns
        results['details']['success_indicators'] = self._find_patterns(self.SUCCESS_PATTERNS)
        results['details']['critical_failures'] = self._find_patterns(self.CRITICAL_FAILURE_PATTERNS)
        results['details']['warnings'] = self._find_patterns(self.WARNING_PATTERNS)
        results['details']['progress_steps'] = self._find_patterns(self.PROGRESS_PATTERNS)
        results['details']['performance_metrics'] = self._extract_metrics()
        
        # Determine overall status
        results['success'], results['status'], results['summary'] = self._determine_status(results['details'])
        
        self.analysis_results = results
        return results
    
    def _find_patterns(self, patterns: List[str]) -> List[Dict]:
        """Find all matches for given patterns in log lines."""
        matches = []
        for i, line in enumerate(self.log_lines):
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    matches.append({
                        'line_number': i + 1,
                        'pattern': pattern,
                        'matched_text': match.group(0),
                        'full_line': line.strip(),
                        'timestamp': self._extract_timestamp(line)
                    })
        return matches
    
    def _extract_metrics(self) -> Dict:
        """Extract performance metrics from log."""
        metrics = {}
        for line in self.log_lines:
            for pattern in self.METRICS_PATTERNS:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if "IC.*Train.*Valid.*Test" in pattern:
                        metrics['ic_scores'] = {
                            'train': float(match.group(1)),
                            'valid': float(match.group(2)),
                            'test': float(match.group(3))
                        }
                    elif "RankIC.*Train.*Valid.*Test" in pattern:
                        metrics['rank_ic_scores'] = {
                            'train': float(match.group(1)),
                            'valid': float(match.group(2)),
                            'test': float(match.group(3))
                        }
                    elif "overall_IC" in pattern:
                        metrics['overall_ic'] = float(match.group(1))
                    elif "overall_RankIC" in pattern:
                        metrics['overall_rank_ic'] = float(match.group(1))
                    elif "Best validation score" in pattern:
                        metrics['best_validation_score'] = float(match.group(1))
                    elif "epochs.*early stopping" in pattern:
                        metrics['total_epochs'] = int(match.group(1))
        return metrics
    
    def _extract_timestamp(self, line: str) -> Optional[str]:
        """Extract timestamp from log line."""
        # Pattern: 2025-09-19 10:45:00.426
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})?)', line)
        return timestamp_match.group(1) if timestamp_match else None
    
    def _extract_timestamp_range(self) -> Dict:
        """Extract first and last timestamps from log."""
        first_ts = None
        last_ts = None
        
        for line in self.log_lines:
            ts = self._extract_timestamp(line)
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
        
        return {'first': first_ts, 'last': last_ts}
    
    def _determine_status(self, details: Dict) -> Tuple[bool, str, str]:
        """Determine overall execution status based on analysis."""
        success_indicators = details['success_indicators']
        critical_failures = details['critical_failures']
        warnings = details['warnings']
        
        # Check for critical failures first
        if critical_failures:
            latest_failure = max(critical_failures, key=lambda x: x['line_number'])
            return False, 'FAILED', f"Critical failure detected: {latest_failure['matched_text']}"
        
        # Check for success indicators
        if success_indicators:
            # Look for the highest priority success indicator
            for pattern in self.SUCCESS_PATTERNS:
                for indicator in success_indicators:
                    if indicator['pattern'] == pattern:
                        status = 'SUCCESS_WITH_WARNINGS' if warnings else 'SUCCESS'
                        summary = f"Execution completed successfully: {indicator['matched_text']}"
                        if warnings:
                            summary += f" (with {len(warnings)} warnings)"
                        return True, status, summary
        
        # No clear success or failure - check for partial completion
        if details['progress_steps']:
            return False, 'INCOMPLETE', f"Execution appears incomplete. Found {len(details['progress_steps'])} progress steps but no clear completion indicator."
        
        return False, 'UNKNOWN', "Unable to determine execution status from log content."
    
    def print_report(self, verbose: bool = False):
        """Print a formatted analysis report."""
        if not self.analysis_results:
            print("❌ No analysis results available. Run analyze_log() first.")
            return
        
        results = self.analysis_results
        details = results['details']
        
        # Header
        print("=" * 80)
        print("🔍 QLIB WORKFLOW EXECUTION LOG ANALYSIS")
        print("=" * 80)
        
        # Overall Status
        status_emoji = "✅" if results['success'] else "❌"
        print(f"\n{status_emoji} OVERALL STATUS: {results['status']}")
        print(f"📝 SUMMARY: {results['summary']}")
        
        # Timestamp info
        if details['timestamp_range']['first']:
            print(f"⏰ EXECUTION TIME: {details['timestamp_range']['first']} → {details['timestamp_range']['last']}")
        
        print(f"📄 LOG LINES: {details['total_lines']}")
        
        # Performance Metrics
        if details['performance_metrics']:
            print(f"\n📊 PERFORMANCE METRICS:")
            metrics = details['performance_metrics']
            if 'ic_scores' in metrics:
                ic = metrics['ic_scores']
                print(f"   IC Scores - Train: {ic['train']:.4f}, Valid: {ic['valid']:.4f}, Test: {ic['test']:.4f}")
            if 'rank_ic_scores' in metrics:
                ric = metrics['rank_ic_scores']
                print(f"   RankIC Scores - Train: {ric['train']:.4f}, Valid: {ric['valid']:.4f}, Test: {ric['test']:.4f}")
            if 'overall_ic' in metrics:
                print(f"   Overall IC: {metrics['overall_ic']:.4f}")
            if 'overall_rank_ic' in metrics:
                print(f"   Overall RankIC: {metrics['overall_rank_ic']:.4f}")
            if 'best_validation_score' in metrics:
                print(f"   Best Validation Score: {metrics['best_validation_score']:.6f}")
            if 'total_epochs' in metrics:
                print(f"   Total Epochs: {metrics['total_epochs']}")
        
        # Success Indicators
        if details['success_indicators']:
            print(f"\n✅ SUCCESS INDICATORS ({len(details['success_indicators'])}):")
            for indicator in details['success_indicators'][-3:]:  # Show last 3
                print(f"   Line {indicator['line_number']}: {indicator['matched_text']}")
        
        # Critical Failures
        if details['critical_failures']:
            print(f"\n❌ CRITICAL FAILURES ({len(details['critical_failures'])}):")
            for failure in details['critical_failures'][-3:]:  # Show last 3
                print(f"   Line {failure['line_number']}: {failure['matched_text']}")
        
        # Warnings
        if details['warnings']:
            print(f"\n⚠️  WARNINGS ({len(details['warnings'])}):")
            for warning in details['warnings'][-3:]:  # Show last 3
                print(f"   Line {warning['line_number']}: {warning['matched_text']}")
        
        # Verbose details
        if verbose:
            print(f"\n🔄 PROGRESS STEPS ({len(details['progress_steps'])}):")
            for step in details['progress_steps']:
                print(f"   Line {step['line_number']}: {step['matched_text']}")
        
        print("\n" + "=" * 80)


def run_command_and_check(command: str) -> Tuple[bool, str]:
    """
    Run the command and capture its output for analysis.
    
    Args:
        command: Shell command to execute
        
    Returns:
        Tuple of (success, log_content)
    """
    print(f"🚀 Executing command: {command}")
    print("⏳ This may take several minutes...")
    
    try:
        # Run command and capture output
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Combine stdout and stderr
        log_content = result.stdout + "\n" + result.stderr
        
        print(f"✅ Command completed with exit code: {result.returncode}")
        return result.returncode == 0, log_content
        
    except subprocess.TimeoutExpired:
        return False, "ERROR: Command timed out after 1 hour"
    except Exception as e:
        return False, f"ERROR: Failed to execute command: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Check qlib workflow runner execution log for success/failure indicators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a log file
  python check_execution_log.py execution.log
  
  # Read from stdin
  python scripts/data_collector/akshare/qlib_workflow_runner.py ... | python check_execution_log.py --stdin
  
  # Run command and check immediately
  python check_execution_log.py --command "python scripts/data_collector/akshare/qlib_workflow_runner.py --config scripts/data_collector/akshare/workflow_config_shanghai_simple_transformer.yaml --model transformer --start-date 2022-01-01 --end-date 2024-12-31 --symbols 600 601 --experiment-name test"
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('log_file', nargs='?', help='Path to log file to analyze')
    group.add_argument('--stdin', action='store_true', help='Read log content from stdin')
    group.add_argument('--command', help='Execute command and analyze its output')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose analysis details')
    parser.add_argument('--quiet', '-q', action='store_true', help='Only show success/failure status')
    
    args = parser.parse_args()
    
    # Get log content
    log_content = ""
    command_success = True
    
    if args.command:
        command_success, log_content = run_command_and_check(args.command)
    elif args.stdin:
        log_content = sys.stdin.read()
    else:
        try:
            with open(args.log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
        except FileNotFoundError:
            print(f"❌ Error: Log file '{args.log_file}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading log file: {e}")
            sys.exit(1)
    
    if not log_content.strip():
        print("❌ Error: No log content to analyze")
        sys.exit(1)
    
    # Analyze log
    checker = QlibWorkflowLogChecker()
    results = checker.analyze_log(log_content)
    
    # Print results
    if args.quiet:
        status_emoji = "✅" if results['success'] else "❌"
        print(f"{status_emoji} {results['status']}: {results['summary']}")
    else:
        checker.print_report(verbose=args.verbose)
    
    # Exit with appropriate code
    # Success if both command execution and log analysis indicate success
    overall_success = command_success and results['success']
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
