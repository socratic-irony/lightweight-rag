#!/usr/bin/env python3
"""
Quick code quality metrics script for lightweight-rag.
Run this to get a snapshot of current code quality metrics.
"""

import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def get_test_coverage():
    """Get test coverage percentage."""
    print("\n📊 Running test coverage...")
    code, out, err = run_cmd("pytest --cov=lightweight_rag --cov-report=term-missing -q 2>&1 | grep TOTAL")
    if code == 0 and out:
        parts = out.split()
        for i, part in enumerate(parts):
            if part == "TOTAL":
                try:
                    coverage = parts[i+3].rstrip('%')
                    return float(coverage)
                except:
                    pass
    return None


def get_complexity_metrics():
    """Get cyclomatic complexity metrics."""
    print("\n📊 Analyzing cyclomatic complexity...")
    code, out, err = run_cmd("radon cc lightweight_rag/ -a -s 2>&1 | tail -3")
    if code == 0 and out:
        lines = out.strip().split('\n')
        for line in lines:
            if "blocks" in line and "Average complexity:" in line:
                # Extract complexity grade
                parts = line.split("Average complexity:")
                if len(parts) > 1:
                    complexity = parts[1].strip().split()[0]
                    return complexity
    return None


def get_maintainability():
    """Get maintainability index."""
    print("\n📊 Calculating maintainability index...")
    code, out, err = run_cmd("radon mi lightweight_rag/ -s 2>&1")
    if code == 0 and out:
        # Count A, B, C ratings
        a_count = out.count(" - A (")
        b_count = out.count(" - B (")
        c_count = out.count(" - C (")
        total = a_count + b_count + c_count
        if total > 0:
            return f"{a_count}A/{b_count}B/{c_count}C (of {total})"
    return None


def count_high_complexity_functions():
    """Count functions with high complexity."""
    print("\n📊 Counting high-complexity functions...")
    code, out, err = run_cmd("radon cc lightweight_rag/ -n C -s 2>&1 | grep -c ' - '")
    if code == 0 and out:
        try:
            return int(out.strip())
        except:
            pass
    return None


def get_flake8_violations():
    """Get flake8 violation count."""
    print("\n📊 Running flake8 checks...")
    code, out, err = run_cmd("flake8 lightweight_rag/ --max-line-length=100 --statistics 2>&1 | tail -1")
    # Count lines with violations
    code, out, err = run_cmd("flake8 lightweight_rag/ --max-line-length=100 2>&1 | wc -l")
    if code == 0 and out:
        try:
            return int(out.strip())
        except:
            pass
    return None


def get_security_issues():
    """Get security issues from bandit."""
    print("\n📊 Running security scan...")
    code, out, err = run_cmd("bandit -r lightweight_rag/ -f txt 2>&1 | grep 'No issues identified'")
    if "No issues identified" in out:
        return 0
    # Try to count issues
    code, out, err = run_cmd("bandit -r lightweight_rag/ -f txt 2>&1 | grep -c 'Issue:'")
    if code == 0 and out:
        try:
            return int(out.strip())
        except:
            pass
    return None


def main():
    """Generate code quality report."""
    print("=" * 70)
    print("🔍 LIGHTWEIGHT RAG - CODE QUALITY METRICS")
    print("=" * 70)
    
    # Get all metrics
    coverage = get_test_coverage()
    complexity = get_complexity_metrics()
    maintainability = get_maintainability()
    high_complexity = count_high_complexity_functions()
    violations = get_flake8_violations()
    security = get_security_issues()
    
    # Display results
    print("\n" + "=" * 70)
    print("📋 CURRENT METRICS")
    print("=" * 70)
    
    print(f"\n✅ Test Coverage: {coverage}%" if coverage else "\n❌ Test Coverage: Failed to calculate")
    target_coverage = 80
    if coverage:
        if coverage >= target_coverage:
            print(f"   Target: {target_coverage}% ✓ PASSED")
        else:
            gap = target_coverage - coverage
            print(f"   Target: {target_coverage}% ✗ Need {gap:.1f}% more")
    
    print(f"\n✅ Avg Cyclomatic Complexity: {complexity}" if complexity else "\n❌ Complexity: Failed to calculate")
    print("   Target: <= B (10) " + ("✓ PASSED" if complexity and complexity in ['A', 'B'] else "✗ NEEDS WORK"))
    
    print(f"\n✅ Maintainability: {maintainability}" if maintainability else "\n❌ Maintainability: Failed to calculate")
    
    print(f"\n⚠️  High Complexity Functions (C+): {high_complexity}" if high_complexity is not None else "\n❌ High Complexity: Failed to count")
    print("   Target: 0 " + ("✓ PASSED" if high_complexity == 0 else f"✗ {high_complexity} need refactoring"))
    
    print(f"\n⚠️  Flake8 Violations: {violations}" if violations is not None else "\n❌ Violations: Failed to count")
    print("   Target: < 10 " + ("✓ PASSED" if violations is not None and violations < 10 else "✗ NEEDS CLEANUP"))
    
    print(f"\n✅ Security Issues: {security}" if security is not None else "\n❌ Security: Failed to scan")
    print("   Target: 0 " + ("✓ PASSED" if security == 0 else f"✗ {security} issues found"))
    
    # Overall grade
    print("\n" + "=" * 70)
    print("📊 OVERALL ASSESSMENT")
    print("=" * 70)
    
    passed = 0
    total = 0
    
    if coverage is not None:
        total += 1
        if coverage >= target_coverage:
            passed += 1
    
    if complexity:
        total += 1
        if complexity in ['A', 'B']:
            passed += 1
    
    if high_complexity is not None:
        total += 1
        if high_complexity <= 5:  # Some tolerance
            passed += 1
    
    if violations is not None:
        total += 1
        if violations < 50:  # Current state
            passed += 1
    
    if security is not None:
        total += 1
        if security == 0:
            passed += 1
    
    if total > 0:
        score = (passed / total) * 100
        print(f"\n✨ Score: {passed}/{total} checks passed ({score:.0f}%)")
        
        if score >= 80:
            print("   Grade: A - Excellent! 🌟")
        elif score >= 70:
            print("   Grade: B+ - Good, minor improvements needed ✓")
        elif score >= 60:
            print("   Grade: B - Acceptable, some work needed ⚠️")
        else:
            print("   Grade: C - Needs significant improvement ⚠️⚠️")
    
    print("\n" + "=" * 70)
    print("📖 For detailed analysis, see: docs/CODE_QUALITY_REPORT.md")
    print("📋 For action items, see: docs/CODE_QUALITY_CHECKLIST.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
