#!/usr/bin/env python3
"""
Simple test runner script for BabyLlama
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ {description} passed!")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"✗ {description} failed!")
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.stdout:
            print("STDOUT:", result.stdout)

    return result.returncode == 0


def main():
    """Run test suite with various options"""
    print("BabyLlama Test Suite Runner")
    print("=" * 60)

    # Check if in project root
    if not Path("pyproject.toml").exists():
        print("Error: Must run from project root directory")
        sys.exit(1)

    all_passed = True

    # 1. Run linting checks
    if "--no-lint" not in sys.argv:
        all_passed &= run_command(["ruff", "check", "."], "Ruff linting")
        all_passed &= run_command(["black", "--check", "."], "Black formatting check")

    # 2. Run unit tests
    if "--no-unit" not in sys.argv:
        all_passed &= run_command(
            ["pytest", "tests/", "-v", "-m", "not integration", "-x"], "Unit tests"
        )

    # 3. Run integration tests
    if "--integration" in sys.argv:
        all_passed &= run_command(
            ["pytest", "tests/test_integration.py", "-v", "-m", "integration"],
            "Integration tests",
        )

    # 4. Run with coverage
    if "--coverage" in sys.argv:
        all_passed &= run_command(
            ["pytest", "--cov=.", "--cov-report=term-missing", "--cov-report=html"],
            "Tests with coverage",
        )
        print("\nCoverage report generated in htmlcov/index.html")

    # 5. Run quick smoke test
    if "--smoke" in sys.argv:
        all_passed &= run_command(
            [
                "python",
                "test_model.py",
            ],
            "Smoke test",
        )

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
