# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""ANSI color utilities for terminal output."""


class Colors:
    """ANSI color codes for terminal output."""

    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print a bold header with separator lines."""
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")


def print_section(title: str) -> None:
    """Print a section title with cyan color."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")


def print_info(text: str) -> None:
    """Print info text in cyan."""
    print(f"{Colors.CYAN}{text}{Colors.RESET}")


def print_success(text: str) -> None:
    """Print success text in green."""
    print(f"{Colors.GREEN}{text}{Colors.RESET}")


def print_warning(text: str) -> None:
    """Print warning text in yellow."""
    print(f"{Colors.YELLOW}{text}{Colors.RESET}")


def print_error(text: str) -> None:
    """Print error text in red."""
    print(f"{Colors.RED}{text}{Colors.RESET}")


def print_label(label: str) -> None:
    """Print a label in magenta (e.g., 'Test 1:', 'Task:', etc.)."""
    print(f"{Colors.MAGENTA}{label}{Colors.RESET}")


def print_labeled_text(label: str, text: str, color: str = Colors.CYAN) -> None:
    """Print a colored label followed by uncolored text.

    Args:
        label: Label text (will be colored)
        text: Content text (will be uncolored)
        color: ANSI color code for the label (default: CYAN)
    """
    print(f"{color}{label}{Colors.RESET} {text}")


def print_step(index: int, goal: str, success: bool, output: str = None, error: str = None) -> None:
    """Print a step execution result.

    Args:
        index: Step number (1-based)
        goal: Step description/goal
        success: Whether the step succeeded
        output: Optional output text (truncated to 60 chars)
        error: Optional error text
    """
    status = f"{Colors.GREEN}✓{Colors.RESET}" if success else f"{Colors.RED}✗{Colors.RESET}"
    print(f"  {index}. {status} {goal}")

    if output:
        output_preview = output.replace('\n', ' ')[:60]
        print(f"       Output: {output_preview}")

    if error and not success:
        print(f"       Error: {error}")
