"""
Pexpect-based tests for uipath debug command with LlamaIndex workflows.

Tests the interactive debugger functionality including:
- Single breakpoint
- Multiple breakpoints
- Step mode (s command)
- Quit debugger (q command)

Regression test for: ContextStateError when using wait_for_event in
breakpoint wrapper (the wrapper must use an InternalContext, not the
workflow-level ExternalContext).

See: https://github.com/UiPath/uipath-integrations-python/pull/274
"""

import re
import sys
from pathlib import Path
from typing import Optional

import pexpect
import pytest

# ---------------------------------------------------------------------------
# Minimal PromptTest helper (mirrors uipath-langchain-python/testcases/common)
# ---------------------------------------------------------------------------


def strip_ansi(text: str) -> str:
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", text)


class PromptTest:
    def __init__(
        self, command: str, test_name: str, prompt: str = "> ", timeout: int = 60
    ):
        self.command = command
        self.test_name = test_name
        self.prompt = prompt
        self.timeout = timeout
        self.child: Optional[pexpect.spawn] = None
        self._log_handle = None
        self._log_path = Path(f"{test_name}.log")

    def start(self):
        self.child = pexpect.spawn(self.command, encoding="utf-8", timeout=self.timeout)
        self._log_handle = open(self._log_path, "w")
        self.child.logfile_read = self._log_handle

    def send_command(self, command: str, expect: Optional[str] = None):
        self.child.expect(self.prompt)
        self.child.sendline(command)
        if expect:
            self.child.expect(expect)

    def expect_eof(self):
        self.child.expect(pexpect.EOF, timeout=self.timeout)

    def get_output(self) -> str:
        if self._log_path.exists():
            if self._log_handle:
                self._log_handle.flush()
            with open(self._log_path, "r", encoding="utf-8") as f:
                return strip_ansi(f.read())
        return ""

    @property
    def before(self) -> str:
        return self.child.before if self.child else ""

    def close(self):
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None
        if self.child:
            self.child.close()
            self.child = None


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

COMMAND = "uv run uipath debug agent --file input.json"
PROMPT = r"> "
TIMEOUT = 60


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_breakpoint():
    """Test setting and hitting a single breakpoint."""
    test = PromptTest(
        command=COMMAND,
        test_name="debug_single_breakpoint",
        prompt=PROMPT,
        timeout=TIMEOUT,
    )
    try:
        test.start()

        test.send_command(
            "b classify_category", expect=r"Breakpoint set at: classify_category"
        )
        test.send_command("c", expect=r"BREAKPOINT.*classify_category.*before")
        test.send_command("c", expect=r"Debug session completed")

        test.expect_eof()

        output = test.get_output()
        assert "ticket_id" in output, "Expected ticket_id in output"

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print(f"\nERROR: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_multiple_breakpoints():
    """Test setting and hitting multiple breakpoints."""
    test = PromptTest(
        command=COMMAND,
        test_name="debug_multiple_breakpoints",
        prompt=PROMPT,
        timeout=TIMEOUT,
    )
    try:
        test.start()

        test.send_command(
            "b analyze_sentiment", expect=r"Breakpoint set at: analyze_sentiment"
        )
        test.send_command(
            "b determine_priority", expect=r"Breakpoint set at: determine_priority"
        )
        test.send_command("c", expect=r"BREAKPOINT.*analyze_sentiment.*before")
        test.send_command("c", expect=r"BREAKPOINT.*determine_priority.*before")
        test.send_command("c", expect=r"Debug session completed")

        test.expect_eof()

        output = test.get_output()
        breakpoint_count = output.count("BREAKPOINT")
        assert breakpoint_count >= 2, (
            f"Expected at least 2 breakpoints hit, got {breakpoint_count}"
        )

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print(f"\nERROR: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_step_mode():
    """Test step mode - breaks on every node."""
    test = PromptTest(
        command=COMMAND, test_name="debug_step_mode", prompt=PROMPT, timeout=TIMEOUT
    )
    try:
        test.start()

        # Step through all 9 workflow steps
        test.send_command("s", expect=r"BREAKPOINT.*analyze_sentiment.*before")
        test.send_command("s", expect=r"BREAKPOINT.*classify_category.*before")
        test.send_command("s", expect=r"BREAKPOINT.*check_urgency.*before")
        test.send_command("s", expect=r"BREAKPOINT.*determine_priority.*before")
        test.send_command("s", expect=r"BREAKPOINT.*check_escalation.*before")
        test.send_command("s", expect=r"BREAKPOINT.*route_to_department.*before")
        test.send_command("s", expect=r"BREAKPOINT.*assign_queue.*before")
        test.send_command("s", expect=r"BREAKPOINT.*generate_response.*before")
        test.send_command("s", expect=r"BREAKPOINT.*finalize_ticket.*before")
        test.send_command("s", expect=r"Debug session completed")

        test.expect_eof()

        output = test.get_output()
        breakpoint_count = output.count("BREAKPOINT")
        assert breakpoint_count >= 9, (
            f"Expected at least 9 breakpoints in step mode, got {breakpoint_count}"
        )

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print(f"\nERROR: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_quit_debugger():
    """Test quitting the debugger early with 'q' command."""
    test = PromptTest(
        command=COMMAND, test_name="debug_quit", prompt=PROMPT, timeout=TIMEOUT
    )
    try:
        test.start()

        test.send_command("b check_urgency", expect=r"Breakpoint set at: check_urgency")
        test.send_command("c", expect=r"BREAKPOINT.*check_urgency.*before")
        test.send_command("q")

        test.expect_eof()

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print(f"\nERROR: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
