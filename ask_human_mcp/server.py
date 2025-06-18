#!/usr/bin/env python3
"""
Ask-Human MCP Server

Your AI can escalate questions to you through a markdown file,
and you just edit the file with answers.

This is basically that but as a proper MCP server.

This avoids hallucinations and other issues with models and their context,
providing them an escape hatch to ask questions.

This is a work in progress, and I'm not sure if it will work for everyone.

I'm using it to help me develop Kallro.com, and it's working pretty well.
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import re
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Set

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastmcp import FastMCP
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Enable nested asyncio support to fix "Already running asyncio in this thread" errors
try:
    import nest_asyncio
    nest_asyncio.apply()
    logger = logging.getLogger("ask-human-mcp")
    logger.debug("Applied nest_asyncio patch for nested event loop support")
except ImportError:
    logger = logging.getLogger("ask-human-mcp")
    logger.warning("nest_asyncio not available - may encounter event loop issues in some environments")

# Cross-platform file locking
try:
    import fcntl  # Unix/Linux/macOS

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

try:
    import msvcrt  # Windows

    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("ask-human-mcp")


class QuestionStatus(Enum):
    """Status of a question in the system."""

    PENDING = "PENDING"
    ANSWERED = "ANSWERED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class AskHumanError(Exception):
    """Base exception for Ask-Human MCP errors."""

    pass


class QuestionTimeoutError(AskHumanError):
    """Raised when a question times out waiting for an answer."""

    pass


class FileAccessError(AskHumanError):
    """Raised when file operations fail."""

    pass


class InputValidationError(AskHumanError):
    """Raised when input validation fails."""

    pass


@dataclass
class AskHumanConfig:
    """Configuration for the Ask-Human MCP server."""

    ask_file: Path
    timeout: int = 1800  # 30 minutes
    max_question_length: int = 10240  # 10KB
    max_context_length: int = 51200  # 50KB
    max_pending_questions: int = 100
    max_file_size: int = 104857600  # 100MB
    cleanup_interval: int = 300  # 5 minutes
    rotation_size: int = 52428800  # 50MB (when to rotate file)


def get_default_ask_file() -> Path:
    """Get platform-appropriate default location for the ask file."""
    if platform.system() == "Windows":
        # Use user's Documents folder on Windows
        home = Path.home()
        documents = home / "Documents"
        if documents.exists():
            return documents / "ask_human.md"
        return home / "ask_human.md"
    else:
        # Use home directory on Unix-like systems
        return Path.home() / "ask_human.md"


def validate_input(text: str, max_length: int, field_name: str) -> str:
    """Validate and sanitize input text."""
    if not isinstance(text, str):
        raise InputValidationError(f"{field_name} must be a string")

    if len(text) > max_length:
        raise InputValidationError(
            f"{field_name} too long: {len(text)} chars (max {max_length})"
        )

    # Basic sanitization - remove control characters except newlines and tabs
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return sanitized


class CrossPlatformFileLock:
    """Cross-platform file locking utility."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.lock_file = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def acquire(self):
        """Acquire file lock in a cross-platform way."""
        try:
            # Create lock file
            lock_path = self.file_path.with_suffix(self.file_path.suffix + ".lock")
            self.lock_file = open(lock_path, "w", encoding="utf-8")

            if HAS_FCNTL:
                # Unix/Linux/macOS
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif HAS_MSVCRT:
                # Windows
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                # Fallback - just hope for the best
                logger.warning("No file locking available on this platform")

        except OSError as e:
            if self.lock_file:
                self.lock_file.close()
                self.lock_file = None
            raise RuntimeError(f"Could not acquire file lock: {e}")

    def release(self):
        """Release file lock."""
        if self.lock_file:
            try:
                if HAS_FCNTL:
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                elif HAS_MSVCRT:
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)

                self.lock_file.close()

                # Clean up lock file
                lock_path = self.file_path.with_suffix(self.file_path.suffix + ".lock")
                try:
                    lock_path.unlink()
                except (FileNotFoundError, PermissionError):
                    pass  # Lock file already removed or can't remove

            except Exception as e:
                logger.warning(f"Error releasing file lock: {e}")
            finally:
                self.lock_file = None


def normalize_line_endings(text: str) -> str:
    """Normalize line endings to be consistent across platforms."""
    # Convert all line endings to \n
    return text.replace("\r\n", "\n").replace("\r", "\n")


def safe_write_text(file_path: Path, content: str, encoding: str = "utf-8") -> bool:
    """Safely write text to file with proper error handling and atomic operations."""
    try:
        # Normalize line endings
        content = normalize_line_endings(content)

        # Write to temporary file first, then move (atomic operation)
        temp_file = file_path.with_suffix(file_path.suffix + ".tmp")

        with open(temp_file, "w", encoding=encoding, newline="\n") as f:
            f.write(content)
            f.flush()  # Ensure data is written
            os.fsync(f.fileno())  # Force write to disk

        # Atomic move (works on all platforms)
        if platform.system() == "Windows":
            # Windows requires the target to not exist
            if file_path.exists():
                file_path.unlink()

        temp_file.replace(file_path)
        return True

    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        # Clean up temp file if it exists
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            pass
        return False


def safe_read_text(file_path: Path, encoding: str = "utf-8") -> Optional[str]:
    """Safely read text from file with proper encoding detection and error handling."""
    if not file_path.exists():
        return None

    try:
        # Try UTF-8 first
        with open(file_path, encoding=encoding) as f:
            content = f.read()

        # Normalize line endings
        return normalize_line_endings(content)

    except UnicodeDecodeError:
        # Try UTF-8 with BOM
        try:
            with open(file_path, encoding="utf-8-sig") as f:
                content = f.read()
            return normalize_line_endings(content)
        except UnicodeDecodeError:
            # Try system default encoding
            try:
                with open(file_path, encoding=None) as f:
                    content = f.read()
                return normalize_line_endings(content)
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                return None
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None


class MarkdownWatcher(FileSystemEventHandler):
    """
    Watch for changes to the markdown file

    This is how we know when the human has provided an answer
    """

    def __init__(self, file_path: Path):
        super().__init__()
        self.file_path = file_path
        self._callbacks: Dict[str, asyncio.Future] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop to use for callbacks."""
        self._loop = loop

    def on_modified(self, event):
        # Handle both the file and its parent directory
        if not event.is_directory and (
            event.src_path == str(self.file_path)
            or Path(event.src_path).name == self.file_path.name
        ):
            self._schedule_notify_callbacks()

    def on_moved(self, event):
        # Handle file moves/renames
        if not event.is_directory and (
            hasattr(event, "dest_path") and event.dest_path == str(self.file_path)
        ):
            self._schedule_notify_callbacks()

    def _schedule_notify_callbacks(self):
        """Schedule the callback notification in the correct event loop."""
        if self._loop and not self._loop.is_closed():
            # Schedule the coroutine in the correct event loop
            asyncio.run_coroutine_threadsafe(self._notify_callbacks(), self._loop)
        else:
            # Fallback: try to get the running loop (might not work from thread)
            try:
                loop = asyncio.get_running_loop()
                asyncio.run_coroutine_threadsafe(self._notify_callbacks(), loop)
            except RuntimeError:
                # No event loop available - log and continue
                logger.warning("No event loop available for file change notification")

    async def _notify_callbacks(self):
        # wake up agents
        for q_id, future in list(self._callbacks.items()):
            if not future.done():
                future.set_result(True)
        # Clear completed callbacks
        self._callbacks = {k: v for k, v in self._callbacks.items() if not v.done()}

    async def register_callback(self, q_id: str) -> asyncio.Future:
        # sign up to get notified when the file changes
        future = asyncio.Future()
        self._callbacks[q_id] = future
        return future

    async def unregister_callback(self, q_id: str):
        # stop caring about file changes for this question
        if q_id in self._callbacks:
            future = self._callbacks.pop(q_id)
            if not future.done():
                future.cancel()


class AskHumanServer:
    """
    The main server that handles all the magic

    This is where AI questions come in, get written to a file,
    and we wait for human answers. It's like a really slow chat app.
    """

    def __init__(self, config: AskHumanConfig):
        self.config = config
        self.ask_file = config.ask_file.resolve()  # Get absolute path
        self.pending_questions: Dict[str, float] = {}
        self.answered_questions: Set[str] = set()  # Track answered questions
        self.mcp = FastMCP("ask-human")
        self._shutdown_event = asyncio.Event()
        self._cleanup_task: Optional[asyncio.Task] = None

        # make sure we have a file to work with
        self._init_file()

        # set up the file watcher
        self.watcher = MarkdownWatcher(self.ask_file)
        self.observer = Observer()
        self.observer.schedule(self.watcher, str(self.ask_file.parent), recursive=False)

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

        self._register_tools()

    def _setup_signal_handlers(self):
        """Set up cross-platform signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self._shutdown_event.set()

        # Handle common signals across platforms
        if platform.system() != "Windows":
            # Unix-like systems
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            try:
                signal.signal(signal.SIGHUP, signal_handler)
            except AttributeError:
                pass  # SIGHUP not available on all Unix systems
        else:
            # Windows
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

    def _init_file(self):
        # set up the Q&A file if it doesn't exist yet
        if not self.ask_file.exists():
            # Ensure parent directory exists
            self.ask_file.parent.mkdir(parents=True, exist_ok=True)

            initial_content = (
                "# Ask-Human Q&A Log\n\n"
                "This file contains questions from AI agents and your responses.\n"
                "Just replace 'PENDING' with your answers and save to continue.\n\n"
            )

            if not safe_write_text(self.ask_file, initial_content):
                raise FileAccessError(f"Could not create ask file: {self.ask_file}")

        # Check file size and rotate if needed
        self._check_file_rotation()

        # Touch the file to update timestamp (cross-platform way)
        try:
            self.ask_file.touch(exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not touch file {self.ask_file}: {e}")

    def _check_file_rotation(self):
        """Check if file needs rotation and rotate if necessary."""
        if not self.ask_file.exists():
            return

        file_size = self.ask_file.stat().st_size
        if file_size > self.config.rotation_size:
            logger.info(
                f"File size ({file_size} bytes) exceeds rotation size "
                f"({self.config.rotation_size} bytes), rotating..."
            )

            # Create archive filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_name = f"{self.ask_file.stem}.{timestamp}.md"
            archive_path = self.ask_file.parent / archive_name

            try:
                # Move current file to archive
                self.ask_file.rename(archive_path)
                logger.info(f"Archived large file to {archive_path}")

                # Create new empty file
                initial_content = (
                    "# Ask-Human Q&A Log\n\n"
                    f"Previous questions archived to: {archive_name}\n\n"
                    "This file contains questions from AI agents and your responses.\n"
                    "Just replace 'PENDING' with your answers and save to continue.\n\n"
                )

                if not safe_write_text(self.ask_file, initial_content):
                    logger.error("Failed to create new file after rotation")
                    # Try to restore the original file
                    try:
                        archive_path.rename(self.ask_file)
                        logger.info("Restored original file after rotation failure")
                    except Exception as restore_e:
                        logger.error(
                            f"Failed to restore file after rotation failure: "
                            f"{restore_e}"
                        )

            except Exception as e:
                logger.error(f"Failed to rotate file: {e}")

    async def _periodic_cleanup(self):
        """Periodically clean up old questions and perform maintenance."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=self.config.cleanup_interval
                )
                # If we get here, shutdown was requested
                break
            except asyncio.TimeoutError:
                # Timeout is expected - perform cleanup
                try:
                    await self._cleanup_timeouts()
                except Exception as e:
                    logger.error(f"Error during periodic cleanup: {e}")

    def _register_tools(self):
        # register all the MCP tools that AIs can call

        @self.mcp.tool(
            description="Ask the human a question and wait for them to answer"
        )
        async def ask_human(question: str, context: str = "") -> str:
            """
            Ask the human a question and wait for response

            Args:
                question: What you actually want to know
                context: Extra info that might help (like file paths,
                         error messages, etc.)

            Returns:
                Whatever the human typed as an answer
            """
            return await self._handle_question(question, context)

        @self.mcp.tool(description="See what questions are still waiting for answers")
        async def list_pending_questions() -> str:
            # check what questions are still hanging around waiting for answers
            return await self._list_pending()

        @self.mcp.tool(
            description="Get some stats about how many questions have been asked"
        )
        async def get_qa_stats() -> str:
            # get some basic stats about the Q&A session
            return await self._get_stats()

    async def _handle_question(self, question: str, context: str) -> str:
        # Validate inputs
        question = validate_input(question, self.config.max_question_length, "Question")
        context = validate_input(context, self.config.max_context_length, "Context")

        # Check resource limits
        if len(self.pending_questions) >= self.config.max_pending_questions:
            raise AskHumanError(
                f"Too many pending questions ({len(self.pending_questions)}). "
                f"Please answer some questions before asking new ones."
            )

        # Check file size
        if (
            self.ask_file.exists()
            and self.ask_file.stat().st_size > self.config.max_file_size
        ):
            raise FileAccessError(
                f"Ask file too large ({self.ask_file.stat().st_size} bytes). "
                f"Please archive or clean up the file."
            )

        # handle when an AI asks us something
        q_id = f"Q{uuid.uuid4().hex[:8]}"  # generate a random question ID
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # clean up any old questions that timed out
        await self._cleanup_timeouts()
        try:
            # Register callback first, then write question
            change_future = await self.watcher.register_callback(q_id)

            # Write the question to the file
            if not self._append_question(q_id, question, context, timestamp):
                await self.watcher.unregister_callback(q_id)
                raise FileAccessError(
                    f"Couldn't write question {q_id} to {self.ask_file}"
                )

            # Check if answer already exists (race condition protection)
            existing_answer = self._read_answer(q_id)
            if existing_answer:
                await self.watcher.unregister_callback(q_id)
                logger.info(f"✅ Found existing answer for {q_id}")
                return existing_answer

            # keep track of this question
            self.pending_questions[q_id] = time.time()

            # Let the human know there's a new question
            log_safe_question = question.replace("\n", "\\n")[:100]
            log_safe_context = context.replace("\n", "\\n")[:100]

            logger.info(f"🚨 New question: {q_id}")
            logger.info(
                f"   Question: {log_safe_question}"
                f"{'...' if len(question) > 100 else ''}"
            )
            logger.info(
                f"   Context: {log_safe_context}{'...' if len(context) > 100 else ''}"
            )
            logger.info(f"   Edit {self.ask_file} and replace PENDING with your answer")

            try:
                start_time = time.time()

                # Keep checking for an answer until timeout
                while time.time() - start_time < self.config.timeout:
                    # Check for shutdown signal
                    if self._shutdown_event.is_set():
                        raise RuntimeError("Server is shutting down")

                    # check if we got an answer
                    answer = self._read_answer(q_id)
                    if answer:
                        # CRITICAL FIX: Mark question as answered and clean up
                        self.answered_questions.add(q_id)
                        self.pending_questions.pop(q_id, None)
                        logger.info(f"✅ Got an answer for {q_id}")
                        return answer

                    # wait for the file to change (or timeout)
                    try:
                        await asyncio.wait_for(change_future, timeout=5.0)
                        # File changed! Register for the next notification
                        change_future = await self.watcher.register_callback(q_id)
                    except asyncio.TimeoutError:
                        # Just a periodic check - keep going
                        continue

                # We ran out of time :(
                logger.warning(
                    f"⏰ Question {q_id} timed out after {self.config.timeout} seconds"
                )
                raise QuestionTimeoutError(
                    f"No answer received for question {q_id} within "
                    f"{self.config.timeout} seconds. "
                    f"Please check {self.ask_file} and provide an answer."
                )

            finally:
                # Clean up after ourselves
                self.pending_questions.pop(q_id, None)
                await self.watcher.unregister_callback(q_id)

        except Exception:
            # Ensure cleanup on any error
            self.pending_questions.pop(q_id, None)
            await self.watcher.unregister_callback(q_id)
            raise

    def _append_question(
        self, q_id: str, question: str, context: str, timestamp: str
    ) -> bool:
        """Add a new question to the markdown file with file locking"""
        try:
            with CrossPlatformFileLock(self.ask_file):
                # Read existing content
                existing_content = safe_read_text(self.ask_file) or ""

                # Append new question
                new_content = existing_content + (
                    f"\n---\n\n### {q_id}\n\n"
                    f"**Timestamp:** {timestamp}  \n"
                    f"**Question:** {question}  \n"
                    f"**Context:** {context}  \n"
                    f"**Answer:** PENDING\n\n"
                )

                return safe_write_text(self.ask_file, new_content)

        except Exception as e:
            logger.error(f"Failed to write question {q_id}: {e}")
            return False

    def _read_answer(self, q_id: str) -> Optional[str]:
        """Read the answer for a question ID from the markdown file
        using robust parsing"""
        try:
            content = safe_read_text(self.ask_file)
            if not content:
                return None

            # Use regex for more robust parsing
            # Look for the question section and extract the answer
            pattern = (
                rf"### {re.escape(q_id)}\s*\n.*?\*\*Answer:\*\*\s*(.+?)"
                r"(?=\n\n---|\n\n###|$)"
            )
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

            if match:
                answer = match.group(1).strip()
                # Check if it's still pending (case-insensitive)
                if answer.lower() == "pending":
                    return None
                return answer

            return None

        except Exception as e:
            logger.error(f"Failed to read answer for {q_id}: {e}")
            return None

    async def _cleanup_timeouts(self):
        """Remove questions that have timed out"""
        current_time = time.time()
        timed_out = []

        for q_id, start_time in self.pending_questions.items():
            if current_time - start_time > self.config.timeout:
                timed_out.append(q_id)

        for q_id in timed_out:
            self.pending_questions.pop(q_id, None)
            await self.watcher.unregister_callback(q_id)
            logger.warning(f"⏰ Cleaned up timed out question: {q_id}")

    async def _list_pending(self) -> str:
        """show what questions are still waiting for answers"""
        if not self.pending_questions:
            return "No pending questions."

        lines = ["📋 Pending Questions:\n"]

        for q_id, start_time in self.pending_questions.items():
            elapsed = int(time.time() - start_time)
            question_text = self._get_question_text(q_id)

            if question_text:
                # Truncate long questions for display
                display_text = (
                    question_text[:100] + "..."
                    if len(question_text) > 100
                    else question_text
                )
                lines.append(f"• {q_id}: {display_text} (waiting {elapsed}s)")
            else:
                lines.append(
                    f"• {q_id}: [Question text not found] (waiting {elapsed}s)"
                )

        lines.append(f"\n💡 Edit {self.ask_file} to provide answers")
        return "\n".join(lines)

    def _get_question_text(self, q_id: str) -> Optional[str]:
        """get the actual question text for a given question ID using robust parsing"""
        try:
            content = safe_read_text(self.ask_file)
            if not content:
                return None

            # Use regex for more robust parsing
            pattern = (
                rf"### {re.escape(q_id)}\s*\n.*?\*\*Question:\*\*\s*(.+?)"
                r"(?=\s*\*\*Context|\s*\*\*Answer|\n\n)"
            )
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

            if match:
                return match.group(1).strip()

            return None

        except Exception as e:
            logger.error(f"Failed to get question text for {q_id}: {e}")
            return None

    async def _get_stats(self) -> str:
        """show some basic stats about how things are going"""
        try:
            content = safe_read_text(self.ask_file) or ""

            # Count total questions by looking for question IDs
            question_pattern = r"### Q[a-f0-9]{8}"
            total_questions = len(re.findall(question_pattern, content, re.IGNORECASE))

            # Count answered questions (not PENDING)
            # Extract all answers and filter out PENDING ones
            answer_pattern = r"\*\*Answer:\*\*\s*([^\n]+)"
            all_answers = re.findall(answer_pattern, content, re.IGNORECASE)
            answered_questions = len([
                a for a in all_answers if a.strip().upper() != "PENDING"
            ])

            pending_count = len(self.pending_questions)

            # Calculate success rate
            if total_questions > 0:
                success_rate = (answered_questions / total_questions) * 100
            else:
                success_rate = 0

            file_size = self.ask_file.stat().st_size if self.ask_file.exists() else 0

            rotation_mb = self.config.rotation_size / 1024 / 1024

            stats = f"""📊 Ask-Human MCP Statistics

Questions:
• Total: {total_questions}
• Currently Pending: {pending_count}
• Answered: {answered_questions}
• Success Rate: {success_rate:.1f}%

File:
• Path: {self.ask_file}
• Size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)
• Rotation at: {self.config.rotation_size:,} bytes ({rotation_mb:.1f} MB)

Limits:
• Max Question Length: {self.config.max_question_length:,} chars
• Max Context Length: {self.config.max_context_length:,} chars
• Max Pending: {self.config.max_pending_questions}
• Timeout: {self.config.timeout}s ({self.config.timeout // 60}min)

Platform: {platform.system()} {platform.release()}
"""

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return "Failed to get statistics"

    def start_watching(self):
        """start watching the file for changes"""
        # Set the event loop for the watcher
        try:
            loop = asyncio.get_running_loop()
            self.watcher.set_event_loop(loop)
        except RuntimeError:
            logger.warning("No running event loop found when starting file watcher")
        
        self.observer.start()

        # Start periodic cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

        logger.info(f"📁 Watching {self.ask_file} for changes")

    def stop_watching(self):
        """stop watching the file"""
        # Stop periodic cleanup
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join(timeout=5.0)  # Don't wait forever

        logger.info("👋 Stopped watching file")

    def get_fastapi_app(self) -> FastAPI:
        """Get FastAPI app for HTTP mode"""
        app = FastAPI(title="Ask-Human MCP Server")

        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "pending_questions": len(self.pending_questions),
                "file_path": str(self.ask_file),
                "file_exists": self.ask_file.exists(),
            }

        @app.get("/sse")
        async def sse_endpoint(request: Request):
            return StreamingResponse(
                self._sse_handler(request), media_type="text/plain"
            )

        return app

    async def _sse_handler(self, request: Request):
        """Handle Server-Sent Events for MCP communication."""
        logger.info("🔗 New MCP SSE connection")

        # Send initial connection message
        yield f"event: mcp_ready\ndata: {json.dumps({'status': 'connected'})}\n\n"

        try:
            # Handle incoming MCP messages
            async for line in request.stream():
                # Process MCP protocol messages here
                # This is a basic implementation - you may need to adapt
                # based on your specific MCP client requirements
                logger.debug(f"Received MCP message: {line}")

        except Exception as e:
            logger.error(f"SSE connection error: {e}")
        finally:
            logger.info("🔗 MCP SSE connection closed")


def create_arg_parser() -> argparse.ArgumentParser:
    """set up command line arguments"""
    parser = argparse.ArgumentParser(
        description="Ask-Human MCP Server - Let AI agents escalate questions to humans"
    )

    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=get_default_ask_file(),
        help=f"Path to Q&A markdown file (default: {get_default_ask_file()})",
    )

    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=1800,
        help="Question timeout in seconds (default: 1800 = 30min)",
    )

    parser.add_argument(
        "--max-question-length",
        type=int,
        default=10240,
        help="Maximum question length in characters (default: 10240)",
    )

    parser.add_argument(
        "--max-context-length",
        type=int,
        default=51200,
        help="Maximum context length in characters (default: 51200)",
    )

    parser.add_argument(
        "--max-pending",
        type=int,
        default=100,
        help="Maximum number of pending questions (default: 100)",
    )

    parser.add_argument(
        "--max-file-size",
        type=int,
        default=104857600,
        help="Maximum file size in bytes (default: 104857600 = 100MB)",
    )

    parser.add_argument(
        "--rotation-size",
        type=int,
        default=52428800,
        help="File size at which to rotate in bytes (default: 52428800 = 50MB)",
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="Run HTTP server on this port (default: stdio mode)",
    )

    parser.add_argument(
        "--host", default="127.0.0.1", help="HTTP server host (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser


async def run_stdio_mode(server: AskHumanServer):
    """run in stdio mode (the normal MCP way)"""
    logger.info("🚀 Starting Ask-Human MCP Server in stdio mode")
    server.start_watching()

    try:
        await server.mcp.run_async()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        server.stop_watching()


async def run_http_mode(server: AskHumanServer, host: str, port: int):
    """run in HTTP mode"""
    logger.info(f"🚀 Starting Ask-Human MCP Server on http://{host}:{port}")
    server.start_watching()

    app = server.get_fastapi_app()

    try:
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server_instance = uvicorn.Server(config)
        await server_instance.serve()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        server.stop_watching()


async def async_main():
    """async main entry point"""
    parser = create_arg_parser()
    args = parser.parse_args()

    # turn on debug mode if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create configuration
    config = AskHumanConfig(
        ask_file=args.file.resolve(),
        timeout=args.timeout,
        max_question_length=args.max_question_length,
        max_context_length=args.max_context_length,
        max_pending_questions=args.max_pending,
        max_file_size=args.max_file_size,
        rotation_size=args.rotation_size,
    )

    # Create the server
    server = AskHumanServer(config)

    logger.info(f"📝 Q&A file: {server.ask_file}")
    logger.info(f"⏰ Timeout: {args.timeout}s")
    logger.info(f"🖥️  Platform: {platform.system()} {platform.release()}")
    logger.info(f"📏 Question limit: {config.max_question_length} chars")
    logger.info(f"📄 Context limit: {config.max_context_length} chars")
    logger.info(f"📊 Max pending: {config.max_pending_questions}")

    try:
        if args.port:
            await run_http_mode(server, args.host, args.port)
        else:
            await run_stdio_mode(server)
    except KeyboardInterrupt:
        logger.info("👋 Shutting down...")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


def main():
    """main entry point for console script"""
    try:
        # mcp library runs and event loop so this runs in a nested event loop context fyi
        logger.debug("Starting async_main with nest_asyncio support")
        return asyncio.run(async_main())
            
    except KeyboardInterrupt:
        logger.info("👋 Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
