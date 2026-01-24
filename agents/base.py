"""
Agent Base Class - Foundation for Agentic Architecture

Provides:
- Standardized agent interface with run/validate/cleanup lifecycle
- Built-in timeout handling with configurable limits
- Retry logic with exponential backoff
- Progress callbacks for real-time UI updates
- Error isolation and graceful degradation
- State management for checkpoint/resume capability
"""
import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
import traceback


class AgentStatus(Enum):
    """Status states for an agent."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class AgentResult:
    """Standardized result from any agent."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: int = 0
    retries_used: int = 0


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    timeout_seconds: float = 120.0  # Default 2 minute timeout
    max_retries: int = 0  # No retries by default
    retry_delay_seconds: float = 1.0  # Initial retry delay
    retry_backoff_multiplier: float = 2.0  # Exponential backoff
    required: bool = True  # If False, failure doesn't stop pipeline


class PipelineContext:
    """
    Shared context passed between agents in the pipeline.
    Accumulates results and provides access to previous stage outputs.
    """

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = initial_data or {}
        self._stage_results: Dict[str, AgentResult] = {}
        self._metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "stages_completed": []
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in context."""
        self._data[key] = value

    def update(self, data: Dict[str, Any]) -> None:
        """Update context with multiple values."""
        self._data.update(data)

    def get_stage_result(self, stage_name: str) -> Optional[AgentResult]:
        """Get result from a specific stage."""
        return self._stage_results.get(stage_name)

    def set_stage_result(self, stage_name: str, result: AgentResult) -> None:
        """Store result from a stage."""
        self._stage_results[stage_name] = result
        if result.success:
            self._metadata["stages_completed"].append(stage_name)

    @property
    def data(self) -> Dict[str, Any]:
        """Access raw data dict."""
        return self._data

    @property
    def metadata(self) -> Dict[str, Any]:
        """Access metadata."""
        return self._metadata

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dict for checkpointing."""
        return {
            "data": self._data,
            "metadata": self._metadata,
            "stage_results": {
                name: {
                    "success": r.success,
                    "error": r.error,
                    "metadata": r.metadata
                }
                for name, r in self._stage_results.items()
            }
        }


# Type for progress callback: (agent_name, status, message, progress_pct)
ProgressCallback = Callable[[str, AgentStatus, str, float], None]


class AgentBase(ABC):
    """
    Abstract base class for all pipeline agents.

    Subclasses must implement:
    - execute(context): The actual agent logic
    - validate_input(context): Verify required inputs exist

    Optional overrides:
    - cleanup(context, result): Post-execution cleanup
    - get_progress_weight(): Relative weight for progress calculation
    """

    def __init__(
        self,
        name: str,
        config: Optional[AgentConfig] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        self.name = name
        self.config = config or AgentConfig()
        self.progress_callback = progress_callback
        self._status = AgentStatus.PENDING
        self._start_time: Optional[float] = None

    @property
    def status(self) -> AgentStatus:
        """Current agent status."""
        return self._status

    def _update_status(self, status: AgentStatus, message: str = "", progress: float = 0.0):
        """Update status and notify callback."""
        self._status = status
        if self.progress_callback:
            self.progress_callback(self.name, status, message, progress)

    @abstractmethod
    def validate_input(self, context: PipelineContext) -> Optional[str]:
        """
        Validate that required inputs are present in context.

        Returns:
            None if valid, error message string if invalid
        """
        pass

    @abstractmethod
    def execute(self, context: PipelineContext) -> AgentResult:
        """
        Execute the agent's core logic.

        This is called by run() after validation and handles the actual work.
        Should NOT handle timeouts or retries - that's done by run().

        Args:
            context: Pipeline context with inputs from previous stages

        Returns:
            AgentResult with success/failure and any output data
        """
        pass

    def cleanup(self, context: PipelineContext, result: AgentResult) -> None:
        """
        Optional cleanup after execution.
        Called regardless of success/failure.
        """
        pass

    def get_progress_weight(self) -> float:
        """
        Relative weight for progress calculation.
        Override to indicate heavier/lighter agents.
        Default is 1.0.
        """
        return 1.0

    def run(self, context: PipelineContext) -> AgentResult:
        """
        Run the agent with full lifecycle management.

        Handles:
        - Input validation
        - Timeout enforcement
        - Retry logic with backoff
        - Progress callbacks
        - Error isolation
        - Cleanup

        Args:
            context: Pipeline context

        Returns:
            AgentResult with outcome
        """
        self._start_time = time.time()
        retries_used = 0
        last_error = None

        # Validate input
        self._update_status(AgentStatus.RUNNING, "Validating inputs...", 0.0)
        validation_error = self.validate_input(context)
        if validation_error:
            result = AgentResult(
                success=False,
                error=f"Input validation failed: {validation_error}",
                error_type="ValidationError",
                execution_time_ms=self._elapsed_ms()
            )
            self._update_status(AgentStatus.FAILED, validation_error, 0.0)
            return result

        # Execute with retries
        current_delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries + 1):
            if attempt > 0:
                retries_used = attempt
                self._update_status(
                    AgentStatus.RETRYING,
                    f"Retry {attempt}/{self.config.max_retries}...",
                    0.0
                )
                time.sleep(current_delay)
                current_delay *= self.config.retry_backoff_multiplier

            try:
                self._update_status(AgentStatus.RUNNING, "Executing...", 0.1)

                # Execute with timeout
                result = self._execute_with_timeout(context)
                result.retries_used = retries_used
                result.execution_time_ms = self._elapsed_ms()

                if result.success:
                    self._update_status(AgentStatus.SUCCESS, "Complete", 1.0)
                    self.cleanup(context, result)
                    return result
                else:
                    last_error = result.error
                    # Don't retry on validation/logic errors, only on transient failures
                    if result.error_type not in ["TransientError", "TimeoutError"]:
                        break

            except TimeoutError:
                last_error = f"Agent timed out after {self.config.timeout_seconds}s"
                result = AgentResult(
                    success=False,
                    error=last_error,
                    error_type="TimeoutError",
                    execution_time_ms=self._elapsed_ms(),
                    retries_used=retries_used
                )
                self._update_status(AgentStatus.TIMEOUT, last_error, 0.0)
                # Timeouts can be retried

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                result = AgentResult(
                    success=False,
                    error=last_error,
                    error_type=type(e).__name__,
                    metadata={"traceback": traceback.format_exc()},
                    execution_time_ms=self._elapsed_ms(),
                    retries_used=retries_used
                )
                # Unexpected exceptions generally shouldn't be retried
                break

        # All retries exhausted
        final_result = AgentResult(
            success=False,
            error=last_error or "Unknown error",
            error_type=result.error_type if 'result' in locals() else "UnknownError",
            execution_time_ms=self._elapsed_ms(),
            retries_used=retries_used
        )
        self._update_status(AgentStatus.FAILED, last_error or "Failed", 0.0)
        self.cleanup(context, final_result)
        return final_result

    def _execute_with_timeout(self, context: PipelineContext) -> AgentResult:
        """Execute with timeout enforcement.

        Note: We avoid ThreadPoolExecutor because Streamlit's context (session state,
        secrets, etc.) doesn't transfer to new threads, causing NoSessionContext errors.
        Instead, we run directly and rely on well-behaved agents to respect reasonable
        execution times. For true timeout enforcement, consider using signal-based
        timeouts on Unix or multiprocessing (with serializable context).
        """
        # Run directly without threading to preserve Streamlit context
        # The timeout is advisory - agents should be designed to complete reasonably
        return self.execute(context)

    def _elapsed_ms(self) -> int:
        """Get elapsed time since start in milliseconds."""
        if self._start_time is None:
            return 0
        return int((time.time() - self._start_time) * 1000)


class AsyncAgentBase(AgentBase):
    """
    Async version of AgentBase for agents that benefit from async I/O.
    Useful for LLM calls, network requests, etc.
    """

    @abstractmethod
    async def execute_async(self, context: PipelineContext) -> AgentResult:
        """Async version of execute."""
        pass

    def execute(self, context: PipelineContext) -> AgentResult:
        """Sync wrapper that runs the async execute."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.execute_async(context))
        finally:
            loop.close()

    async def run_async(self, context: PipelineContext) -> AgentResult:
        """
        Async version of run with full lifecycle management.
        """
        self._start_time = time.time()
        retries_used = 0
        last_error = None

        # Validate input
        self._update_status(AgentStatus.RUNNING, "Validating inputs...", 0.0)
        validation_error = self.validate_input(context)
        if validation_error:
            result = AgentResult(
                success=False,
                error=f"Input validation failed: {validation_error}",
                error_type="ValidationError",
                execution_time_ms=self._elapsed_ms()
            )
            self._update_status(AgentStatus.FAILED, validation_error, 0.0)
            return result

        # Execute with retries
        current_delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries + 1):
            if attempt > 0:
                retries_used = attempt
                self._update_status(
                    AgentStatus.RETRYING,
                    f"Retry {attempt}/{self.config.max_retries}...",
                    0.0
                )
                await asyncio.sleep(current_delay)
                current_delay *= self.config.retry_backoff_multiplier

            try:
                self._update_status(AgentStatus.RUNNING, "Executing...", 0.1)

                # Execute with timeout
                result = await asyncio.wait_for(
                    self.execute_async(context),
                    timeout=self.config.timeout_seconds
                )
                result.retries_used = retries_used
                result.execution_time_ms = self._elapsed_ms()

                if result.success:
                    self._update_status(AgentStatus.SUCCESS, "Complete", 1.0)
                    self.cleanup(context, result)
                    return result
                else:
                    last_error = result.error
                    if result.error_type not in ["TransientError", "TimeoutError"]:
                        break

            except asyncio.TimeoutError:
                last_error = f"Agent timed out after {self.config.timeout_seconds}s"
                self._update_status(AgentStatus.TIMEOUT, last_error, 0.0)

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                break

        final_result = AgentResult(
            success=False,
            error=last_error or "Unknown error",
            execution_time_ms=self._elapsed_ms(),
            retries_used=retries_used
        )
        self._update_status(AgentStatus.FAILED, last_error or "Failed", 0.0)
        self.cleanup(context, final_result)
        return final_result
