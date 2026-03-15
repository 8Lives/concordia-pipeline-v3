"""
LLM Service - Claude API Integration

Provides a unified interface for LLM operations in the pipeline:
- Value resolution (mapping ambiguous values to standard terms)
- Data review (validating harmonized output)
- Decision making (orchestration choices)

Usage:
    from llm.service import LLMService

    llm = LLMService(api_key="sk-ant-...")

    # Resolve ambiguous value
    result = llm.resolve_value(
        variable="RACE",
        value="Caucasian/White",
        valid_values=["White", "Black or African American", "Asian"],
        context="Clinical trial demographics"
    )
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class LLMModel(Enum):
    """Available Claude models."""
    OPUS = "claude-opus-4-6"
    SONNET = "claude-sonnet-4-6"
    HAIKU = "claude-haiku-4-5-20251001"


@dataclass
class LLMResponse:
    """Response from LLM call."""
    success: bool
    content: str
    parsed_data: Optional[Dict[str, Any]] = None
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def input_tokens(self) -> int:
        return self.usage.get("input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return self.usage.get("output_tokens", 0)


class LLMService:
    """
    Service for Claude API interactions.

    Handles:
    - API authentication
    - Request formatting
    - Response parsing
    - Error handling with retries
    - Token usage tracking
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = LLMModel.SONNET.value,
        max_retries: int = 2,
        timeout: float = 30.0
    ):
        """
        Initialize the LLM service.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            default_model: Default model to use
            max_retries: Max retry attempts for failed calls
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.default_model = default_model
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self._total_usage = {"input_tokens": 0, "output_tokens": 0}

        # Load API key from environment if not provided
        if not self.api_key:
            import os
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def _get_client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key not configured. "
                    "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
                )
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Anthropic client initialized")
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Run: pip install anthropic"
                )
        return self._client

    @property
    def is_configured(self) -> bool:
        """Check if the service is properly configured."""
        return bool(self.api_key)

    @property
    def model(self) -> str:
        """Get the default model name."""
        return self.default_model

    @property
    def total_usage(self) -> Dict[str, int]:
        """Get total token usage across all calls."""
        return self._total_usage.copy()

    @property
    def total_tokens_used(self) -> int:
        """Get total tokens used across all calls."""
        return self._total_usage["input_tokens"] + self._total_usage["output_tokens"]

    def call(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False
    ) -> LLMResponse:
        """
        Make a call to Claude API.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            model: Model override (uses default if not specified)
            max_tokens: Max response tokens
            temperature: Sampling temperature (0.0 for deterministic)
            json_mode: If True, expect JSON response

        Returns:
            LLMResponse with result or error
        """
        model = model or self.default_model

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Add JSON instruction if needed
        if json_mode and system:
            system = f"{system}\n\nRespond with valid JSON only, no markdown formatting."
        elif json_mode:
            system = "Respond with valid JSON only, no markdown formatting."

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                client = self._get_client()

                # Make API call
                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages,
                }
                if system:
                    kwargs["system"] = system

                response = client.messages.create(**kwargs)

                # Extract content
                content = response.content[0].text if response.content else ""

                # Track usage
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
                self._total_usage["input_tokens"] += usage["input_tokens"]
                self._total_usage["output_tokens"] += usage["output_tokens"]

                # Parse JSON if requested
                parsed_data = None
                if json_mode:
                    try:
                        # Handle potential markdown code blocks
                        json_content = content.strip()
                        if json_content.startswith("```"):
                            lines = json_content.split("\n")
                            json_content = "\n".join(lines[1:-1])
                        parsed_data = json.loads(json_content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON response: {e}")

                return LLMResponse(
                    success=True,
                    content=content,
                    parsed_data=parsed_data,
                    model=model,
                    usage=usage
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    import time
                    time.sleep(1.0 * (attempt + 1))  # Exponential backoff

        return LLMResponse(
            success=False,
            content="",
            error=last_error,
            model=model
        )

    def resolve_value(
        self,
        variable: str,
        value: str,
        valid_values: List[str],
        context: Optional[str] = None,
        spec_reference: Optional[str] = None
    ) -> LLMResponse:
        """
        Resolve an ambiguous value to a standard term.

        Args:
            variable: Variable name (e.g., "RACE", "SEX")
            value: The ambiguous value to resolve
            valid_values: List of valid standard values
            context: Additional context about the data
            spec_reference: Reference to specification rule

        Returns:
            LLMResponse with resolved value in parsed_data["resolved_value"]
        """
        system = """You are a clinical data standardization expert. Your task is to map
source data values to standardized CDISC SDTM terminology.

Rules:
1. Only map to values in the provided valid_values list
2. If the value clearly matches one option, return that option
3. If the value is ambiguous but can be reasonably inferred, make the mapping
4. If the value cannot be mapped with confidence, return null
5. Provide your reasoning"""

        prompt = f"""Map the following value to a standardized term.

Variable: {variable}
Source Value: "{value}"
Valid Values: {json.dumps(valid_values)}
{f'Context: {context}' if context else ''}
{f'Specification: {spec_reference}' if spec_reference else ''}

Respond in JSON format:
{{
    "resolved_value": "<matched value or null if unmappable>",
    "confidence": "<high|medium|low>",
    "reasoning": "<brief explanation>"
}}"""

        return self.call(prompt, system=system, json_mode=True, temperature=0.0)

    def review_harmonized_data(
        self,
        data_sample: List[Dict[str, Any]],
        variable_rules: Dict[str, str],
        qc_issues: Optional[List[Dict[str, Any]]] = None,
        column_stats: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> LLMResponse:
        """
        Review harmonized data for quality and compliance using STOPLIGHT grading.

        Args:
            data_sample: Sample rows from harmonized data
            variable_rules: Rules for each variable from spec
            qc_issues: Any QC issues already identified
            column_stats: Column statistics (presence, completeness)

        Returns:
            LLMResponse with review findings in parsed_data including stoplight grade
        """
        # Use the new stoplight grading rules
        system = """You are a senior clinical data manager reviewing harmonized trial data.

IMPORTANT: Ignore all SDTM requirements for this review. Use the following stoplight grading rules:

## STOPLIGHT GRADING RULES:

**GREEN** - All of the following 5 core variables are present and populated:
1. SEX
2. RACE
3. ETHNIC
4. AGE or AGEGP (at least one must be present)
5. COUNTRY

**YELLOW** - Missing no more than 2 of the 5 core variables, OR flagged formatting issues with any of the core variables

**RED** - Missing 3 or more of the 5 core variables

## Your Review Should:
1. Count which core variables are present/missing
2. Check for formatting issues in core variables
3. Assign the appropriate stoplight color based on the rules above
4. Provide actionable recommendations for any issues found

Focus ONLY on the 5 core variables (SEX, RACE, ETHNIC, AGE/AGEGP, COUNTRY). Other variables should not affect the stoplight grade."""

        prompt = f"""Review this harmonized Demographics (DM) dataset using the STOPLIGHT GRADING RULES.

## Data Sample (5 rows):
```json
{json.dumps(data_sample[:5], indent=2, default=str)}
```

## Column Statistics (check for presence and completeness of core variables):
```json
{json.dumps(column_stats or {}, indent=2, default=str)}
```

## Pre-identified QC Issues ({len(qc_issues or [])} total):
```json
{json.dumps((qc_issues or [])[:10], indent=2, default=str)}
```

## Your Task:
1. Check which of the 5 CORE variables are present: SEX, RACE, ETHNIC, AGE/AGEGP, COUNTRY
2. Identify any formatting issues with core variables
3. Apply the STOPLIGHT rules to determine the grade (GREEN/YELLOW/RED)

Provide your review in JSON:
{{
    "stoplight": "GREEN|YELLOW|RED",
    "core_variables_present": ["list variables that are present and properly populated"],
    "core_variables_missing": ["list variables that are missing or empty"],
    "core_variables_count": <number of core variables present out of 5>,
    "formatting_issues": ["list any formatting issues with core variables"],
    "overall_quality": "good|acceptable|needs_attention|poor",
    "critical_issues": [
        {{"issue": "description", "severity": "critical|high|medium", "recommendation": "fix"}}
    ],
    "approval": "GREEN|YELLOW|RED",
    "reason": "Brief explanation of why this stoplight grade was assigned",
    "recommendations": ["actionable recommendations for improvement"]
}}"""

        return self.call(
            prompt,
            system=system,
            json_mode=True,
            max_tokens=2048,
            temperature=0.0
        )

    def decide_next_action(
        self,
        current_state: Dict[str, Any],
        available_actions: List[str],
        context: str
    ) -> LLMResponse:
        """
        Make an orchestration decision about next action.

        Args:
            current_state: Current pipeline state
            available_actions: List of possible actions
            context: Description of the decision context

        Returns:
            LLMResponse with decision in parsed_data["action"]
        """
        system = """You are a clinical data pipeline orchestrator. Based on the current
state and available actions, decide the best next step.

Consider:
1. Data quality requirements
2. Efficiency (avoid unnecessary steps)
3. Error recovery (handle failures gracefully)
4. Completeness (ensure all required steps are done)"""

        prompt = f"""Decide the next action for the harmonization pipeline.

## Current State:
{json.dumps(current_state, indent=2, default=str)}

## Available Actions:
{json.dumps(available_actions)}

## Context:
{context}

Respond in JSON format:
{{
    "action": "<chosen action from available_actions>",
    "reasoning": "<why this action>",
    "confidence": "<high|medium|low>"
}}"""

        return self.call(prompt, system=system, json_mode=True, temperature=0.0)
