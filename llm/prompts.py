"""
Prompt Templates for LLM Operations

Centralized prompt management for consistent LLM interactions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json


@dataclass
class PromptTemplates:
    """Collection of prompt templates for pipeline operations."""

    @staticmethod
    def value_resolution_system() -> str:
        """System prompt for value resolution tasks."""
        return """You are a clinical data standardization expert specializing in CDISC SDTM.

Your task is to map source data values to standardized terminology. You must:
1. Only use values from the provided valid_values list
2. Apply clinical domain knowledge to interpret abbreviations and variations
3. Handle common data quality issues (typos, case variations, legacy codes)
4. Return null if a value truly cannot be mapped with reasonable confidence

Common mappings you should know:
- SEX: M/Male/1 → Male, F/Female/2 → Female, U/Unknown/UNK → Unknown
- RACE: Various spellings/codes to standard CDISC race categories
- ETHNIC: Hispanic variations, Not Hispanic variations
- Country codes: ISO 2/3 letter codes to full names"""

    @staticmethod
    def value_resolution_prompt(
        variable: str,
        value: str,
        valid_values: List[str],
        context: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate prompt for value resolution."""
        prompt = f"""Map this value to standardized terminology.

Variable: {variable}
Source Value: "{value}"
Valid Values: {json.dumps(valid_values)}"""

        if context:
            prompt += f"\nContext: {context}"

        if examples:
            prompt += "\n\nExamples of correct mappings:"
            for ex in examples:
                prompt += f"\n- \"{ex['source']}\" → \"{ex['target']}\""

        prompt += """

Respond in JSON:
{
    "resolved_value": "<value from valid_values or null>",
    "confidence": "high|medium|low",
    "reasoning": "<brief explanation>"
}"""
        return prompt

    @staticmethod
    def review_system() -> str:
        """System prompt for data review tasks."""
        return """You are a senior clinical data manager reviewing harmonized trial data.

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

    @staticmethod
    def review_prompt(
        data_sample: List[Dict[str, Any]],
        column_stats: Dict[str, Dict[str, Any]],
        qc_issues: List[Dict[str, Any]],
        spec_summary: str
    ) -> str:
        """Generate prompt for data review."""
        return f"""Review this harmonized Demographics (DM) dataset using the STOPLIGHT GRADING RULES.

## Data Sample (5 rows):
```json
{json.dumps(data_sample[:5], indent=2, default=str)}
```

## Column Statistics (check for presence and completeness of core variables):
```json
{json.dumps(column_stats, indent=2, default=str)}
```

## Pre-identified QC Issues ({len(qc_issues)} total):
```json
{json.dumps(qc_issues[:10], indent=2, default=str)}
```
{f"... and {len(qc_issues) - 10} more issues" if len(qc_issues) > 10 else ""}

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

    @staticmethod
    def orchestration_system() -> str:
        """System prompt for orchestration decisions."""
        return """You are an intelligent pipeline orchestrator for clinical data harmonization.

Your role is to:
1. Analyze the current pipeline state
2. Determine the optimal next action
3. Handle errors and edge cases gracefully
4. Ensure data quality at each step

Pipeline stages: ingest → map → harmonize → qc → review → export

Decision factors:
- Previous stage success/failure
- Data quality indicators
- Configuration options
- Error recovery needs"""

    @staticmethod
    def orchestration_prompt(
        stage_results: Dict[str, Dict[str, Any]],
        current_stage: str,
        available_actions: List[str],
        error_context: Optional[str] = None
    ) -> str:
        """Generate prompt for orchestration decision."""
        prompt = f"""Determine the next pipeline action.

## Stage Results:
```json
{json.dumps(stage_results, indent=2, default=str)}
```

## Current Stage: {current_stage}
## Available Actions: {json.dumps(available_actions)}
"""

        if error_context:
            prompt += f"\n## Error Context:\n{error_context}\n"

        prompt += """
Decide the next action in JSON:
{
    "action": "<action from available_actions>",
    "reasoning": "<why this is the right choice>",
    "fallback_action": "<backup if primary fails>",
    "warnings": ["any concerns about this path"]
}"""
        return prompt

    @staticmethod
    def batch_resolution_prompt(
        variable: str,
        values: List[str],
        valid_values: List[str]
    ) -> str:
        """Generate prompt for batch value resolution (more efficient)."""
        return f"""Map these source values to standardized terminology.

Variable: {variable}
Valid Values: {json.dumps(valid_values)}

Source Values to Map:
{json.dumps(values)}

For each value, provide the mapping in JSON array format:
[
    {{"source": "<original>", "resolved": "<mapped value or null>", "confidence": "high|medium|low"}},
    ...
]

Only include values that need mapping (skip obvious exact matches)."""
