from __future__ import annotations
from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field


class DateRange(BaseModel):
    start: str = Field(description="Start date in YYYY-MM-DD format")
    end: str = Field(description="End date in YYYY-MM-DD format")


class OrchestratorOutput(BaseModel):
    report_type: Literal["weekly", "monthly", "adhoc"] = Field(
        description="Type of report to generate"
    )
    fraud_pillar: str = Field(
        description=(
            "Fraud pillar to analyze. One of: fraud_loss, promo_abuse, coin2dd, "
            "appid_breakdown, general"
        )
    )
    date_range: DateRange = Field(description="Date range for the analysis")
    reasoning: str = Field(description="Brief explanation of chosen report type and pillar")


class Finding(BaseModel):
    title: str = Field(description="Short title of the finding")
    finding: str = Field(description="Detailed description with specific numbers")
    evidence: List[str] = Field(description="Supporting data points and metrics")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Severity level"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    pillar: str = Field(description="Fraud pillar this finding relates to")
    recommended_action: str = Field(description="Specific operational action to take")


class FraudAnalysisOutput(BaseModel):
    findings: List[Finding] = Field(description="List of identified fraud findings")
    overall_risk_level: Literal["low", "medium", "high", "critical"] = Field(
        description="Overall risk level for the period"
    )
    summary: str = Field(description="One-paragraph executive summary of fraud analysis")


class ValidationIssue(BaseModel):
    issue: str = Field(description="Description of the validation problem")
    severity: Literal["minor", "major", "blocking"] = Field(description="Issue severity")
    suggested_fix: str = Field(description="How to address this issue")


class ValidationOutput(BaseModel):
    validated: bool = Field(description="Whether findings pass validation")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall validation confidence")
    issues_found: List[ValidationIssue] = Field(
        default_factory=list, description="Issues found during validation"
    )
    next_step: Literal["report", "query", "retrieval"] = Field(
        description="Next workflow step"
    )
    validation_notes: str = Field(description="Detailed validation assessment")
