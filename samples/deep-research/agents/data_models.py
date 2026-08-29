"""Data models for the Deep Research Workflow."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


@dataclass
class ResearchPlan:
    """Structured research plan with questions and methodology."""

    topic: str
    research_questions: List[str]
    methodology: str
    expected_sections: List[str]
    priority_order: List[int]


@dataclass
class ResearchResult:
    """Results from executing a research question."""

    question: str
    findings: str
    sources: List[str]
    confidence_score: float


@dataclass
class FinalReport:
    """Final synthesized research report."""

    topic: str
    executive_summary: str
    sections: Dict[str, str]
    sources: List[str]
    generated_at: datetime
