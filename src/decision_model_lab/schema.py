from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExpectedDecision(BaseModel):
    """Ground truth kept extensible until candidate output contracts are known."""

    model_config = ConfigDict(extra="allow")

    label: str | None = None


class DecisionSpec(BaseModel):
    """Model-independent description of the decision requested by one case."""

    type: Literal["noul", "choice", "score"]
    options: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_options(self) -> "DecisionSpec":
        if len(set(self.options)) != len(self.options):
            raise ValueError("decision options must be unique")
        if any(not option for option in self.options):
            raise ValueError("decision options must not be empty")
        if self.type == "noul" and self.options:
            raise ValueError("noul decisions must not define options")
        if self.type == "choice" and len(self.options) < 2:
            raise ValueError("choice decisions require at least two options")
        if self.type == "score" and not 2 <= len(self.options) <= 10:
            raise ValueError("score decisions require between 2 and 10 ordered options")
        return self


class EvaluationCase(BaseModel):
    """Canonical input unit shared by every candidate model."""

    id: str = Field(min_length=1)
    context: str = Field(min_length=1)
    question: str = Field(min_length=1)
    definitions: dict[str, Any] = Field(default_factory=dict)
    decision: DecisionSpec
    expected: ExpectedDecision
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionResult(BaseModel):
    """Model-independent observation produced by one evaluated decision."""

    case_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_output: Any = None
    latency_ms: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_probabilities(self) -> "DecisionResult":
        for label, probability in self.probabilities.items():
            if not label:
                raise ValueError("probability labels must not be empty")
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"probability for {label!r} must be between 0 and 1")

        if self.probabilities:
            total = sum(self.probabilities.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError("probabilities must sum to 1")

        return self
