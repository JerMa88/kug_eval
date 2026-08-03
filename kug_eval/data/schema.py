from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class DataContractError(ValueError):
    """Raised when an input data sample violates the required schema contract."""
    pass


class GeneralizationTaskItem(BaseModel):
    """
    Standardized schema contract for generalization evaluation items.
    
    Fields:
      id: Unique identifier for the sample.
      document: Factual context / memorization prompt context.
      query: Downstream, applied, or multi-hop generalization query.
      target_entity: Exact ground-truth string answer.
      category: Optional category classification (e.g. 'car_wash', 'reversal', 'multi_hop').
      metadata: Optional dictionary of additional properties.
    """
    id: str = Field(..., description="Unique sample identifier")
    document: str = Field(..., description="Context snippet containing factual update")
    query: str = Field(..., description="Generalization query")
    target_entity: str = Field(..., description="Target answer string")
    category: Optional[str] = Field("general", description="Task category")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary metadata")

    @field_validator("id", "document", "query", "target_entity")
    @classmethod
    def validate_non_empty(cls, value: str, info) -> str:
        if not isinstance(value, str) or not value.strip():
            raise DataContractError(f"Field '{info.field_name}' must be a non-empty string.")
        return value.strip()

    def get_memorization_prompt(self) -> str:
        """Returns standard memorization prompt P_mem."""
        return f"Context: {self.document}\nQuery: What entity is this about?\nAnswer: {self.target_entity}"

    def get_generalization_prompt(self) -> str:
        """Returns standard generalization prompt P_gen."""
        return f"Query: {self.query}\nAnswer: {self.target_entity}"
