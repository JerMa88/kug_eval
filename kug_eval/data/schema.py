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
        """
        Factual Retrieval Prompt (P_mem): Tests whether the model can extract
        the target entity from the given context document WITHOUT the answer
        being provided in the prompt.

        The model must read the document and identify the relevant entity.
        This is a genuine context-extraction task, not answer echoing.
        """
        return f"Context: {self.document}\nBased only on the context above, what is the main entity being described? Answer with the entity name only."

    def get_generalization_prompt(self) -> str:
        """
        Applied Generalization Prompt (P_gen): Tests whether the model can
        answer the downstream query WITHOUT the context being provided.

        The model must rely on internalized knowledge or reasoning to answer.
        This is the core generalization test.
        """
        return f"Question: {self.query}\nAnswer with the entity name or value only."
