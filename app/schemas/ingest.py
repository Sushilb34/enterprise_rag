from pydantic import BaseModel
from typing import Optional


class IngestRequest(BaseModel):
    """
    Request schema for ingestion.
    """

    data_path: Optional[str] = "data/raw"


class IngestResponse(BaseModel):
    """
    Response schema after ingestion.
    """

    message: str
    documents_processed: int