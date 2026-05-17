from pydantic import BaseModel, Field


class DeleteRequest(BaseModel):
    doc_ids: list[str]
