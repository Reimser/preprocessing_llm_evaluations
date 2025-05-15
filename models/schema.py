from pydantic import BaseModel
from typing import List

class ZeugnisInfo(BaseModel):
    position: str
    zeitraum: str
    taetigkeiten: List[str]
    leistungsnote: str
    verhaltensnote: str
