from models.schema import ZeugnisInfo

def to_freitext(info: ZeugnisInfo) -> str:
    """Unstrukturierter Freitext aus HR-Profil"""
    return f"{info.position} von {info.zeitraum}. Tätigkeiten: {', '.join(info.taetigkeiten)}. Leistung: {info.leistungsnote}. Verhalten: {info.verhaltensnote}."

def to_bulletpoints(info: ZeugnisInfo) -> str:
    """Stichpunktartige Auflistung"""
    return "\n".join([
        f"- Position: {info.position}",
        f"- Zeitraum: {info.zeitraum}",
        f"- Tätigkeiten: {', '.join(info.taetigkeiten)}",
        f"- Leistungsbeurteilung: {info.leistungsnote}",
        f"- Verhaltensbeurteilung: {info.verhaltensnote}"
    ])

def to_json_format(info: ZeugnisInfo) -> dict:
    """JSON-strukturierte Eingabe"""
    return info.dict()
