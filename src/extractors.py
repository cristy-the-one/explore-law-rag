from pydantic import BaseModel


class ExtractedRule(BaseModel):
    subject: str
    action: str
    conditions: str = ""


def create_contextual_summary(llm_ro, text: str, metadata: dict) -> str:
    prompt = (
        "Rezuma pe scurt fragmentul legislativ următor, păstrând contextul juridic.\n"
        f"Lege: {metadata.get('law_nr', 'unknown')}\n\n{text[:4000]}"
    )
    return llm_ro.complete(prompt).text.strip()


def extract_rules(llm_legal, text: str, node_id: str) -> list[ExtractedRule]:
    prompt = (
        "Extrage reguli juridice în format JSON listă cu chei: "
        "subject, action, conditions.\n"
        f"NODE: {node_id}\n\n{text[:5000]}"
    )
    response = llm_legal.complete(prompt).text.strip()
    if not response:
        return []
    return []
