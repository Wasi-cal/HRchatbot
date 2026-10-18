"""Rule TOPICS the active grounding_core prompt must cover, matched by small keyword sets instead of exact phrases, so a rewrite or
tightening of the prompt wording does not break the suites while a dropped rule still does. A topic is present when EVERY pattern
of its list matches (case-insensitive). Behavioural checks (what the model actually answers) stay strict and are not here."""
import re

TOPICS = {
    "grounding: answer only from the excerpts": [r"\bonly\b", r"excerpts"],
    "grounding: no outside / general knowledge": [r"outside knowledge|general knowledge"],
    "grounding: do not guess": [r"never guess|do not guess|don't guess"],
    "no citations of titles, sections or pages": [r"title", r"section", r"page"],
    "never say 'excerpts' / 'context' wording": [r"never say[^\n]*excerpts", r"context"],
    "group labels ([Applies to: ...])": [r"\[Applies to:"],
    "never infer the user's group from earlier topics": [r"\binfer\b", r"group", r"earlier"],
    "group is the user's own only when stated in the first person": [r"first person"],
    "no 'not entitled' / 'gets less' claims": [r"not entitled", r"gets less"],
    "couldn't-find phrasing for a stated group": [r"couldn't find <topic> for <group>"],
    "no clarifying question about the group": [r"clarifying question"],
    "leave counting states the assumption": [r"assumption"],
    "working days only, a week is five": [r"working day", r"\bfive\b"],
    "no other teams (IT, legal, finance)": [r"\bIT\b", r"legal", r"finance"],
}


def has_topic(prompt: str, topic: str) -> bool:
    return all(re.search(p, prompt, re.IGNORECASE) for p in TOPICS[topic])


def missing_topics(prompt: str, topics) -> list[str]:
    return [t for t in topics if not has_topic(prompt, t)]
