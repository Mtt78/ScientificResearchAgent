from typing import List


def list_of_items_to_grammatical_text(items: List[str]) -> str:
    if len(items) <= 1: return ''.join(items)
    if len(items) == 2: return ' and '.join(items)
    return '{}, and {}'.format(', '.join(items[:-1]), items[-1])


def parse_feedback_sections(text):
    """Parse required labels and an explicit 1-5 rating; never infer missing fields."""
    import re
    from models.openai import normalize_section_labels
    text = normalize_section_labels(text)
    pattern = r'^Review:[ \t]*(.*?)^Feedback:[ \t]*(.*?)^Rating(?:[ \t]*\(1-5\))?:[ \t\n]*[*_]*[ \t]*(?:Rating:[*_ \t]*)?([1-5])(?:[ \t]*/[ \t]*5)?'
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if match:
        return {'review': match.group(1).strip(), 'feedback': match.group(2).strip(), 'rating': int(match.group(3))}
    return {'review': None, 'feedback': None, 'rating': None}
