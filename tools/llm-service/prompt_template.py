"""
Simple prompt template system.

Supports {placeholder} slots filled via render(**kwargs).
Templates can be loaded from strings or files.
"""

from pathlib import Path
from typing import Any, Dict, Optional


class PromptTemplate:
    """A prompt template with named placeholders.

    Uses Python's str.format_map for substitution.
    Placeholders are written as ``{name}`` in the template string.

    Example::

        tpl = PromptTemplate("Translate '{text}' to {language}.")
        prompt = tpl.render(text="Hello", language="French")
        # => "Translate 'Hello' to French."
    """

    def __init__(self, template: str, *, name: Optional[str] = None):
        self.template = template
        self.name = name

    def render(self, **kwargs: Any) -> str:
        """Fill placeholders and return the final prompt string."""
        return self.template.format_map(kwargs)

    @classmethod
    def from_file(cls, path: str, *, name: Optional[str] = None) -> "PromptTemplate":
        """Load a template from a text file."""
        text = Path(path).read_text(encoding="utf-8")
        return cls(text, name=name or Path(path).stem)

    def __repr__(self) -> str:
        preview = self.template[:60] + ("..." if len(self.template) > 60 else "")
        return f"PromptTemplate(name={self.name!r}, template={preview!r})"
