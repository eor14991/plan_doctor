"""
Infrastructure: PromptBuilder
REPLACES: TemplateParser with dynamic __import__ + string.Template (security risk).
USES: Jinja2 FileSystemLoader + ChoiceLoader for language fallback.
StrictUndefined: missing template variables raise UndefinedError immediately.
"""
from __future__ import annotations
import logging
import os
from typing import Any
import jinja2

logger = logging.getLogger(__name__)
_TEMPLATES_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


class PromptBuilder:
    def __init__(self, language: str = "en", default_language: str = "en") -> None:
        self.language = language
        self.default_language = default_language
        self._env = self._build_environment(language, default_language)

    def _build_environment(self, language: str, default_language: str) -> jinja2.Environment:
        lang_dir = os.path.join(_TEMPLATES_BASE, language)
        default_dir = os.path.join(_TEMPLATES_BASE, default_language)
        loaders: list[jinja2.BaseLoader] = []
        if os.path.isdir(lang_dir):
            loaders.append(jinja2.FileSystemLoader(lang_dir))
        if os.path.isdir(default_dir) and default_dir != lang_dir:
            loaders.append(jinja2.FileSystemLoader(default_dir))
        if not loaders:
            raise ValueError(f"No template directories found for language='{language}'")
        return jinja2.Environment(
            loader=jinja2.ChoiceLoader(loaders),
            undefined=jinja2.StrictUndefined,
            auto_reload=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, context: dict[str, Any] | None = None) -> str:
        template_file = f"{template_name}.j2"
        try:
            template = self._env.get_template(template_file)
            return template.render(context or {}).strip()
        except jinja2.TemplateNotFound:
            logger.error("Template not found", extra={"template": template_file})
            raise
        except jinja2.UndefinedError as exc:
            logger.error("Missing template variable", extra={"template": template_file, "error": str(exc)})
            raise

    def set_language(self, language: str) -> None:
        self.language = language
        self._env = self._build_environment(language, self.default_language)
