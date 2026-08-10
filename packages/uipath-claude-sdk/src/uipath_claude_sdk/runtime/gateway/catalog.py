"""Model id resolution against the tenant's LLM discovery catalog.

Discovery reports the ids a tenant can actually reach, which are neither
stable across tenants nor the names a developer would type. The catalog maps
friendly names such as ``claude-sonnet-4-5`` onto the tenant's real id and
carries the vendor and api flavor discovery reported for it, so the routing
strategy is always chosen from discovery rather than hardcoded.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Any, Protocol

from uipath.llm_client.settings import UiPathBaseSettings
from uipath.llm_client.settings.constants import (
    API_FLAVOR_TO_VENDOR_TYPE,
    BYOM_TO_ROUTING_FLAVOR,
    ApiFlavor,
    VendorType,
)

from .errors import GatewayShimError, ModelNotInCatalogError

logger = logging.getLogger(__name__)

FAMILIES: tuple[str, ...] = ("opus", "sonnet", "haiku")

_VERSION_SUFFIX = re.compile(r"-\d{8}-v\d+:\d+$")
_DATE_IN_NAME = re.compile(r"[-@](\d{8})")
_VERTEX_VERSION = "v1beta1"


class UiPathModelSpec(Protocol):
    """Structural contract the gateway needs from ``UiPathModel``.

    Only the requested model name is required. A concrete ``UiPathModel`` may
    carry more, and the gateway ignores anything it does not need.
    """

    @property
    def model(self) -> str: ...


@dataclass(frozen=True)
class ResolvedModel:
    """A tenant model id plus the routing facts discovery reported for it.

    Attributes:
        requested: The name the developer wrote, or the family word when the
            model was found by family rather than by name.
        model_id: The ``modelName`` discovery reported, e.g.
            ``anthropic.claude-sonnet-4-5-20250929-v1:0``. Used for logging and
            call records, never sent upstream.
        wire_name: The name the request is made with. Discovery decides the
            route, not the name: the other UiPath integrations look a model up
            to learn its vendor and api flavor and then send the name the
            developer wrote, so a name that works in ``uipath-langchain`` works
            here. Family models have no developer-supplied name, so they fall
            back to the catalogue id.
    """

    requested: str
    model_id: str
    vendor_type: str
    api_flavor: str | None
    api_version: str | None = None
    wire_name: str = ""

    def __post_init__(self) -> None:
        if not self.wire_name:
            object.__setattr__(self, "wire_name", self.requested)


@dataclass(frozen=True)
class ResolvedModelSet:
    """The primary model plus the family models used for auxiliary CLI traffic."""

    primary: ResolvedModel
    haiku: ResolvedModel
    sonnet: ResolvedModel
    opus: ResolvedModel

    def for_family(self, family: str) -> ResolvedModel:
        """Return the resolved model for a family word, or the primary model."""
        match family:
            case "haiku":
                return self.haiku
            case "sonnet":
                return self.sonnet
            case "opus":
                return self.opus
            case _:
                return self.primary


class ModelCatalog:
    """Alias table over the models the tenant's discovery endpoint reports."""

    def __init__(self, models: list[dict[str, Any]]) -> None:
        self._models = models
        self._aliases = _build_alias_index(models)

    @classmethod
    def from_settings(cls, settings: UiPathBaseSettings) -> ModelCatalog:
        """Build a catalog from a discovery call. Blocking, run it off the loop."""
        return cls(settings.get_available_models())

    @property
    def available(self) -> list[str]:
        """Every model name discovery reported, in the order it reported them."""
        return [str(m.get("modelName", "")) for m in self._models if m.get("modelName")]

    @property
    def models(self) -> list[dict[str, Any]]:
        """The raw discovery records."""
        return list(self._models)

    def resolve(self, requested: str) -> ResolvedModel:
        """Resolve a developer-supplied name to a tenant model id.

        Raises:
            ModelNotInCatalogError: when no alias matches.
        """
        info = self._aliases.get(requested) or self._aliases.get(requested.lower())
        if info is None:
            raise ModelNotInCatalogError(requested, self.available)
        return describe(requested, info)

    def resolve_family(self, family: str, *, fallback: ResolvedModel) -> ResolvedModel:
        """Resolve a family word to the newest matching model, or the fallback.

        A family word is ours, not the developer's, and no gateway would accept
        ``haiku`` as a model, so these are the one case that goes upstream by
        catalogue id.
        """
        info = self._aliases.get(family)
        if info is None:
            return fallback
        try:
            described = describe(family, info)
            return replace(described, wire_name=described.model_id)
        except GatewayShimError:
            logger.debug(
                "family %s resolved to an unroutable model, using fallback", family
            )
            return fallback

    def resolve_set(self, requested: str) -> ResolvedModelSet:
        """Resolve the primary model and the three Claude family models."""
        primary = self.resolve(requested)
        return ResolvedModelSet(
            primary=primary,
            haiku=self.resolve_family("haiku", fallback=primary),
            sonnet=self.resolve_family("sonnet", fallback=primary),
            opus=self.resolve_family("opus", fallback=primary),
        )


def describe(requested: str, info: dict[str, Any]) -> ResolvedModel:
    """Derive vendor and api flavor for a discovery record.

    Mirrors the resolution the LLM client's own factory performs: an explicit
    ``apiFlavor`` from discovery wins, the vendor fills in the flavor when
    discovery leaves it unset, and the flavor implies the vendor when only the
    flavor is reported.
    """
    model_id = str(info.get("modelName", requested))
    discovered_flavor = info.get("apiFlavor")

    vendor = info.get("vendor")
    if not vendor and discovered_flavor:
        vendor = API_FLAVOR_TO_VENDOR_TYPE.get(discovered_flavor)
    if not vendor:
        raise ModelNotInCatalogError(requested, [model_id])
    vendor_type = str(vendor).lower()

    api_flavor: str | None = None
    if discovered_flavor:
        api_flavor = str(
            BYOM_TO_ROUTING_FLAVOR.get(discovered_flavor, discovered_flavor)
        )

    api_version: str | None = None
    if api_flavor is None:
        match vendor_type:
            case VendorType.AWSBEDROCK:
                api_flavor = str(ApiFlavor.INVOKE)
            case VendorType.VERTEXAI:
                api_flavor = str(ApiFlavor.ANTHROPIC_CLAUDE)
                api_version = _VERTEX_VERSION

    return ResolvedModel(
        requested=requested,
        model_id=model_id,
        vendor_type=vendor_type,
        api_flavor=api_flavor,
        api_version=api_version,
    )


def alias_candidates(model_name: str) -> list[str]:
    """Derived names a developer might reasonably type for a tenant model id."""
    lowered = model_name.lower()
    stripped = lowered.removeprefix("anthropic.")
    candidates = [
        model_name,
        lowered,
        stripped,
        _VERSION_SUFFIX.sub("", lowered),
        _VERSION_SUFFIX.sub("", stripped),
        lowered.split("@", 1)[0],
        stripped.split("@", 1)[0],
    ]
    seen: dict[str, None] = {}
    for candidate in candidates:
        if candidate:
            seen.setdefault(candidate, None)
    return list(seen)


def _build_alias_index(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}

    for info in models:
        name = info.get("modelName")
        if not name:
            continue
        index[str(name)] = info
        index.setdefault(str(name).lower(), info)

    for info in models:
        name = info.get("modelName")
        if not name:
            continue
        for alias in alias_candidates(str(name)):
            index.setdefault(alias, info)

    for family in FAMILIES:
        newest = _newest_in_family(models, family)
        if newest is not None:
            index.setdefault(family, newest)

    return index


def _newest_in_family(
    models: list[dict[str, Any]], family: str
) -> dict[str, Any] | None:
    matches = [
        info for info in models if family in str(info.get("modelName", "")).lower()
    ]
    if not matches:
        return None
    return max(matches, key=_recency_key)


def _recency_key(info: dict[str, Any]) -> tuple[str, str]:
    name = str(info.get("modelName", ""))
    dates = _DATE_IN_NAME.findall(name.lower())
    return (dates[-1] if dates else "", name)


__all__ = [
    "FAMILIES",
    "ModelCatalog",
    "ResolvedModel",
    "ResolvedModelSet",
    "UiPathModelSpec",
    "alias_candidates",
    "describe",
]
