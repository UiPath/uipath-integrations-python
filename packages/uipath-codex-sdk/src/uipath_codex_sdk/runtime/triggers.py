"""The resume trigger handler codex runs use."""

from __future__ import annotations

from typing import Any

from uipath.core.triggers import UiPathResumeTriggerType
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler

_IXP_TYPES = (
    UiPathResumeTriggerType.IXP_EXTRACTION,
    UiPathResumeTriggerType.IXP_VS_ESCALATION,
)


class CodexResumeTriggerHandler(UiPathResumeTriggerHandler):
    """``UiPathResumeTriggerHandler`` that reads codex's own triggers.

    A Batch Transform trigger of ``WaitBatchTransformFile`` is read by
    :func:`~uipath_codex_sdk.batch_transform_tools.read_batch_transform_trigger`
    (status only; the tool downloads the result); every other trigger as the
    platform reads it.
    """

    async def read_trigger(self, trigger: Any) -> Any | None:
        from ..batch_transform_tools import (
            is_codex_batch_transform_trigger,
            read_batch_transform_trigger,
        )

        if is_codex_batch_transform_trigger(trigger):
            return await read_batch_transform_trigger(trigger)
        if getattr(trigger, "trigger_type", None) in _IXP_TYPES:
            return await self._read_ixp(trigger)
        return await super().read_trigger(trigger)

    async def _read_ixp(self, trigger: Any) -> Any | None:
        """The platform's IXP reading, with its failures as faulted triggers.

        A failed extraction or validation operation, or a validation task
        with no status (deleted), fails the run with a clear message instead
        of an unexpected exception.
        """
        from uipath.core.errors import ErrorCategory, UiPathFaultedTriggerError
        from uipath.platform.errors import OperationFailedException

        kind = (
            "IXP extraction"
            if trigger.trigger_type == UiPathResumeTriggerType.IXP_EXTRACTION
            else "Document validation"
        )
        try:
            return await super().read_trigger(trigger)
        except OperationFailedException as e:
            raise UiPathFaultedTriggerError(
                ErrorCategory.SYSTEM,
                f"{kind} '{trigger.item_key}' failed: {getattr(e, 'message', e)}",
            ) from e
        except KeyError as e:
            raise UiPathFaultedTriggerError(
                ErrorCategory.SYSTEM,
                f"{kind} task '{trigger.item_key}' has no status; it may have been "
                "deleted.",
            ) from e


__all__ = ["CodexResumeTriggerHandler"]
