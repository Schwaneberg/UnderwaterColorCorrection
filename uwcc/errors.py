"""Errors safe to present directly to the user."""


class UWCCError(Exception):
    """An actionable application error."""


class Cancelled(UWCCError):
    """An operation was canceled, not failed."""
