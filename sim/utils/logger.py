"""Subscriber che inoltra le notifiche al logger standard."""

import logging

from ..observer.notifications import Notification


_logger = logging.getLogger('sim.notifications')


def log_notification(notification: Notification) -> None:
    """Usa il livello e la rappresentazione testuale della notifica."""
    _logger.log(notification.level, '%s', notification)
