"""Subscriber che inoltra le notifiche al logger standard."""

import logging

from .notifications import Notification, NotificationType


_logger = logging.getLogger('sim.notifications')


def log_notification(notification: Notification) -> None:
    """Delega alla notifica la sua rappresentazione testuale."""
    level = logging.ERROR if notification.type is NotificationType.SIMULATION_ERROR else logging.INFO
    _logger.log(level, '%s', notification)
