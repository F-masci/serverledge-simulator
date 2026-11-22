"""Subscriber che inoltra le notifiche della simulazione al logger standard."""

import logging

from ..observer.notifications import Notification

# Logger dedicato al flusso delle notifiche di simulazione.
_logger = logging.getLogger('sim.notifications')


def log_notification(notification: Notification) -> None:
    """Registra la notifica usando il livello e la rappresentazione testuale dichiarati."""
    _logger.log(notification.level, '%s', notification)
