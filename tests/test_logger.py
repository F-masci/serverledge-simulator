import logging
import unittest

from sim.model.function import Function
from sim.model.request import Request
from sim.model.runtime import BuiltinRuntime
from sim.observer import RequestArrived
from sim.utils.logger import log_notification


class NotificationLoggerTest(unittest.TestCase):
    def test_logger_uses_the_level_declared_by_the_notification_class(self):
        class WarningArrival(RequestArrived):
            level = logging.WARNING

        request = Request(Function('hello', 128, 1, BuiltinRuntime('python314')))
        notification = WarningArrival('run', 0.0, request)

        with self.assertLogs('sim.notifications', level='INFO') as logs:
            log_notification(notification)

        record, = logs.records
        self.assertEqual(record.levelno, logging.WARNING)
        self.assertEqual(record.getMessage(), str(notification))
