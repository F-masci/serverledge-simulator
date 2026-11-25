import logging
import unittest

from simulator.model.execution.function import Function
from simulator.model.execution.request import Request
from simulator.model.execution.runtime import BuiltinRuntime
from simulator.observer import RequestArrived
from simulator.utils.logger import log_notification


class NotificationLoggerTest(unittest.TestCase):
    def test_logger_uses_the_level_declared_by_the_notification_class(self):
        class WarningArrival(RequestArrived):
            level = logging.WARNING

        request = Request(Function('hello', 128, 1, BuiltinRuntime('python314')))
        notification = WarningArrival('run', 0.0, request)

        with self.assertLogs('simulator.notifications', level='INFO') as logs:
            log_notification(notification)

        record, = logs.records
        self.assertEqual(record.levelno, logging.WARNING)
        self.assertEqual(record.getMessage(), str(notification))
