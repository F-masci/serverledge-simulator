"""Verifiche del calendario e dell'esecuzione a eventi discreti.

Il progressivo risolve i pareggi senza confrontare eventi. Le occorrenze
conservano il proprio stato e gli eventi usano il contesto della replica.
"""

import unittest
from dataclasses import dataclass

from simulator.core import Event, EventStatus
from simulator.core.simulation import Simulation
from simulator.io.scenario import Scenario


@dataclass(frozen=True)
class RecordEvent(Event):
    """Evento di prova: registra etichetta e orologio senza logica di piattaforma."""

    label: str
    records: list

    def execute(self):
        self.records.append((self.label, self.ctx.engine.now))


class EngineTest(unittest.TestCase):
    def setUp(self):
        scenario = Scenario([{
            'name': 'test', 'runtime': 'python314', 'memory_mb': 128, 'cpu_demand': 1,
            'arrival_rate': 1, 'service_rate': 1, 'first_arrival_time': 0,
        }])
        self.engine = Simulation(scenario)._ctx.engine
        self.records = []

    def event(self, label):
        return RecordEvent(label, self.records, ctx=self.engine.ctx)

    def drain(self):
        count = 0
        while self.engine.step():
            count += 1
        return count

    def test_chronological_order_and_fifo_at_equal_times(self):
        """L'heap ordina per tempo; eventi simultanei mantengono l'inserimento."""
        self.engine.schedule(3, self.event('last'))
        self.engine.schedule(1, self.event('first'))
        self.engine.schedule(1, self.event('second'))
        self.assertEqual(self.drain(), 3)
        self.assertEqual(self.records, [('first', 1), ('second', 1), ('last', 3)])
        self.assertEqual(self.engine.now, 3)

    def test_event_can_schedule_immediate_and_future_events(self):
        """Un evento usa il tempo corrente; i figli simultanei seguono quelli già presenti."""
        records = self.records

        class Parent(Event):
            def execute(self):
                engine = self.ctx.engine
                records.append(('parent', engine.now))
                engine.schedule_after(0, RecordEvent('child', records, ctx=self.ctx))
                engine.schedule_after(2, RecordEvent('future', records, ctx=self.ctx))

        self.engine.schedule(1, Parent(ctx=self.engine.ctx))
        self.engine.schedule(1, self.event('existing'))
        self.assertEqual(self.drain(), 4)
        self.assertEqual(self.records, [('parent', 1), ('existing', 1), ('child', 1), ('future', 3)])

    def test_empty_calendar(self):
        """Senza eventi il tempo resta fermo."""
        self.assertFalse(self.engine.step())
        self.assertEqual(self.drain(), 0)
        self.assertEqual(self.engine.now, 0)

    def test_invalid_times_and_events_are_rejected(self):
        """Nessun viaggio nel passato, tempo non finito o oggetto senza contratto Event."""
        self.engine.schedule(2, self.event('anchor'))
        self.assertTrue(self.engine.step())
        for time in (-1, 1, float('nan'), float('inf'), True):
            with self.subTest(time=time):
                with self.assertRaises(ValueError):
                    self.engine.schedule(time, self.event('invalid'))
        for delay in (-1, float('inf'), float('nan'), True):
            with self.assertRaises(ValueError):
                self.engine.schedule_after(delay, self.event('invalid'))
        with self.assertRaises(TypeError):
            self.engine.schedule(2, object())
        self.assertEqual(self.engine.now, 2)

    def test_failure_is_propagated_without_hiding_partial_execution(self):
        """Un errore marca FAILED, mantiene il tempo dell'evento e non perde i successivi."""
        class Failure(Event):
            def execute(self):
                raise ValueError('failure')

        failed = self.engine.schedule(1, Failure(ctx=self.engine.ctx))
        self.engine.schedule(3, self.event('later'))
        with self.assertRaisesRegex(ValueError, 'failure'):
            self.engine.step()
        self.assertEqual(failed.status, EventStatus.FAILED)
        self.assertEqual(self.engine.now, 1)
        self.assertEqual(self.drain(), 1)

    def test_recursive_execution_is_rejected_but_scheduling_is_allowed(self):
        """Un handler può pianificare, ma non avviare step dentro un altro handler."""
        test = self

        class Recursive(Event):
            def execute(self):
                engine = self.ctx.engine
                with test.assertRaises(RuntimeError):
                    engine.step()
                engine.schedule_after(1, test.event('after'))

        self.engine.schedule(0, Recursive(ctx=self.engine.ctx))
        self.assertEqual(self.drain(), 2)
        self.assertEqual(self.records, [('after', 1)])

    def test_event_cannot_be_scheduled_in_another_replica(self):
        """Il calendario rifiuta un evento che usa il contesto di un'altra replica."""
        other_engine = Simulation(self.engine.ctx.scenario)._ctx.engine
        with self.assertRaisesRegex(ValueError, 'stesso contesto'):
            other_engine.schedule(1, self.event('foreign'))
        self.assertFalse(other_engine.step())

    def test_event_contract_is_abstract_and_schedule_is_read_only(self):
        """Gli eventi concreti implementano execute; lo stato è in sola lettura."""
        with self.assertRaises(TypeError):
            Event(ctx=self.engine.ctx)
        handle = self.engine.schedule(1, self.event('test'))
        with self.assertRaises(AttributeError):
            handle.status = EventStatus.EXECUTED


if __name__ == '__main__':
    unittest.main()
