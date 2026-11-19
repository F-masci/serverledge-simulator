"""Verifiche del calendario e dell'esecuzione a eventi discreti.

Simulation.schedule, Simulation.run, Simulation.handle ed Event.
Il progressivo risolve i pareggi senza confrontare eventi; il dispatch è
polimorfico. Orizzonte inclusivo, handle per occorrenza e stati di fallimento
sono contratti espliciti del nuovo motore, non copie dell'interfaccia originale.
"""

from dataclasses import dataclass
import unittest

from sim.core import Event, EventStatus, SimulationEngine


@dataclass
class RecordEvent(Event):
    """Evento di prova: registra etichetta e orologio senza logica di piattaforma."""

    label: str
    records: list

    def execute(self, simulation, engine, infrastructure):
        self.records.append((self.label, engine.now))


class EngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine()
        self.records = []

    def event(self, label):
        return RecordEvent(label, self.records)

    def test_chronological_order_and_fifo_at_equal_times(self):
        """L'heap ordina per tempo; eventi simultanei mantengono l'inserimento."""
        self.engine.schedule(3, self.event('last'))
        self.engine.schedule(1, self.event('first'))
        self.engine.schedule(1, self.event('second'))
        self.assertEqual(self.engine.run(), 3)
        self.assertEqual(self.records, [('first', 1), ('second', 1), ('last', 3)])
        self.assertEqual(self.engine.now, 3)

    def test_event_can_schedule_immediate_and_future_events(self):
        """Un evento usa il tempo corrente; i figli simultanei seguono quelli già presenti."""
        records = self.records

        class Parent(Event):
            def execute(self, simulation, engine, infrastructure):
                records.append(('parent', engine.now))
                engine.schedule_after(0, RecordEvent('child', records))
                engine.schedule_after(2, RecordEvent('future', records))

        self.engine.schedule(1, Parent())
        self.engine.schedule(1, self.event('existing'))
        self.assertEqual(self.engine.run(), 4)
        self.assertEqual(self.records, [('parent', 1), ('existing', 1), ('child', 1), ('future', 3)])

    def test_horizon_includes_boundary_and_allows_resumption(self):
        """Run include gli eventi sul confine e conserva quelli successivi."""
        self.engine.schedule(2, self.event('boundary'))
        self.engine.schedule(5, self.event('later'))
        self.assertEqual(self.engine.run(until=2), 1)
        self.assertEqual(self.engine.now, 2)
        self.assertEqual(self.engine.next_event_time, 5)
        self.assertEqual(self.engine.run(until=4), 0)
        self.assertEqual(self.engine.now, 4)
        self.assertTrue(self.engine.step())
        self.assertEqual(self.engine.now, 5)
        self.assertFalse(self.engine.step())

    def test_cancelled_occurrence_does_not_execute_or_advance_clock(self):
        """La cancellazione riguarda una sola occorrenza ed è idempotente."""
        event = self.event('shared')
        cancelled = self.engine.schedule(10, event)
        active = self.engine.schedule(2, event)
        self.assertTrue(cancelled.cancel())
        self.assertFalse(cancelled.cancel())
        self.assertEqual(self.engine.pending_count, 1)
        self.assertEqual(self.engine.run(), 1)
        self.assertEqual(self.records, [('shared', 2)])
        self.assertEqual(self.engine.now, 2)
        self.assertEqual(active.status, EventStatus.EXECUTED)
        self.assertFalse(active.cancel())
        self.assertIsNone(self.engine.next_event_time)

    def test_empty_calendar_and_explicit_horizon(self):
        """Senza eventi il tempo resta fermo, salvo un orizzonte esplicito."""
        self.assertFalse(self.engine.step())
        self.assertEqual(self.engine.run(), 0)
        self.assertEqual(self.engine.now, 0)
        self.assertEqual(self.engine.run(until=7), 0)
        self.assertEqual(self.engine.now, 7)

    def test_invalid_times_and_events_are_rejected(self):
        """Nessun viaggio nel passato, tempo non finito o oggetto senza contratto Event."""
        self.engine.run(until=2)
        for time in (-1, 1, float('nan'), float('inf'), True):
            with self.subTest(time=time):
                with self.assertRaises(ValueError):
                    self.engine.schedule(time, self.event('invalid'))
                with self.assertRaises(ValueError):
                    self.engine.run(until=time)
        for delay in (-1, float('inf'), float('nan'), True):
            with self.assertRaises(ValueError):
                self.engine.schedule_after(delay, self.event('invalid'))
        with self.assertRaises(TypeError):
            self.engine.schedule(2, object())
        self.assertEqual(self.engine.pending_count, 0)
        self.assertEqual(self.engine.now, 2)

    def test_failure_is_propagated_without_hiding_partial_execution(self):
        """Un errore marca FAILED, mantiene il tempo dell'evento e non perde i successivi."""
        class Failure(Event):
            def execute(self, simulation, engine, infrastructure):
                raise ValueError('failure')

        failed = self.engine.schedule(1, Failure())
        self.engine.schedule(3, self.event('later'))
        with self.assertRaisesRegex(ValueError, 'failure'):
            self.engine.run(until=5)
        self.assertEqual(failed.status, EventStatus.FAILED)
        self.assertEqual(self.engine.now, 1)
        self.assertEqual(self.engine.pending_count, 1)
        self.assertEqual(self.engine.run(), 1)

    def test_recursive_execution_is_rejected_but_scheduling_is_allowed(self):
        """Un handler può pianificare, ma non avviare step/run dentro un altro handler."""
        test = self

        class Recursive(Event):
            def execute(self, simulation, engine, infrastructure):
                with test.assertRaises(RuntimeError):
                    engine.step()
                with test.assertRaises(RuntimeError):
                    engine.run()
                engine.schedule_after(1, test.event('after'))

        self.engine.schedule(0, Recursive())
        self.assertEqual(self.engine.run(), 2)
        self.assertEqual(self.records, [('after', 1)])

    def test_event_contract_is_abstract_and_schedule_is_read_only(self):
        """Gli eventi concreti implementano execute; tempo e stato non hanno setter."""
        with self.assertRaises(TypeError):
            Event()
        handle = self.engine.schedule(1, self.event('test'))
        with self.assertRaises(AttributeError):
            handle.time = 3
        with self.assertRaises(AttributeError):
            handle.status = EventStatus.EXECUTED


if __name__ == '__main__':
    unittest.main()
