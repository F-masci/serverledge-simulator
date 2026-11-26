"""Test degli invarianti del motore a eventi discreti (`SimulationEngine`).

Verifica ordinamento cronologico e FIFO a parità di timestamp, pianificazione nidificata
di eventi figli, rifiuto di tempi nel passato/non finiti, isolamento dei fallimenti
e divieto di rientranza di `step()`.
"""

import unittest
from dataclasses import dataclass

from simulator.core import Event, EventStatus
from simulator.core.simulation import Simulation
from simulator.io.scenario import Scenario
from simulator.model.execution.function import Function, FunctionProfile
from simulator.model.execution.runtime import BuiltinRuntime


@dataclass(frozen=True)
class RecordEvent(Event):
    name: str
    records: list
    action: object = None

    def execute(self) -> None:
        self.records.append((self.name, self.ctx.engine.now))
        if self.action is not None:
            self.action(self.ctx.engine)


class EngineTest(unittest.TestCase):
    def setUp(self):
        function = Function('f', 128, 1, BuiltinRuntime('python314'), profile=FunctionProfile(1, 1, 0))
        self.engine = Simulation(Scenario((function,)))._ctx.engine
        self.records = []

    def event(self, name, action=None):
        return RecordEvent(ctx=self.engine.ctx, name=name, records=self.records, action=action)

    def drain(self):
        executed = 0
        while self.engine.step():
            executed += 1
        return executed

    def test_chronological_order_and_fifo_at_equal_times(self):
        """Verifica che il calendario esegua gli eventi in ordine cronologico e FIFO a parità di istante.

        - Pianifica eventi ai tempi `5`, `2` (`first`), `2` (`second`), `0`.
        - Verifica l'ordine esatto di esecuzione `(start@0, first@2, second@2, late@5)` e il
          tempo finale `now == 5`.
        """
        self.engine.schedule(5, self.event('late'))
        self.engine.schedule(2, self.event('first'))
        self.engine.schedule(2, self.event('second'))
        self.engine.schedule(0, self.event('start'))
        self.assertEqual(self.drain(), 4)
        self.assertEqual(self.records, [('start', 0), ('first', 2), ('second', 2), ('late', 5)])
        self.assertEqual(self.engine.now, 5)

    def test_event_can_schedule_immediate_and_future_events(self):
        """Verifica la pianificazione di eventi figli immediati (`delay=0`) e futuri dentro un handler.

        - L'evento `parent@1` pianifica `child-now` con `delay=0` e `child-later` con `delay=2.5`,
          mentre `sibling@1` era già presente in coda.
        - `sibling@1` precede `child-now@1` (FIFO), seguito da `child-later@3.5`, e tutti gli handle
          passano allo stato `EXECUTED`.
        """
        def spawn(engine):
            engine.schedule_after(0, self.event('child-now'))
            engine.schedule_after(2.5, self.event('child-later'))

        parent = self.engine.schedule(1, self.event('parent', spawn))
        sibling = self.engine.schedule(1, self.event('sibling'))
        self.assertEqual(self.drain(), 4)
        self.assertEqual(self.records, [
            ('parent', 1), ('sibling', 1), ('child-now', 1), ('child-later', 3.5),
        ])
        self.assertEqual((parent.status, sibling.status), (EventStatus.EXECUTED, EventStatus.EXECUTED))

    def test_invalid_times_and_events_are_rejected(self):
        """Verifica che il calendario rifiuti tempi nel passato, non finiti/booleani o oggetti non `Event`.

        - Dopo aver avanzato il clock a `t=2`, `schedule` rifiuta `-1`, `1` (passato), `NaN`, `+Inf`, `True`.
        - `schedule_after` rifiuta ritardi negativi, non finiti o booleani.
        - Passare un oggetto che non implementa `Event` solleva `TypeError` senza alterare `now`.
        """
        self.engine.schedule(2, self.event('anchor'))
        self.assertTrue(self.engine.step())
        invalid_event = self.event('invalid')
        for time in (-1, 1, float('nan'), float('inf'), True):
            with self.subTest(time=time), self.assertRaises(ValueError):
                self.engine.schedule(time, invalid_event)
        for delay in (-1, float('inf'), float('nan'), True):
            with self.subTest(delay=delay), self.assertRaises(ValueError):
                self.engine.schedule_after(delay, invalid_event)
        with self.assertRaises(TypeError):
            self.engine.schedule(2, object())
        self.assertEqual(self.engine.now, 2)

    def test_failure_is_propagated_without_hiding_partial_execution(self):
        """Verifica che un'eccezione in un evento lo marchi `FAILED` senza perdere gli eventi successivi.

        - Un evento a `t=1` solleva `ValueError`: `step()` propaga l'eccezione, imposta
          `failed.status == FAILED` e mantiene `now == 1`.
        - L'evento successivo a `t=3` rimane in calendario e viene eseguito al `drain()` successivo.
        """
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
        """Verifica che un handler non possa invocare ricorsivamente `step()`, ma possa pianificare nuovi eventi.

        - Chiamare `engine.step()` dentro `Event.execute()` solleva `RuntimeError`.
        - Subito dopo, lo stesso handler pianifica `after@1` che viene eseguito regolarmente.
        """
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


if __name__ == '__main__':
    unittest.main()
