"""
_simpy_fallback.py — минимальная встроенная замена SimPy на случай,
если основная библиотека не установлена (например, в закрытом
корпоративном контуре без интернета).

Реализует ровно то подмножество API SimPy, которое использует
наш simulation.py:
  - Environment()
  - env.timeout(delay) -> событие
  - env.process(generator) -> процесс
  - env.run(until=N)
  - env.now -> текущее время

Для полноценного использования рекомендуется установить настоящий SimPy:
    pip install simpy>=4.0
"""
import heapq
import itertools


class _Timeout:
    __slots__ = ('env', 'delay')

    def __init__(self, env, delay):
        self.env = env
        self.delay = delay


class _Process:
    """Обёртка над генератором, реализующая SimPy-подобный процесс."""
    __slots__ = ('env', 'generator', '_alive')

    def __init__(self, env, generator):
        self.env = env
        self.generator = generator
        self._alive = True
        # Стартуем сразу: получаем первый yield и планируем
        self._step()

    def _step(self):
        if not self._alive:
            return
        try:
            event = next(self.generator)
        except StopIteration:
            self._alive = False
            return

        if isinstance(event, _Timeout):
            t_fire = self.env.now + event.delay
            heapq.heappush(self.env._queue,
                            (t_fire, next(self.env._counter), self))
        else:
            # Не поддерживаемое событие — пытаемся продолжить
            self._step()


class Environment:
    """Минимальное окружение дискретно-событийной симуляции."""

    def __init__(self):
        self.now = 0
        self._queue = []
        self._counter = itertools.count()

    def timeout(self, delay):
        """Создать событие задержки."""
        return _Timeout(self, delay)

    def process(self, generator):
        """Запустить процесс из генератора."""
        return _Process(self, generator)

    def run(self, until):
        """Прогнать симуляцию до момента until."""
        while self._queue:
            t, _, proc = self._queue[0]
            if t > until:
                self.now = until
                break
            heapq.heappop(self._queue)
            self.now = t
            proc._step()
        else:
            # Очередь опустела — досрочный выход
            self.now = max(self.now, until)
