"""Единственное место, где создаётся HTTP-клиент для Kappa (KI-064).

Зачем модуль ради одной функции: `verify=False` стояло в 15 местах трёх
модулей, включая POST с логином и паролем, и разрослось туда с 11 простым
копированием — каждый вызов конструировал клиента сам, и новый модуль брал
образец у соседа вместе с отключённой проверкой. Пока конструирование
размазано, это повторится. Здесь оно собрано в одну точку, а тест
`test_no_kappa_module_builds_a_client_outside_the_factory` запрещает
конструктор в модулях Kappa.

Отключение было наследием самоподписанного сертификата. Замер 21.10.2026:
сертификат `kappa.nsu.ru:8061` — действующий, выдан GlobalSign для `*.nsu.ru`,
`Verify return code: 0 (ok)`, и запрос со штатной проверкой возвращает 200.
"""

import logging
from typing import Union

import httpx

from config import settings

logger = logging.getLogger(__name__)

# Предупреждать один раз на процесс, а не на каждый запрос: иначе лог утонет.
_warned_once = False


def kappa_client(timeout: Union[float, httpx.Timeout]) -> httpx.AsyncClient:
    """Клиент для запросов в Kappa. Проверку TLS включает по умолчанию.

    Args:
        timeout: секунды или `httpx.Timeout` — среди вызывающих есть и то и
            другое (выгрузка файлов живёт с read=300).

    Отключить проверку можно переменной `KAPPA_TLS_VERIFY=false` — на случай
    машины без набора корневых сертификатов, где иначе отвалится вход. Это
    явное решение, и оно попадает в лог: молча работать без проверки
    сертификата на соединении с учётными данными нельзя.
    """
    global _warned_once
    verify = settings.kappa_tls_verify

    if not verify and not _warned_once:
        _warned_once = True
        logger.warning(
            "Проверка TLS-сертификата Kappa ОТКЛЮЧЕНА (KAPPA_TLS_VERIFY=false). "
            "Соединение, несущее логин и пароль, примет любой сертификат. "
            "Так можно работать только как временным обходом — см. KI-064."
        )

    return httpx.AsyncClient(timeout=timeout, verify=verify)
