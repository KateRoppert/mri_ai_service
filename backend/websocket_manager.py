"""
Менеджер WebSocket соединений для real-time обновлений
"""

from fastapi import WebSocket
from typing import Dict, Set
import logging
import json

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Управляет WebSocket соединениями для каждого run_id"""
    
    def __init__(self):
        # Словарь: run_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, run_id: str, websocket: WebSocket):
        """
        Подключает клиента к определённому run_id
        
        Args:
            run_id: ID запуска pipeline
            websocket: WebSocket соединение
        """
        await websocket.accept()
        
        if run_id not in self.active_connections:
            self.active_connections[run_id] = set()
        
        self.active_connections[run_id].add(websocket)
        logger.info(f"WebSocket подключён к run_id: {run_id}. Всего подключений: {len(self.active_connections[run_id])}")
    
    def disconnect(self, run_id: str, websocket: WebSocket):
        """
        Отключает клиента от run_id
        
        Args:
            run_id: ID запуска pipeline
            websocket: WebSocket соединение
        """
        if run_id in self.active_connections:
            self.active_connections[run_id].discard(websocket)
            logger.info(f"WebSocket отключён от run_id: {run_id}. Осталось подключений: {len(self.active_connections[run_id])}")
            
            # Удаляем пустые множества
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]
    
    async def broadcast(self, run_id: str, message: dict):
        """
        Отправляет сообщение всем подключённым к run_id клиентам
        
        Args:
            run_id: ID запуска pipeline
            message: Словарь с данными для отправки
        """
        if run_id not in self.active_connections:
            return
        
        # Список для удаления мёртвых соединений
        dead_connections = []
        
        for websocket in self.active_connections[run_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Ошибка отправки через WebSocket для run_id {run_id}: {e}")
                dead_connections.append(websocket)
        
        # Удаляем мёртвые соединения
        for websocket in dead_connections:
            self.disconnect(run_id, websocket)
    
    def has_active_connections(self, run_id: str) -> bool:
        """
        Проверяет, есть ли активные подключения для run_id
        
        Args:
            run_id: ID запуска pipeline
            
        Returns:
            True если есть активные подключения
        """
        return run_id in self.active_connections and len(self.active_connections[run_id]) > 0


# Создаём глобальный экземпляр
ws_manager = WebSocketManager()