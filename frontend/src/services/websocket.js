/**
 * WebSocket клиент для real-time обновлений прогресса
 */

class WebSocketService {
  constructor() {
    this.ws = null;
    this.runId = null;
    this.listeners = [];
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  /**
   * Подключиться к WebSocket для конкретного run_id
   */
  connect(runId, onMessage, onError) {
    this.runId = runId;
    
    // WebSocket URL (прокси настроен в vite.config.js)
    const wsUrl = `ws://${window.location.host}/ws/pipeline/${runId}`;
    
    console.log('Подключение к WebSocket:', wsUrl);
    
    this.ws = new WebSocket(wsUrl);
    
    this.ws.onopen = () => {
      console.log('WebSocket подключён к run_id:', runId);
      this.reconnectAttempts = 0;
      
      // Запускаем ping каждые 30 секунд для keep-alive
      this.startPing();
    };
    
    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WebSocket сообщение:', data);
        
        // Вызываем callback
        if (onMessage) {
          onMessage(data);
        }
        
        // Уведомляем всех подписчиков
        this.listeners.forEach(listener => listener(data));
      } catch (error) {
        console.error('Ошибка парсинга WebSocket сообщения:', error);
      }
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket ошибка:', error);
      if (onError) {
        onError(error);
      }
    };
    
    this.ws.onclose = () => {
      console.log('WebSocket отключён');
      this.stopPing();
      
      // Попытка переподключения
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        console.log(`Попытка переподключения ${this.reconnectAttempts}/${this.maxReconnectAttempts}...`);
        setTimeout(() => {
          this.connect(runId, onMessage, onError);
        }, 2000);
      }
    };
  }

  /**
   * Отправить ping для keep-alive
   */
  startPing() {
    this.pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send('ping');
      }
    }, 30000); // каждые 30 секунд
  }

  /**
   * Остановить ping
   */
  stopPing() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * Добавить подписчика на сообщения
   */
  subscribe(callback) {
    this.listeners.push(callback);
    
    // Возвращаем функцию отписки
    return () => {
      this.listeners = this.listeners.filter(listener => listener !== callback);
    };
  }

  /**
   * Отключиться от WebSocket
   */
  disconnect() {
    this.stopPing();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    this.listeners = [];
    this.runId = null;
  }

  /**
   * Проверить, подключён ли WebSocket
   */
  isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }
}

// Создаём singleton экземпляр
const wsService = new WebSocketService();

export default wsService;