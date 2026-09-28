import { useState } from 'react';
import { Card, Form, Input, Button, Alert, Typography } from 'antd';
import { UserOutlined, LockOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

function KappaLogin({ onLoginSuccess, kappaReachable = true, onSkip }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Достать человеческое сообщение из неудачного ответа.
   *
   * Тело ошибки — не всегда JSON: необработанное исключение отдаётся
   * Uvicorn'ом простой строкой "Internal Server Error", и попытка её
   * распарсить показывала оператору «JSON.parse: unexpected character»
   * вместо объяснения, что Kappa недоступна.
   */
  const readError = async (response) => {
    try {
      const data = await response.json();
      if (data?.detail) return data.detail;
    } catch {
      // тело не JSON — сообщение соберём по коду ответа ниже
    }
    if (response.status === 503) {
      return 'Kappa сейчас недоступна. Попробуйте позже.';
    }
    if (response.status === 401) {
      return 'Неверный логин или пароль Kappa.';
    }
    return `Не удалось войти (код ${response.status}).`;
  };

  const handleSubmit = async (values) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/kappa/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          login_id: values.login_id,
          passwd: values.passwd,
        }),
      });

      if (!response.ok) {
        throw new Error(await readError(response));
      }

      const data = await response.json();
      onLoginSuccess(data);
    } catch (err) {
      // TypeError от fetch означает, что до бэкенда не достучались вовсе
      // (сеть, контейнер лежит) — это не «ошибка авторизации».
      setError(
        err instanceof TypeError
          ? 'Нет связи с сервисом. Проверьте, запущен ли он.'
          : err.message,
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: 'calc(100vh - 64px)',
      padding: '24px',
    }}>
      <Card style={{ width: 420, textAlign: 'center' }}>
        <Title level={4} style={{ marginBottom: 4 }}>
          Авторизация
        </Title>
        <Text type="secondary" style={{ display: 'block', marginBottom: 24 }}>
          Войдите через учётную запись Kappa
        </Text>

        {/* Форма остаётся точкой входа даже когда Kappa лежит: именно ввод
            логина ставит вход в очередь, и прятать её — значит лишать
            оператора этой возможности. */}
        {kappaReachable === false && (
          <Alert
            type="warning"
            showIcon
            style={{ marginBottom: 16, textAlign: 'left' }}
            message="Kappa сейчас недоступна"
            description={
              'Введите данные — вход выполнится автоматически, как только '
              + 'связь восстановится, и результаты уйдут в Kappa сами. '
              + 'Либо продолжайте без входа: обработка работает в любом случае.'
            }
          />
        )}

        {error && (
          <Alert
            message={error}
            type="error"
            showIcon
            closable
            onClose={() => setError(null)}
            style={{ marginBottom: 16 }}
          />
        )}

        <Form onFinish={handleSubmit} layout="vertical" size="large">
          <Form.Item
            name="login_id"
            rules={[{ required: true, message: 'Введите логин' }]}
          >
            <Input prefix={<UserOutlined />} placeholder="Логин" />
          </Form.Item>

          <Form.Item
            name="passwd"
            rules={[{ required: true, message: 'Введите пароль' }]}
          >
            <Input.Password prefix={<LockOutlined />} placeholder="Пароль" />
          </Form.Item>

          <Form.Item style={{ marginBottom: onSkip ? 8 : 0 }}>
            <Button type="primary" htmlType="submit" loading={loading} block>
              Войти
            </Button>
          </Form.Item>
          {onSkip && (
            <Button type="link" block onClick={onSkip}>
              Продолжить без входа
            </Button>
          )}
        </Form>
      </Card>
    </div>
  );
}

export default KappaLogin;