import { useState } from 'react';
import { Card, Form, Input, Button, Alert, Typography } from 'antd';
import { UserOutlined, LockOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

function KappaLogin({ onLoginSuccess }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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
        const data = await response.json();
        throw new Error(data.detail || 'Ошибка авторизации');
      }

      const data = await response.json();
      onLoginSuccess(data);
    } catch (err) {
      setError(err.message);
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

          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading} block>
              Войти
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
}

export default KappaLogin;