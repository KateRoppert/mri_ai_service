/**
 * Главный компонент приложения
 */
import { useState } from 'react';
import { Layout, Typography, Space, Divider } from 'antd';
import { RocketOutlined } from '@ant-design/icons';
import PipelineForm from './components/PipelineForm';
import ProgressMonitor from './components/ProgressMonitor';
import './App.css';

const { Header, Content } = Layout;
const { Title, Text } = Typography;

function App() {
  const [activeRun, setActiveRun] = useState(null);
  const [completedRuns, setCompletedRuns] = useState([]);

  /**
   * Обработчик успешного запуска pipeline
   */
  const handlePipelineStarted = (response) => {
    console.log('Pipeline запущен:', response);
    setActiveRun({
      runId: response.run_id,
      status: response.status,
      createdAt: response.created_at,
    });
  };

  /**
   * Обработчик завершения pipeline
   */
  const handlePipelineComplete = (data) => {
    console.log('Pipeline завершён:', data);
    
    // Добавляем в список завершённых
    setCompletedRuns(prev => [...prev, {
      runId: data.run_id,
      status: data.status,
      completedAt: new Date().toISOString(),
    }]);
    
    // Убираем из активных
    // (оставляем на экране для просмотра результатов)
    // setActiveRun(null);
  };

  return (
    <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
      {/* Шапка */}
      <Header style={{ 
        background: '#fff', 
        padding: '0 24px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      }}>
        <Space align="center" size="middle">
          <RocketOutlined style={{ fontSize: 32, color: '#1890ff' }} />
          <div>
            <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
              AI-Сервис Распознавания Поражений Головного Мозга
            </Title>
            <Text type="secondary">Клиническое применение</Text>
          </div>
        </Space>
      </Header>

      {/* Основной контент */}
      <Content style={{ padding: '24px' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          
          {/* Форма запуска */}
          {!activeRun && (
            <PipelineForm onPipelineStarted={handlePipelineStarted} />
          )}

          {/* Мониторинг активного запуска */}
          {activeRun && (
            <>
              <ProgressMonitor
                runId={activeRun.runId}
                onComplete={handlePipelineComplete}
              />
              
              <Divider />
              
              {/* Кнопка для запуска нового pipeline */}
              <PipelineForm onPipelineStarted={handlePipelineStarted} />
            </>
          )}

          {/* История завершённых запусков (опционально) */}
          {completedRuns.length > 0 && (
            <div style={{ marginTop: 24 }}>
              <Text type="secondary">
                Завершено запусков: {completedRuns.length}
              </Text>
            </div>
          )}
        </div>
      </Content>
    </Layout>
  );
}

export default App;