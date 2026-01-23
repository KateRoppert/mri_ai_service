/**
 * Главный компонент приложения
 */
import { useState } from 'react';
import { Layout, Typography, Space, Divider } from 'antd';
import { RocketOutlined } from '@ant-design/icons';
import PipelineForm from './components/PipelineForm';
import ProgressMonitor from './components/ProgressMonitor';
import PipelineHistory from './components/PipelineHistory';
import QualityReport from './components/QualityReport';
import NIfTIViewer from './components/NIfTIViewer';
import './App.css';

const { Header, Content } = Layout;
const { Title, Text } = Typography;

function App() {
  const [activeRun, setActiveRun] = useState(null);
  const [completedRuns, setCompletedRuns] = useState([]);

  const [historyQualityReportRunId, setHistoryQualityReportRunId] = useState(null);
  const [historyVisualizationRunId, setHistoryVisualizationRunId] = useState(null);
  const [showHistoryQualityReport, setShowHistoryQualityReport] = useState(false);
  const [showHistoryVisualization, setShowHistoryVisualization] = useState(false);


  /**
   * Показать отчёт из истории
   */
  const handleShowHistoryQualityReport = (runId) => {
    setHistoryQualityReportRunId(runId);
    setShowHistoryQualityReport(true);
  };

  /**
   * Показать визуализацию из истории
   */
  const handleShowHistoryVisualization = (runId) => {
    setHistoryVisualizationRunId(runId);
    setShowHistoryVisualization(true);
  };

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
    <Layout.Header style={{ background: '#1890ff', padding: '0 24px' }}>
      <Typography.Title level={3} style={{ color: 'white', margin: '16px 0' }}>
        🧠 Система распознавания поражений головного мозга
      </Typography.Title>
    </Layout.Header>
    
    <Layout.Content style={{ padding: '24px', maxWidth: 1400, margin: '0 auto', width: '100%' }}>
      <Tabs
        defaultActiveKey="pipeline"
        size="large"
        items={[
          {
            key: 'pipeline',
            label: (
              <span>
                <RocketOutlined />
                Запуск обработки
              </span>
            ),
            children: (
              <>
                {!activeRun ? (
                  <Card>
                    <PipelineForm onPipelineStarted={handlePipelineStarted} />
                  </Card>
                ) : (
                  <ProgressMonitor
                    runId={activeRun}
                    onComplete={handlePipelineComplete}
                  />
                )}

                {completedRuns.length > 0 && (
                  <>
                    <Divider>Новая обработка</Divider>
                    <Card>
                      <PipelineForm onPipelineStarted={handlePipelineStarted} />
                    </Card>
                  </>
                )}
              </>
            ),
          },
          {
            key: 'history',
            label: (
              <span>
                <HistoryOutlined />
                История запусков
              </span>
            ),
            children: (
              <PipelineHistory
                onShowQualityReport={handleShowHistoryQualityReport}
                onShowVisualization={handleShowHistoryVisualization}
              />
            ),
          },
        ]}
      />

      {/* Модальные окна для истории */}
      {showHistoryQualityReport && (
        <QualityReport
          runId={historyQualityReportRunId}
          visible={showHistoryQualityReport}
          onClose={() => setShowHistoryQualityReport(false)}
        />
      )}

      {showHistoryVisualization && (
        <NIfTIViewer
          runId={historyVisualizationRunId}
          visible={showHistoryVisualization}
          onClose={() => setShowHistoryVisualization(false)}
        />
      )}
    </Layout.Content>
  </Layout>
);
}

export default App;