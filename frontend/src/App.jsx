/**
 * Главный компонент приложения
 */
import { useState } from 'react';
import { Layout, Typography, Space, Divider, Tabs, Card, Button } from 'antd';
import { RocketOutlined, HistoryOutlined, LogoutOutlined, CheckCircleOutlined } from '@ant-design/icons';
import KappaLogin from './components/KappaLogin';
import PipelineForm from './components/PipelineForm';
import ProgressMonitor from './components/ProgressMonitor';
import PipelineHistory from './components/PipelineHistory';
import QualityReport from './components/QualityReport';
import ClinicalReport from './components/ClinicalReport';
import NIfTIViewer from './components/NIfTIViewer';
import ValidationPanel from './components/ValidationPanel';
import './App.css';
import { getEntitiesForRun } from './services/api';

const { Header, Content } = Layout;
const { Title, Text } = Typography;

function App() {
  const [activeRun, setActiveRun] = useState(null);
  const [completedRuns, setCompletedRuns] = useState([]);

  const [historyQualityReportRunId, setHistoryQualityReportRunId] = useState(null);
  const [historyVisualizationRunId, setHistoryVisualizationRunId] = useState(null);
  const [historyVisualizationLesionType, setHistoryVisualizationLesionType] = useState('glioblastoma');
  const [showHistoryQualityReport, setShowHistoryQualityReport] = useState(false);
  const [showHistoryVisualization, setShowHistoryVisualization] = useState(false);
  const [historyClinicalReportRunId, setHistoryClinicalReportRunId] = useState(null);
  const [historyClinicalReportLesionType, setHistoryClinicalReportLesionType] = useState('glioblastoma');
  const [showHistoryClinicalReport, setShowHistoryClinicalReport] = useState(false);
  const [historyValidationRef, setHistoryValidationRef] = useState(null);
  const [kappaSession, setKappaSession] = useState(null);

  const handleLoginSuccess = (data) => {
    setKappaSession(data);
    localStorage.setItem('kappa_session_id', data.session_id);
  };

  const handleLogout = async () => {
    if (kappaSession?.session_id) {
      try {
        await fetch(`/api/kappa/logout?session_id=${kappaSession.session_id}`, {
          method: 'POST',
        });
      } catch (e) {
        console.error('Logout error:', e);
      }
    }
    setKappaSession(null);
    localStorage.removeItem('kappa_session_id');
  };

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
  const handleShowHistoryVisualization = async (runId, lesionType = 'glioblastoma') => {
    setHistoryVisualizationRunId(runId);
    setHistoryVisualizationLesionType(lesionType);
    setHistoryValidationRef(null);
    setShowHistoryVisualization(true);

    // Keep all_entities so the viewer can correct the entity when the
    // user switches patients/sessions in a multi-patient run.
    try {
      const result = await getEntitiesForRun(runId);
      if (result.entities && result.entities.length > 0) {
        const e = result.entities[0];
        setHistoryValidationRef({
          entity_id: e.entity_id,
          dataset_id: e.dataset_id,
          all_entities: result.entities,
        });
      }
    } catch (err) {
      console.error('Ошибка загрузки entity для валидации:', err);
    }
  };

  const handleShowHistoryClinicalReport = (runId, lesionType = 'glioblastoma') => {
    setHistoryClinicalReportRunId(runId);
    setHistoryClinicalReportLesionType(lesionType);
    setShowHistoryClinicalReport(true);
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
      lesionType: response.lesion_type || 'glioblastoma',
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
      <Layout.Header style={{
        background: '#1890ff',
        padding: '0 24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <Typography.Title level={3} style={{ color: 'white', margin: 0 }}>
          🧠 Система распознавания поражений головного мозга
        </Typography.Title>
        {kappaSession && (
          <Space>
            <Text style={{ color: 'white' }}>
              {kappaSession.first_name} {kappaSession.last_name}
            </Text>
            <Button
              icon={<LogoutOutlined />}
              onClick={handleLogout}
              size="small"
              ghost
            >
              Выход
            </Button>
          </Space>
        )}
      </Layout.Header>

      <Layout.Content style={{ padding: '24px', maxWidth: 1400, margin: '0 auto', width: '100%' }}>
        {!kappaSession ? (
          <KappaLogin onLoginSuccess={handleLoginSuccess} />
        ) : (
          <>
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
                          runId={activeRun.runId}
                          lesionType={activeRun.lesionType}
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
                      onShowClinicalReport={handleShowHistoryClinicalReport}
                    />
                  ),
                },
                {
                  key: 'validation',
                  label: (
                    <span>
                      <CheckCircleOutlined />
                      Валидация
                    </span>
                  ),
                  children: <ValidationPanel />,
              },
              ]}
            />

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
                validationRef={historyValidationRef}
                lesionType={historyVisualizationLesionType}
                onValidationRefChange={(ref) =>
                  setHistoryValidationRef((prev) => ({ ...prev, ...ref }))
                }
              />
            )}
            {showHistoryClinicalReport && (
              <ClinicalReport
                runId={historyClinicalReportRunId}
                visible={showHistoryClinicalReport}
                onClose={() => setShowHistoryClinicalReport(false)}
                lesionType={historyClinicalReportLesionType}
              />
            )}
          </>
        )}
      </Layout.Content>
    </Layout>
  );
}

export default App;