/**
 * Компонент для мониторинга выполнения pipeline
 */
import { useEffect, useState } from 'react';
import { Card, Progress, Space, Tag, Button, Divider, Alert } from 'antd';
import { 
  SyncOutlined, 
  CheckCircleOutlined, 
  CloseCircleOutlined,
  ReloadOutlined 
} from '@ant-design/icons';
import StageProgress from './StageProgress';
import QualityReport from './QualityReport'; 
import NIfTIViewer from './NIfTIViewer'; 
import ClinicalReport from './ClinicalReport';
import wsService from '../services/websocket';
import { getPipelineStatus } from '../services/api';

const ProgressMonitor = ({ runId, onComplete }) => {
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [stages, setStages] = useState({});
  const [overallProgress, setOverallProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState(0);
  const [status, setStatus] = useState('running');
  const [error, setError] = useState(null);
  const [showQualityReport, setShowQualityReport] = useState(false); 
  const [showVisualization, setShowVisualization] = useState(false);
  const [showClinicalReport, setShowClinicalReport] = useState(false);

  /**
   * Подключение к WebSocket при монтировании компонента
   */
  useEffect(() => {
    // Сначала получаем текущий статус через REST API
    fetchInitialStatus();

    // Ждём 500мс перед подключением к WebSocket
    // (даём backend время запустить мониторинг)
    const connectTimeout = setTimeout(() => {
        wsService.connect(
        runId,
        handleWebSocketMessage,
        (error) => {
            // Не показываем ошибку если pipeline завершён
            if (status !== 'completed' && status !== 'failed') {
            handleWebSocketError(error);
            }
        }
        );
    }, 500);

    // Отключаемся при размонтировании
    return () => {
        clearTimeout(connectTimeout);
        wsService.disconnect();
    };
    }, [runId]);

  /**
   * Получить начальный статус через REST API
   */
  const fetchInitialStatus = async () => {
    try {
      const data = await getPipelineStatus(runId);
      updateStatus(data);
    } catch (error) {
      // Не показываем ошибку — WebSocket подхватит обновления
      console.warn('Начальный запрос статуса не удался, ожидаем WebSocket:', error);
    }
  };

  /**
   * Обработчик сообщений от WebSocket
   */
  const handleWebSocketMessage = (data) => {
    if (data.type === 'progress_update') {
      updateStatus(data);
    }
  };

  /**
   * Обработчик ошибок WebSocket
   */
  const handleWebSocketError = (error) => {
    console.error('WebSocket ошибка:', error);
    setError('Потеряно соединение с сервером');
  };

  /**
   * Открыть 3D визуализацию
   */
  const handleShowVisualization = () => {
    setShowVisualization(true);
  };

  /**
   * Закрыть 3D визуализацию
   */
  const handleCloseVisualization = () => {
    setShowVisualization(false);
  };

  const handleShowClinicalReport = () => setShowClinicalReport(true);
  const handleCloseClinicalReport = () => setShowClinicalReport(false);

  /**
   * Обновить состояние на основе данных от backend
   */
  const updateStatus = (data) => {
    setStatus(data.status);
    setOverallProgress(data.overall_progress || 0);
    setCurrentStage(data.current_stage || 0);
    
    if (data.stages) {
      setStages(data.stages);
    }

    // Если pipeline завершён - уведомляем родительский компонент
    if (data.status === 'completed' || data.status === 'failed') {
      if (onComplete) {
        onComplete(data);
      }
    }
  };

  /**
   * Открыть модальное окно с отчётом о качестве
   */
  const handleShowQualityReport = () => {
    setShowQualityReport(true);
  };

  /**
   * Закрыть модальное окно с отчётом
   */
  const handleCloseQualityReport = () => {
    setShowQualityReport(false);
  };

  /**
   * Определяем общий статус
   */
  const getOverallStatus = () => {
    switch (status) {
      case 'completed':
        return {
          color: 'success',
          icon: <CheckCircleOutlined />,
          text: 'Обработка завершена',
        };
      case 'failed':
        return {
          color: 'error',
          icon: <CloseCircleOutlined />,
          text: 'Ошибка выполнения',
        };
      case 'running':
      default:
        return {
          color: 'processing',
          icon: <SyncOutlined spin />,
          text: 'Выполняется обработка',
        };
    }
  };

  const overallStatus = getOverallStatus();

  return (
    <Card
      title={
        <Space>
          {overallStatus.icon}
          <span>{overallStatus.text}</span>
        </Space>
      }
      extra={
        <Tag color={overallStatus.color}>
          ID: {runId.substring(0, 8)}...
        </Tag>
      }
    >
      {/* Общий прогресс */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ marginBottom: 8 }}>
          <strong>Общий прогресс:</strong>
        </div>
        <Progress
          percent={Math.round(overallProgress)}
          status={
            status === 'completed' ? 'success' :
            status === 'failed' ? 'exception' : 'active'
          }
          size={['default', 12]}
        />
      </div>

      {/* Ошибка (если есть) */}
      {error && (
        <Alert
          title="Ошибка"
          description={error}
          type="error"
          showIcon
          closable
          style={{ marginBottom: 16 }}
        />
      )}

      <Divider>Этапы обработки</Divider>

      {/* Список этапов */}
      {Object.keys(stages).length > 0 ? (
        Object.entries(stages).map(([stageNum, stageData]) => (
            <StageProgress
            key={stageNum}
            stageNumber={stageData.stage_number}
            stageName={stageData.stage_name}
            status={stageData.status}
            progress={Math.round(stageData.progress)}
            onShowQualityReport={stageData.stage_number === 3 ? handleShowQualityReport : null}
            onShowVisualization={stageData.stage_number === 6 ? handleShowVisualization : null}
            onShowClinicalReport={(stageData.stage_number === 6 || stageData.stage_number === 7) ? handleShowClinicalReport : null}
            />
        ))
        ) : (
        <p style={{ textAlign: 'center', color: '#999' }}>
            Ожидание запуска этапов...
        </p>
        )}

      {/* Кнопка обновления (если соединение потеряно) */}
      {error && (
        <Button
          icon={<ReloadOutlined />}
          onClick={fetchInitialStatus}
          block
          style={{ marginTop: 16 }}
        >
          Обновить статус
        </Button>
      )}
      
      {/* Модальное окно отчёта о качестве */}
      <QualityReport
        runId={runId}
        visible={showQualityReport}
        onClose={handleCloseQualityReport}
      />
      {/* Модальное окно 3D визуализации */}
      <NIfTIViewer
        runId={runId}
        visible={showVisualization}
        onClose={handleCloseVisualization}
      />
      <ClinicalReport
        runId={runId}
        visible={showClinicalReport}
        onClose={handleCloseClinicalReport}
      />
    </Card>
  );
};

export default ProgressMonitor;