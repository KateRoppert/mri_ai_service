/**
 * Компонент для отображения прогресса одного этапа
 */
import { Progress, Space, Tag, Button } from 'antd';
import { 
  CheckCircleOutlined, 
  CloseCircleOutlined, 
  SyncOutlined,
  ClockCircleOutlined,
  FileTextOutlined,
  EyeOutlined,
  PieChartOutlined,
  EnvironmentOutlined,
} from '@ant-design/icons';

const StageProgress = ({ stageNumber, stageName, status, progress, onShowQualityReport, onShowVisualization, onShowVolumeReport, onShowLobarReport }) => {
  /**
   * Определяем цвет и иконку в зависимости от статуса
   */
  const getStatusConfig = () => {
    switch (status) {
      case 'completed':
        return {
          color: 'success',
          icon: <CheckCircleOutlined />,
          text: 'Завершён',
          progressStatus: 'success',
        };
      case 'running':
        return {
          color: 'processing',
          icon: <SyncOutlined spin />,
          text: 'Выполняется',
          progressStatus: 'active',
        };
      case 'failed':
        return {
          color: 'error',
          icon: <CloseCircleOutlined />,
          text: 'Ошибка',
          progressStatus: 'exception',
        };
      case 'pending':
      default:
        return {
          color: 'default',
          icon: <ClockCircleOutlined />,
          text: 'Ожидание',
          progressStatus: 'normal',
        };
    }
  };

  const statusConfig = getStatusConfig();
  
  // Показываем кнопку отчёта только для 3-го этапа после завершения
  const showQualityButton = stageNumber === 3 && status === 'completed' && onShowQualityReport;
  
  // Показываем кнопку визуализации только для 5-го этапа после завершения
  const showVisualizationButton = stageNumber === 6 && status === 'completed' && onShowVisualization;
  const showVolumeButton = stageNumber === 6 && status === 'completed' && onShowVolumeReport;
  const showLobarButton = stageNumber === 7 && status === 'completed' && onShowLobarReport;

  return (
    <div style={{ marginBottom: 16 }}>
      <Space style={{ width: '100%', marginBottom: 8, justifyContent: 'space-between' }}>
        <Space size="middle">
          <Tag color={statusConfig.color} icon={statusConfig.icon}>
            {statusConfig.text}
          </Tag>
          <strong>Этап {stageNumber}:</strong>
          <span>{stageName}</span>
        </Space>
        
        <Space>
          {/* Кнопка отчёта о качестве */}
          {showQualityButton && (
            <Button
              type="link"
              size="small"
              icon={<FileTextOutlined />}
              onClick={onShowQualityReport}
            >
              Просмотреть отчёт
            </Button>
          )}

          {/* Кнопка отчёта об объёмах */}
          {showVolumeButton && (
            <Button
              type="link"
              size="small"
              icon={<PieChartOutlined />}
              onClick={onShowVolumeReport}
            >
              Объёмы опухоли
            </Button>
          )}

          {showLobarButton && (
            <Button
              type="link"
              size="small"
              icon={<EnvironmentOutlined />}
              onClick={onShowLobarReport}
            >
              Локализация
            </Button>
          )}
          
          {/* Кнопка визуализации */}
          {showVisualizationButton && (
            <Button
              type="primary"
              size="small"
              icon={<EyeOutlined />}
              onClick={onShowVisualization}
            >
              3D Визуализация
            </Button>
          )}
        </Space>
      </Space>
      
      <Progress
        percent={progress}
        status={statusConfig.progressStatus}
        strokeColor={
          status === 'completed' ? '#52c41a' :
          status === 'running' ? '#1890ff' :
          status === 'failed' ? '#ff4d4f' : '#d9d9d9'
        }
      />
    </div>
  );
};

export default StageProgress;