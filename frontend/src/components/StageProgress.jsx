/**
 * Компонент для отображения прогресса одного этапа
 */
import { Progress, Space, Tag, Button } from 'antd';
import { 
  CheckCircleOutlined, 
  CloseCircleOutlined, 
  SyncOutlined,
  ClockCircleOutlined,
  FileTextOutlined
} from '@ant-design/icons';

const StageProgress = ({ stageNumber, stageName, status, progress, onShowQualityReport }) => {
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
  
  // Показываем кнопку отчёта только для 4-го этапа после завершения
  const showQualityButton = stageNumber === 4 && status === 'completed' && onShowQualityReport;

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