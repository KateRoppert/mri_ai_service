/**
 * Компонент для отображения прогресса одного этапа
 */
import { Progress, Space, Tag } from 'antd';
import { 
  CheckCircleOutlined, 
  CloseCircleOutlined, 
  SyncOutlined,
  ClockCircleOutlined 
} from '@ant-design/icons';

const StageProgress = ({ stageNumber, stageName, status, progress }) => {
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

  return (
    <div style={{ marginBottom: 16 }}>
      <Space style={{ width: '100%', marginBottom: 8 }} size="middle">
        <Tag color={statusConfig.color} icon={statusConfig.icon}>
          {statusConfig.text}
        </Tag>
        <strong>Этап {stageNumber}:</strong>
        <span>{stageName}</span>
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