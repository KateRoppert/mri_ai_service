/**
 * Компонент для отображения истории запусков pipeline
 */
import { useState, useEffect } from 'react';
import { Table, Tag, Space, Button, Select, Card, message } from 'antd';
import { 
  EyeOutlined, 
  FileTextOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  MedicineBoxOutlined,
} from '@ant-design/icons';
import { getPipelineHistory } from '../services/api';

const PipelineHistory = ({ onShowVisualization, onShowQualityReport, onShowClinicalReport }) => {
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [total, setTotal] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(20);
  const [statusFilter, setStatusFilter] = useState('all');

  /**
   * Загружаем историю при монтировании и при изменении фильтров
   */
  useEffect(() => {
    fetchHistory();
  }, [currentPage, statusFilter]);

  /**
   * Получить историю запусков
   */
  const fetchHistory = async () => {
    setLoading(true);
    
    try {
      const offset = (currentPage - 1) * pageSize;
      const data = await getPipelineHistory(pageSize, offset);
      
      // Фильтруем по статусу если выбран фильтр
      let filteredRuns = data.runs || [];
      if (statusFilter !== 'all') {
        filteredRuns = filteredRuns.filter(run => run.status === statusFilter);
      }
      
      setHistory(filteredRuns);
      setTotal(statusFilter === 'all' ? data.total : filteredRuns.length);
    } catch (err) {
      console.error('Ошибка загрузки истории:', err);
      message.error('Не удалось загрузить историю запусков');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Получить конфигурацию для отображения статуса
   */
  const getStatusConfig = (status) => {
    switch (status) {
      case 'completed':
        return {
          color: 'success',
          icon: <CheckCircleOutlined />,
          text: 'Завершён',
        };
      case 'running':
        return {
          color: 'processing',
          icon: <SyncOutlined spin />,
          text: 'Выполняется',
        };
      case 'failed':
        return {
          color: 'error',
          icon: <CloseCircleOutlined />,
          text: 'Ошибка',
        };
      case 'pending':
      default:
        return {
          color: 'default',
          icon: <SyncOutlined />,
          text: 'Ожидание',
        };
    }
  };

  /**
   * Форматирование даты
   */
  const formatDate = (dateString) => {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return date.toLocaleString('ru-RU', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  /**
    * Форматировать длительность из секунд
    */
  const formatDuration = (durationSeconds) => {
  if (!durationSeconds) return '-';

  const minutes = Math.floor(durationSeconds / 60);
  const seconds = durationSeconds % 60;

  return `${minutes}м ${seconds}с`;
  };

  /**
   * Колонки таблицы
   */
  const columns = [
    {
      title: 'ID',
      dataIndex: 'run_id',
      key: 'run_id',
      width: 100,
      render: (id) => (
        <span style={{ fontFamily: 'monospace', fontSize: 11 }}>
          {id.substring(0, 8)}...
        </span>
      ),
    },
    {
      title: 'Дата запуска',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (date) => formatDate(date),
      sorter: (a, b) => new Date(a.created_at) - new Date(b.created_at),
    },
    {
      title: 'Статус',
      dataIndex: 'status',
      key: 'status',
      width: 130,
      render: (status) => {
        const config = getStatusConfig(status);
        return (
          <Tag color={config.color} icon={config.icon}>
            {config.text}
          </Tag>
        );
      },
    },
    {
      title: 'Входные данные',
      dataIndex: 'input_path',
      key: 'input_path',
      ellipsis: true,
      render: (path) => (
        <span style={{ fontSize: 12 }} title={path}>
          {path}
        </span>
      ),
    },
    {
      title: 'Качество',
      key: 'quality',
      width: 100,
      render: (_, record) => {
        if (!record.quality_score) return '-';
        
        const score = record.quality_score;
        let color = 'default';
        if (score >= 80) color = 'success';
        else if (score >= 60) color = 'warning';
        else color = 'error';
        
        return (
          <Tag color={color}>
            {score.toFixed(1)}
          </Tag>
        );
      },
      sorter: (a, b) => (a.quality_score || 0) - (b.quality_score || 0),
    },
    {
      title: 'Длительность',
      key: 'duration',
      width: 100,
      render: (_, record) => formatDuration(record.duration_seconds),
    },
    {
      title: 'Действия',
      key: 'actions',
      width: 120,
      render: (_, record) => (
        <Space direction="vertical" size={2}>
          {record.status === 'completed' && record.current_stage >= 3 && (
            <Button
              type="link"
              size="small"
              icon={<FileTextOutlined />}
              onClick={() => onShowQualityReport(record.run_id)}
            >
              Отчёт
            </Button>
          )}
          {record.status === 'completed' && record.current_stage >= 7 && (
            <Button
              type="link"
              size="small"
              icon={<MedicineBoxOutlined />}
              onClick={() => onShowClinicalReport(record.run_id)}
            >
              Отчёт
            </Button>
          )}
          {record.status === 'completed' && record.current_stage >= 6 && (
            <Button
              type="primary"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => onShowVisualization(record.run_id)}
            >
              3D
            </Button>
          )}
        </Space>
      ),
    },
  ];

  return (
    <Card 
      title="История запусков"
      extra={
        <Space>
          <Select
            value={statusFilter}
            onChange={setStatusFilter}
            style={{ width: 150 }}
            options={[
              { label: 'Все статусы', value: 'all' },
              { label: 'Завершённые', value: 'completed' },
              { label: 'Выполняются', value: 'running' },
              { label: 'С ошибками', value: 'failed' },
            ]}
          />
          <Button
            icon={<ReloadOutlined />}
            onClick={fetchHistory}
            loading={loading}
          >
            Обновить
          </Button>
        </Space>
      }
    >
      <Table
        columns={columns}
        dataSource={history}
        rowKey="run_id"
        loading={loading}
        pagination={{
          current: currentPage,
          pageSize: pageSize,
          total: total,
          onChange: setCurrentPage,
          showSizeChanger: false,
          showTotal: (total) => `Всего запусков: ${total}`,
        }}
      />
    </Card>
  );
};

export default PipelineHistory;