/**
 * Панель валидации — список сессий, доступных для проверки врачом-экспертом
 */
import { useState, useEffect } from 'react';
import { Card, Table, Tag, Button, Alert, Spin, Space, Typography } from 'antd';
import { CheckCircleOutlined, EyeOutlined, ReloadOutlined } from '@ant-design/icons';
import { getValidationEntities, getLesionTypes } from '../services/api';

const { Text } = Typography;

const STATUS_COLORS = {
  New: 'default',
  Labeled: 'blue',
  'Under Verification': 'orange',
  Verified: 'green',
  'Self-Verified': 'cyan',
};

const ValidationPanel = ({ onViewEntity }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [entities, setEntities] = useState([]);
  const [lesionTypes, setLesionTypes] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState(null);

  useEffect(() => {
    loadLesionTypes();
  }, []);

  useEffect(() => {
    if (selectedDatasetId) {
      loadEntities(selectedDatasetId);
    }
  }, [selectedDatasetId]);

  const loadLesionTypes = async () => {
    try {
      const types = await getLesionTypes();
      setLesionTypes(types);
      // Выбираем первый тип с привязанным dataset_id
      const first = types.find((t) => t.dataset_id);
      if (first) {
        setSelectedDatasetId(first.dataset_id);
      }
    } catch (err) {
      console.error('Ошибка загрузки типов поражений:', err);
      setError('Не удалось загрузить типы поражений');
    }
  };

  const loadEntities = async (datasetId) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getValidationEntities(datasetId);
      setEntities(data.entities || []);
    } catch (err) {
      console.error('Ошибка загрузки сессий:', err);
      setError('Не удалось загрузить список сессий');
    } finally {
      setLoading(false);
    }
  };

  const columns = [
    {
      title: 'Идентификатор',
      dataIndex: 'dsEntityName',
      key: 'name',
      render: (name) => <Text strong>{name}</Text>,
    },
    {
      title: 'Тип поражения',
      key: 'lesion',
      render: (_, entity) => {
        const lt = entity.dsEntityInfo?.lesion_type;
        return lt ? <Tag>{lt}</Tag> : '—';
      },
    },
    {
      title: 'Модальности',
      key: 'modalities',
      render: (_, entity) => {
        const mods = entity.dsEntityInfo?.modalities || [];
        return (
          <Space size={4}>
            {mods.map((m) => (
              <Tag key={m}>{m.toUpperCase()}</Tag>
            ))}
          </Space>
        );
      },
    },
    {
      title: 'Объём опухоли',
      key: 'volume',
      render: (_, entity) => {
        const v = entity.dsEntityInfo?.volume_report?.total_tumor_cm3;
        return v != null ? `${v.toFixed(2)} см³` : '—';
      },
    },
    {
      title: 'Статус',
      key: 'status',
      render: (_, entity) => {
        const status = entity.entityStatusIntrep || 'New';
        return <Tag color={STATUS_COLORS[status] || 'default'}>{status}</Tag>;
      },
    },
    {
      title: 'Дата загрузки',
      dataIndex: 'createdOn',
      key: 'created',
      render: (date) => date ? new Date(date).toLocaleString('ru-RU') : '—',
    },
    {
      title: 'Действия',
      key: 'actions',
      render: (_, entity) => (
        <Space>
          <Button
            type="primary"
            icon={<EyeOutlined />}
            size="small"
            onClick={() => onViewEntity && onViewEntity(entity, selectedDatasetId)}
          >
            Просмотреть
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <Card
      title={
        <Space>
          <CheckCircleOutlined />
          <span>Сессии для валидации</span>
        </Space>
      }
      extra={
        <Button
          icon={<ReloadOutlined />}
          onClick={() => selectedDatasetId && loadEntities(selectedDatasetId)}
          loading={loading}
        >
          Обновить
        </Button>
      }
    >
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

      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
          <p style={{ marginTop: 16 }}>Загрузка сессий...</p>
        </div>
      ) : (
        <Table
          columns={columns}
          dataSource={entities}
          rowKey={(entity) => entity.dsEntityId}
          pagination={{ pageSize: 20 }}
          locale={{ emptyText: 'Нет доступных сессий' }}
        />
      )}
    </Card>
  );
};

export default ValidationPanel;