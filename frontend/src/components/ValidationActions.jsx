/**
 * Панель действий валидации: Подтвердить / Отклонить / Редактировать
 * Показывает текущее состояние и историю голосов.
 */
import { useState, useEffect } from 'react';
import { Button, Space, Tag, Tooltip, message, Popconfirm } from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  EditOutlined,
  UndoOutlined,
} from '@ant-design/icons';
import { validationAction, getEntityValidation } from '../services/api';

const STATUS_TAG = {
  3: { text: 'Размечено', color: 'blue' },
  4: { text: 'На проверке', color: 'orange' },
  5: { text: 'Верифицировано', color: 'green' },
};

const ValidationActions = ({ entityId, datasetId, onStatusChange }) => {
  const [loading, setLoading] = useState(false);
  const [votes, setVotes] = useState(null);
  const [myVote, setMyVote] = useState(null);

  useEffect(() => {
    if (entityId) {
      loadState();
    }
  }, [entityId]);

  const loadState = async () => {
    try {
      const data = await getEntityValidation(entityId);
      setVotes(data.votes);
      setMyVote(data.my_vote);
    } catch (err) {
      console.error('Ошибка загрузки состояния валидации:', err);
    }
  };

  const doAction = async (action) => {
    setLoading(true);
    try {
      const result = await validationAction(datasetId, entityId, action);
      setVotes(result.votes);
      setMyVote(action === 'revoke' ? null : action);

      const messages = {
        confirm: 'Сегментация подтверждена',
        reject: 'Сегментация отклонена',
        revoke: 'Решение отозвано',
      };
      message.success(messages[action]);

      if (onStatusChange) {
        onStatusChange(result);
      }
    } catch (err) {
      console.error('Ошибка:', err);
      message.error('Не удалось выполнить действие');
    } finally {
      setLoading(false);
    }
  };

  if (!entityId) return null;

  const confirms = votes?.confirms_count || 0;
  const rejects = votes?.rejects_count || 0;
  const kappaStatus =
    confirms >= 2 ? 5 : (confirms + rejects >= 1 ? 4 : 3);
  const statusInfo = STATUS_TAG[kappaStatus];

  return (
    <Space size="middle" align="center">
      <Tooltip
        title={
          <>
            Подтверждений: {confirms}
            <br />
            Отклонений: {rejects}
          </>
        }
      >
        <Tag color={statusInfo.color} style={{ fontSize: 13, padding: '4px 10px' }}>
          {statusInfo.text}
        </Tag>
      </Tooltip>

      {myVote === 'confirm' || myVote === 'reject' ? (
        <>
          <Tag
            icon={myVote === 'confirm' ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
            color={myVote === 'confirm' ? 'green' : 'red'}
          >
            Ваше решение: {myVote === 'confirm' ? 'подтверждено' : 'отклонено'}
          </Tag>
          <Popconfirm
            title="Отозвать ваше решение?"
            onConfirm={() => doAction('revoke')}
            okText="Да"
            cancelText="Нет"
          >
            <Button icon={<UndoOutlined />} size="small" loading={loading}>
              Отозвать
            </Button>
          </Popconfirm>
        </>
      ) : (
        <>
          <Button
            type="primary"
            icon={<CheckCircleOutlined />}
            onClick={() => doAction('confirm')}
            loading={loading}
          >
            Подтвердить
          </Button>
          <Button
            danger
            icon={<CloseCircleOutlined />}
            onClick={() => doAction('reject')}
            loading={loading}
          >
            Отклонить
          </Button>
        </>
      )}

      <Tooltip title="Редактирование сегментации — в следующей версии">
        <Button icon={<EditOutlined />} disabled>
          Редактировать
        </Button>
      </Tooltip>
    </Space>
  );
};

export default ValidationActions;