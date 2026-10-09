/**
 * Модальное окно со списком сессий текущего запуска, требующих
 * внимания врача — неполные, и полные, где есть альтернативные
 * (исключённые) серии-кандидаты.
 */
import { useState, useEffect } from 'react';
import { Modal, Table, Tag, Space, Button, Alert, Spin, message, Popconfirm } from 'antd';
import { ReloadOutlined, SyncOutlined } from '@ant-design/icons';
import {
  getIncompletePatients, requeuePipelineRun, cancelQueuedRequeue,
} from '../services/api';
import IncompletePatientDetail from './IncompletePatientDetail';

const IncompletePatients = ({ runId, visible, onClose, canRequeue = true, onRequeued }) => {
  const [loading, setLoading] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [error, setError] = useState(null);
  const [selectedSession, setSelectedSession] = useState(null);
  const [requeuing, setRequeuing] = useState(false);
  // Приходит с сервера, а не держится в этой вкладке: иначе врач,
  // обновивший страницу, не узнал бы, что запуск уже стоит в очереди.
  const [queued, setQueued] = useState(false);
  // Занятость пути знает сервер: canRequeue выводится из статуса этого
  // прогона и для остановленного тоже ложен, хотя путь свободен.
  const [pathBusy, setPathBusy] = useState(!canRequeue);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    if (visible && runId) {
      fetchSessions();
    }
  }, [visible, runId]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchSessions = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getIncompletePatients(runId);
      const newSessions = data.sessions || [];
      setSessions(newSessions);
      setQueued(data.queued_requeue === true);
      setPathBusy(data.path_busy === true);
      setSelectedSession((prev) => {
        if (!prev) return prev;
        const updated = newSessions.find(
          (s) => s.patient_id === prev.patient_id && s.session_id === prev.session_id
        );
        return updated || null;
      });
    } catch (err) {
      console.error('Ошибка загрузки списка неполных пациентов:', err);
      setError('Не удалось загрузить список');
    } finally {
      setLoading(false);
    }
  };

  const handleCancelQueued = async () => {
    setCancelling(true);
    try {
      const result = await cancelQueuedRequeue(runId);
      setQueued(false);
      message.success(
        result.cancelled
          ? 'Запуск снят с очереди'
          : 'В очереди ничего не было',
      );
    } catch (err) {
      console.error('Ошибка снятия с очереди:', err);
      message.error('Не удалось снять запуск с очереди');
    } finally {
      setCancelling(false);
    }
  };

  const handleRequeue = async () => {
    setRequeuing(true);
    try {
      const result = await requeuePipelineRun(runId);
      if (result.queued) {
        // Путь занят — запуск встал в очередь. Модалку не закрываем: врач
        // должен увидеть, что произошло, и иметь возможность отменить.
        setQueued(true);
        message.info(
          'Запуск поставлен в очередь — начнётся после завершения текущего.', 6,
        );
        return;
      }
      message.success(
        `Обработка перезапущена (run_id: ${result.run_id.substring(0, 8)}...). ` +
        'Отслеживайте прогресс во вкладке «История запусков».'
      );
      onClose();
      if (onRequeued) {
        onRequeued(result);
      }
    } catch (err) {
      console.error('Ошибка перезапуска:', err);
      const detail = err.response?.data?.detail;
      if (err.response?.status === 409 && detail) {
        message.error(detail);
      } else {
        message.error('Не удалось перезапустить обработку');
      }
    } finally {
      setRequeuing(false);
    }
  };

  const sortedSessions = [...sessions].sort((a, b) => {
    if (a.original_id !== b.original_id) return a.original_id.localeCompare(b.original_id);
    return a.session_id.localeCompare(b.session_id);
  });

  const patientRowSpans = {};
  sortedSessions.forEach((s, idx) => {
    if (idx === 0 || sortedSessions[idx - 1].original_id !== s.original_id) {
      let count = 1;
      while (sortedSessions[idx + count] && sortedSessions[idx + count].original_id === s.original_id) {
        count++;
      }
      patientRowSpans[idx] = count;
    } else {
      patientRowSpans[idx] = 0;
    }
  });

  const columns = [
    {
      title: 'Пациент',
      dataIndex: 'original_id',
      key: 'original_id',
      render: (value, record, index) => ({
        children: value,
        props: { rowSpan: patientRowSpans[index] },
      }),
    },
    { title: 'Сессия', dataIndex: 'session_id', key: 'session_id' },
    { title: 'Дата', dataIndex: 'date', key: 'date' },
    {
      title: 'Статус',
      dataIndex: 'status',
      key: 'status',
      render: (status, record) => {
        if (status === 'incomplete') {
          return <Tag color="orange">Неполная</Tag>;
        }
        if (status === 'discarded') {
          return <Tag color="default">Отброшена</Tag>;
        }
        if (status === 'merged') {
          return <Tag color="default">Объединена → {record.merged_into_session_id}</Tag>;
        }
        const hasAlternatives = (record.excluded_series || []).length > 0;
        return hasAlternatives
          ? <Tag color="blue">Есть альтернативы</Tag>
          : <Tag color="green">Полная</Tag>;
      },
    },
    {
      title: 'Модальности',
      dataIndex: 'available',
      key: 'available',
      render: (available) => (
        <Space wrap>
          {available.map((m) => <Tag color="green" key={m}>{m}</Tag>)}
        </Space>
      ),
    },
    {
      title: '',
      key: 'actions',
      render: (_, record) => (
        <Button size="small" onClick={() => setSelectedSession(record)}>
          Подробнее
        </Button>
      ),
    },
  ];

  return (
    <Modal
      title="Пациенты, требующие внимания"
      open={visible}
      onCancel={onClose}
      width={900}
      footer={null}
    >
      {queued && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="Запуск в очереди"
          description="Начнётся автоматически, когда завершится текущая обработка."
          action={
            <Button size="small" loading={cancelling} onClick={handleCancelQueued}>
              Отменить
            </Button>
          }
        />
      )}

      <Space style={{ marginBottom: 16 }}>
        {/* Кнопка доступна всегда. Пока путь занят, запуск встаёт в очередь —
            врачу незачем возвращаться позже и вспоминать, что он хотел. */}
        <Popconfirm
          title={pathBusy
            ? 'Сейчас идёт обработка. Поставить запуск в очередь — начнётся после её завершения?'
            : 'Перезапустить обработку? Уже обработанные пациенты будут пропущены.'}
          onConfirm={handleRequeue}
          okText="Да"
          cancelText="Нет"
        >
          <Button type="primary" icon={<SyncOutlined />} loading={requeuing}>
            {pathBusy ? 'Поставить в очередь' : 'Запустить обработку'}
          </Button>
        </Popconfirm>
        <Button icon={<ReloadOutlined />} onClick={fetchSessions}>
          Обновить
        </Button>
      </Space>

      {error && <Alert type="error" description={error} showIcon style={{ marginBottom: 16 }} />}

      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
        </div>
      ) : (
        <Table
          columns={columns}
          dataSource={sortedSessions}
          rowKey={(r) => `${r.patient_id}_${r.session_id}`}
          pagination={false}
          locale={{ emptyText: 'Нет сессий, требующих внимания' }}
        />
      )}

      <IncompletePatientDetail
        runId={runId}
        session={selectedSession}
        sessions={sessions}
        visible={!!selectedSession}
        onClose={() => setSelectedSession(null)}
        onActionComplete={fetchSessions}
      />
    </Modal>
  );
};

export default IncompletePatients;
