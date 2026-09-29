/**
 * Компонент для отображения истории запусков pipeline
 */
import { useState, useEffect, useRef } from 'react';
import { Table, Tag, Space, Button, Select, Card, message, Modal, List, Tooltip, Alert } from 'antd';
import { 
  EyeOutlined, 
  FileTextOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  MedicineBoxOutlined,
  PauseCircleOutlined,
  CloudUploadOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import {
  getPipelineHistory,
  retryKappaUpload,
  getKappaDeliverySummary,
} from '../services/api';
import { confirmAndResume } from '../utils/resumeRun';

const PipelineHistory = ({ onShowVisualization, onShowQualityReport, onShowClinicalReport, onShowIncompletePatients, onShowPipelineLosses, onRunResumed }) => {
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [total, setTotal] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(20);
  const [statusFilter, setStatusFilter] = useState('all');
  const [deliverySummary, setDeliverySummary] = useState(null);
  const [deliveryDetail, setDeliveryDetail] = useState(null);
  const [retrying, setRetrying] = useState(false);

  /**
   * Загружаем историю при монтировании и при изменении фильтров
   */
  useEffect(() => {
    fetchHistory();
  }, [currentPage, statusFilter]); // eslint-disable-line react-hooks/exhaustive-deps

  // Track "is anything running" in a ref so the poll interval can read the
  // latest value without being torn down and recreated on every history update.
  const hasRunningRef = useRef(false);
  const hasPendingDeliveryRef = useRef(false);
  useEffect(() => {
    hasRunningRef.current = history.some(run => run.status === 'running');
    hasPendingDeliveryRef.current = history.some(
      (run) => run.kappa_upload?.status === 'pending',
    );
  }, [history]);

  // One stable 5s interval per page/filter. It refetches only while something is
  // actually running (read via the ref). Previously `history` was in the deps,
  // so every fetch recreated the interval; combined with runs stuck in 'running'
  // that never resolved, this polled the backend without end.
  useEffect(() => {
    const interval = setInterval(() => {
      if (hasRunningRef.current || hasPendingDeliveryRef.current) {
        fetchHistory({ silent: true });
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [currentPage, statusFilter]); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Получить историю запусков
   */
  const fetchHistory = async (opts = {}) => {
    const silent = opts.silent === true;
    if (!silent) setLoading(true);

    try {
      const offset = (currentPage - 1) * pageSize;
      const data = await getPipelineHistory(pageSize, offset);

      let filteredRuns = data.runs || [];
      if (statusFilter !== 'all') {
        filteredRuns = filteredRuns.filter(run => run.status === statusFilter);
      }

      setHistory(filteredRuns);
      setTotal(statusFilter === 'all' ? data.total : filteredRuns.length);

      // Counts every run, not just this page — the whole point is the run
      // that failed days ago and has scrolled out of sight. A failing
      // summary must never take the history list down with it.
      try {
        setDeliverySummary(await getKappaDeliverySummary());
      } catch (summaryErr) {
        console.warn('Сводка по выгрузке недоступна:', summaryErr);
      }
    } catch (err) {
      console.error('Ошибка загрузки истории:', err);
      if (!silent) message.error('Не удалось загрузить историю запусков');
    } finally {
      if (!silent) setLoading(false);
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
      case 'stopped':
        return {
          color: 'orange',
          icon: <PauseCircleOutlined />,
          text: 'Остановлен',
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
   * Возобновить остановленный запуск. При 409 с differences — диалог
   * выбора: продолжить на сохранённых настройках или отменить.
   */
  const handleResume = async (runId) => {
    // Resuming creates a NEW run; onRunResumed lets the app switch to it so
    // the pipeline tab stops showing the stopped one.
    await confirmAndResume(runId, {
      onResumed: (result) => {
        fetchHistory();
        onRunResumed?.(result);
      },
    });
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

  /** Человеческая подпись к состоянию выгрузки в Kappa. */
  const deliveryLabel = (d) => {
    const have = `${d.delivered ?? 0} из ${d.total ?? 0}`;
    // total = 0 до первой удачной попытки значит «ещё не считали», а не
    // «отправлять нечего». Показывать «0 из 0» — врать оператору, будто
    // работы нет, тогда как её просто ещё не пересчитали.
    const counted = (d.total ?? 0) > 0;
    if (d.status === 'done') return `Kappa ${d.delivered}/${d.total}`;
    if (d.status === 'needs_attention') {
      if (!counted) return 'нужна проверка';
      return d.reason === 'stuck'
        ? `${have} · не удаётся выгрузить`
        : `${have} · нужна проверка`;
    }
    if (d.reason === 'no_session') {
      return counted
        ? `${have} в Kappa · нужен вход в Kappa`
        : 'ожидает выгрузки · нужен вход в Kappa';
    }
    if (d.reason === 'network') {
      return counted
        ? `${have} в Kappa · досылка, когда сервис будет доступен`
        : 'ожидает выгрузки · Kappa недоступна';
    }
    if (!counted) return 'ожидает выгрузки';
    if ((d.delivered ?? 0) < (d.total ?? 0)) {
      return `${have} в Kappa · загружается`;
    }
    return `${have} · досылается`;
  };

  const deliveryHint = (d) => {
    if (d.status === 'done') return 'Все сессии этого запуска есть в Kappa';
    if (d.reason === 'no_session') {
      return 'Нет входа в Kappa. Данные уйдут, как только вход будет выполнен.';
    }
    if (d.reason === 'network') {
      return 'Уже загруженные сессии на месте. Остальные уйдут сами, когда Kappa снова ответит.';
    }
    if (d.status === 'pending') {
      return (d.total ?? 0) > 0
        ? 'Часть сессий уже в Kappa, остальные загружаются сейчас.'
        : 'Счётчик появится после первой попытки выгрузки.';
    }
    return 'Показать подробности выгрузки';
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
      title: 'Kappa',
      key: 'kappa_upload',
      width: 280,
      render: (_, record) => {
        const d = record.kappa_upload;
        if (!d) return <span style={{ color: '#bbb' }}>—</span>;
        const color = d.status === 'done'
          ? 'success'
          : d.status === 'needs_attention' ? 'error' : 'processing';
        const icon = d.status === 'needs_attention'
          ? <WarningOutlined />
          : <CloudUploadOutlined />;
        return (
          <Tooltip title={deliveryHint(d)}>
            <Tag
              color={color}
              icon={icon}
              style={{ cursor: 'pointer', whiteSpace: 'normal', height: 'auto' }}
              onClick={() => setDeliveryDetail(record)}
            >
              {deliveryLabel(d)}
            </Tag>
          </Tooltip>
        );
      },
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
      width: 200,
      render: (_, record) => (
        <Space direction="vertical" size={2}>
          {record.status === 'stopped' && (
            <Button size="small" onClick={() => handleResume(record.run_id)}>
              Возобновить
            </Button>
          )}
          {record.status === 'completed' && record.current_stage >= 3 && (
            <Button
              type="link"
              size="small"
              icon={<FileTextOutlined />}
              onClick={() => onShowQualityReport(record.run_id)}
            >
              Отчёт качества
            </Button>
          )}
          {record.status === 'completed' && record.current_stage >= 7 && (
            <Button
              type="link"
              size="small"
              icon={<MedicineBoxOutlined />}
              onClick={() => onShowClinicalReport(record.run_id, record.lesion_type || 'glioblastoma')}
            >
              Клинический отчёт
            </Button>
          )}
          {record.status === 'completed' && record.current_stage >= 6 && (
            <Button
              type="primary"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => onShowVisualization(record.run_id, record.lesion_type || 'glioblastoma')}
            >
              3D
            </Button>
          )}
          {record.current_stage >= 1 && record.status !== 'pending' && (
            <Button
              type="link"
              size="small"
              onClick={() => onShowIncompletePatients(record.run_id, record.status)}
            >
              Неполные пациенты
            </Button>
          )}
          {record.current_stage >= 1 && record.status !== 'pending' && (
            <Button
              type="link"
              size="small"
              onClick={() => onShowPipelineLosses(record.run_id)}
            >
              Потерянные пациенты
            </Button>
          )}
        </Space>
      ),
    },
  ];

  return (
    <>
    {deliverySummary
      && (deliverySummary.pending > 0 || deliverySummary.needs_attention > 0) && (
      <Alert
        type={deliverySummary.needs_attention > 0 ? 'error' : 'warning'}
        showIcon
        style={{ marginBottom: 16 }}
        message={
          deliverySummary.needs_attention > 0
            ? `Требуют внимания: ${deliverySummary.needs_attention}`
            : `Ожидают выгрузки в Kappa: ${deliverySummary.pending}`
        }
        description={
          deliverySummary.needs_attention > 0
            ? (
              `Автоматический повтор не поможет для ${deliverySummary.needs_attention} `
              + `${deliverySummary.needs_attention === 1 ? 'запуска' : 'запусков'}`
              + (deliverySummary.pending > 0
                ? `; ещё ${deliverySummary.pending} досылается автоматически.`
                : '.')
              + ' Найдите их по тегу в колонке «Kappa» — они могут быть на других страницах.'
            )
            : 'Данные не потеряны — досылка идёт сама. Прогоны могут быть на других страницах истории.'
        }
      />
    )}
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
      <Modal
        open={!!deliveryDetail}
        onCancel={() => setDeliveryDetail(null)}
        title="Выгрузка в Kappa"
        footer={[
          <Button key="close" onClick={() => setDeliveryDetail(null)}>
            Закрыть
          </Button>,
          <Button
            key="retry"
            type="primary"
            loading={retrying}
            onClick={async () => {
              setRetrying(true);
              try {
                await retryKappaUpload(deliveryDetail.run_id);
                message.success('Повтор выполнен');
                setDeliveryDetail(null);
                fetchHistory();
              } catch (e) {
                message.error(
                  e?.response?.data?.detail || 'Не удалось повторить выгрузку',
                );
              } finally {
                setRetrying(false);
              }
            }}
          >
            Повторить сейчас
          </Button>,
        ]}
      >
        {deliveryDetail?.kappa_upload && (
          <>
            <p>
              {deliveryHint(deliveryDetail.kappa_upload)}
              {' '}
              {(deliveryDetail.kappa_upload.total ?? 0) > 0
                ? ` Сейчас в Kappa ${deliveryDetail.kappa_upload.delivered} из `
                  + `${deliveryDetail.kappa_upload.total}.`
                : ' Сколько сессий войдёт в выгрузку, станет известно после'
                  + ' первой попытки.'}
              {deliveryDetail.kappa_upload.next_attempt_at
                && deliveryDetail.kappa_upload.reason === 'network' && (
                <> Повтор:{' '}
                  {new Date(
                    deliveryDetail.kappa_upload.next_attempt_at,
                  ).toLocaleString('ru-RU')}.
                </>
              )}
            </p>
            {(deliveryDetail.kappa_upload.blocked || []).length > 0 && (
              <List
                size="small"
                header="Требуют внимания"
                dataSource={deliveryDetail.kappa_upload.blocked}
                renderItem={(b) => (
                  <List.Item>
                    <strong>{b.session}</strong>: {b.message || b.reason}
                  </List.Item>
                )}
              />
            )}
          </>
        )}
      </Modal>
    </>
  );
};

export default PipelineHistory;