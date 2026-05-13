/**
 * Панель действий валидации: Подтвердить / Отклонить / Редактировать
 * Показывает текущее состояние и историю голосов.
 * Кнопка «Редактировать» открывает модалку:
 *   - Скачать Slicer-пакет → редактировать в 3D Slicer → загрузить маску обратно.
 */
import { useState, useEffect, useRef } from 'react';
import {
  Button, Space, Tag, Tooltip, message, Popconfirm,
  Modal, Typography, Alert, Spin, List,
} from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  EditOutlined,
  UndoOutlined,
  HistoryOutlined,
} from '@ant-design/icons';
import {
  validationAction,
  getEntityValidation,
  getMaskVersions,
  getMaskFileUrl,
  syncMasks,
  checkSlicerAgent,
  openInSlicer,
} from '../services/api';

const { Text } = Typography;

const STATUS_TAG = {
  3: { text: 'Размечено', color: 'blue' },
  4: { text: 'На проверке', color: 'orange' },
  5: { text: 'Верифицировано', color: 'green' },
};

const ValidationActions = ({ entityId, datasetId, runId, onStatusChange, onMaskUploaded, onCloseViewer, onMaskVersionSelect }) => {
  const [loading, setLoading] = useState(false);
  const [votes, setVotes] = useState(null);
  const [myVote, setMyVote] = useState(null);

  // Slicer Agent
  const [slicerStatus, setSlicerStatus] = useState(null); // null | 'checking' | 'available' | 'unavailable'
  const [slicerOpening, setSlicerOpening] = useState(false);
  const [slicerResult, setSlicerResult] = useState(null);

  // История версий
  const [versionsModalOpen, setVersionsModalOpen] = useState(false);
  const [versions, setVersions] = useState([]);
  const [versionsLoading, setVersionsLoading] = useState(false);
  const [activeVersion, setActiveVersion] = useState(1); // v1 = ИИ маска, показывается по умолчанию

  // Polling для Slicer (автообновление после отправки маски из Slicer)
  const slicerPollRef = useRef(null);
  const lastVersionCountRef = useRef(null);

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

  // === Редактирование ===

  const handleEditClick = async () => {
    if (!runId) {
      message.warning('Не удалось определить ID запуска');
      return;
    }

    // Проверяем доступность Slicer Agent
    try {
      const agentStatus = await checkSlicerAgent();
      if (!agentStatus.slicer_found) {
        message.error('3D Slicer Agent не запущен. Запустите slicer_agent.py на хост-машине.');
        return;
      }
    } catch {
      message.error('Не удалось проверить доступность 3D Slicer');
      return;
    }

    // Открываем Slicer с выбранной маской
    await handleOpenInSlicer();
  };

  const handleOpenInSlicer = async () => {
    if (!runId) {
      message.warning('Не удалось определить ID запуска');
      return;
    }
    setSlicerOpening(true);
    setSlicerResult(null);
    try {
      const result = await openInSlicer(runId, activeVersion);
      setSlicerResult(result);
      message.success('3D Slicer открыт с данными пациента');
      // Запускаем polling: проверяем, не появилась ли новая версия маски
      startSlicerPoll();
    } catch (err) {
      console.error('Ошибка запуска Slicer:', err);
      const detail = err.response?.data?.detail || 'Не удалось открыть Slicer';
      message.error(detail);
      setSlicerResult({ success: false, message: detail });
    } finally {
      setSlicerOpening(false);
    }
  };

  /** Polling: каждые 5с проверяем, появилась ли новая версия маски (от Slicer Agent) */
  const startSlicerPoll = () => {
    stopSlicerPoll();
    // Запоминаем текущее кол-во версий
    getMaskVersions(entityId)
      .then((data) => {
        lastVersionCountRef.current = (data.versions || []).length;
      })
      .catch(() => {});

    slicerPollRef.current = setInterval(async () => {
      try {
        const data = await getMaskVersions(entityId);
        const newCount = (data.versions || []).length;
        if (lastVersionCountRef.current !== null && newCount > lastVersionCountRef.current) {
          // Появилась новая маска
          lastVersionCountRef.current = newCount;
          message.success('Получена новая версия маски от 3D Slicer');
          setVersions(data.versions || []);
          if (onMaskUploaded) {
            onMaskUploaded({ fromSlicer: true, versions: data.versions });
          }
          stopSlicerPoll();
        }
      } catch {
        // Игнорируем ошибки polling
      }
    }, 5000);

    // Автоостановка через 2 минуты
    setTimeout(() => stopSlicerPoll(), 120000);
  };

  const stopSlicerPoll = () => {
    if (slicerPollRef.current) {
      clearInterval(slicerPollRef.current);
      slicerPollRef.current = null;
    }
  };

  // Cleanup polling при размонтировании или закрытии модалки
  useEffect(() => {
    return () => stopSlicerPoll();
  }, []);

  // === История версий ===

  const openVersionsModal = async () => {
    setVersionsModalOpen(true);
    setVersionsLoading(true);
    try {
      const data = await getMaskVersions(entityId);
      const versionsList = data.versions || [];
      setVersions(versionsList);
      // Текущая активная = последняя
      if (versionsList.length > 0 && !activeVersion) {
        setActiveVersion(versionsList[versionsList.length - 1].version);
      }
    } catch (err) {
      console.error('Ошибка загрузки версий:', err);
      message.error('Не удалось загрузить историю версий');
    } finally {
      setVersionsLoading(false);
    }
  };

  const handleVersionClick = (version) => {
    setActiveVersion(version.version);
    const maskUrl = getMaskFileUrl(entityId, version.version);
    if (onMaskVersionSelect) {
      onMaskVersionSelect({ ...version, maskUrl });
    }
    setVersionsModalOpen(false);
  };

  const handleSyncMasks = async () => {
    setVersionsLoading(true);
    try {
      const result = await syncMasks(entityId);
      if (result.removed > 0) {
        message.success(`Удалено ${result.removed} осиротевших версий`);
        // Перезагружаем список
        const data = await getMaskVersions(entityId);
        setVersions(data.versions || []);
      } else {
        message.info('Все версии актуальны');
      }
    } catch (err) {
      console.error('Ошибка синхронизации:', err);
      message.error('Не удалось синхронизировать');
    } finally {
      setVersionsLoading(false);
    }
  };

  // === Рендер ===

  if (!entityId) return null;

  const confirms = votes?.confirms_count || 0;
  const rejects = votes?.rejects_count || 0;
  const kappaStatus =
    confirms >= 2 ? 5 : (confirms + rejects >= 1 ? 4 : 3);
  const statusInfo = STATUS_TAG[kappaStatus];

  return (
    <>
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

        <Button
          icon={<EditOutlined />}
          onClick={handleEditClick}
          disabled={!runId}
          loading={slicerOpening}
        >
          Редактировать
        </Button>

        <Tooltip title="История версий маски">
          <Button
            icon={<HistoryOutlined />}
            size="small"
            onClick={openVersionsModal}
          />
        </Tooltip>
      </Space>

      {/* Модалка истории версий */}
      <Modal
        title="История версий маски"
        open={versionsModalOpen}
        onCancel={() => setVersionsModalOpen(false)}
        getContainer={document.body}
        footer={
          <Button size="small" onClick={handleSyncMasks} loading={versionsLoading}>
            Синхронизировать с Каппой
          </Button>
        }
        width={500}
      >
        {versionsLoading ? (
          <div style={{ textAlign: 'center', padding: 24 }}>
            <Spin />
          </div>
        ) : versions.length === 0 ? (
          <Text type="secondary">Версии масок не найдены</Text>
        ) : (
          <>
            <Text type="secondary" style={{ display: 'block', marginBottom: 8, fontSize: 12 }}>
              Нажмите на версию, чтобы переключить визуализацию
            </Text>
          <List
            dataSource={versions}
            renderItem={(v) => (
              <List.Item
                onClick={() => v.available !== false && handleVersionClick(v)}
                style={{
                  cursor: v.available === false ? 'not-allowed' : 'pointer',
                  background: v.version === activeVersion ? '#e6f7ff' : 'transparent',
                  borderLeft: v.version === activeVersion ? '3px solid #1890ff' : '3px solid transparent',
                  paddingLeft: 12,
                  opacity: v.available === false ? 0.45 : 1,
                  transition: 'background 0.2s',
                }}
                onMouseEnter={(e) => {
                  if (v.version !== activeVersion && v.available !== false) e.currentTarget.style.background = '#fafafa';
                }}
                onMouseLeave={(e) => {
                  if (v.version !== activeVersion) e.currentTarget.style.background = 'transparent';
                }}
              >
                <List.Item.Meta
                  title={
                    <Space>
                      <Tag color={v.source === 'ai' ? 'blue' : 'green'}>
                        {v.source === 'ai' ? 'ИИ' : 'Эксперт'}
                      </Tag>
                      <Text strong>Версия {v.version}</Text>
                      {v.version === activeVersion && (
                        <Tag color="blue" style={{ marginLeft: 4 }}>текущая</Tag>
                      )}
                      {v.available === false && (
                        <Tag color="red" style={{ marginLeft: 4 }}>недоступна</Tag>
                      )}
                    </Space>
                  }
                  description={
                    <>
                      <Text type="secondary">
                        {v.uploaded_by_name || 'AI Pipeline'}
                        {' — '}
                        {v.created_at
                          ? new Date(v.created_at).toLocaleString('ru-RU')
                          : '—'}
                      </Text>
                      <br />
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {v.file_name}
                        {v.file_size != null && (
                          <> · {(v.file_size / 1024).toFixed(1)} КБ</>
                        )}
                      </Text>
                    </>
                  }
                />
              </List.Item>
            )}
          />
          </>
        )}
      </Modal>
    </>
  );
};

export default ValidationActions;