/**
 * Панель действий валидации: Подтвердить / Отклонить / Редактировать
 * Показывает текущее состояние и историю голосов.
 * Кнопка «Редактировать» открывает модалку:
 *   - Скачать Slicer-пакет → редактировать в 3D Slicer → загрузить маску обратно.
 */
import { useState, useEffect, useRef } from 'react';
import {
  Button, Space, Tag, Tooltip, message, Popconfirm,
  Modal, Steps, Typography, Alert, Spin, List,
} from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  EditOutlined,
  UndoOutlined,
  DownloadOutlined,
  UploadOutlined,
  HistoryOutlined,
  DesktopOutlined,
} from '@ant-design/icons';
import {
  validationAction,
  getEntityValidation,
  getSlicerPackageUrl,
  uploadMask,
  getMaskVersions,
  getMaskFileUrl,
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

  // Состояние модалки редактирования
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);

  // Slicer Agent
  const [slicerStatus, setSlicerStatus] = useState(null); // null | 'checking' | 'available' | 'unavailable'
  const [slicerOpening, setSlicerOpening] = useState(false);
  const [slicerResult, setSlicerResult] = useState(null);

  // История версий
  const [versionsModalOpen, setVersionsModalOpen] = useState(false);
  const [versions, setVersions] = useState([]);
  const [versionsLoading, setVersionsLoading] = useState(false);
  const [activeVersion, setActiveVersion] = useState(null);

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

  const handleDownloadPackage = () => {
    if (!runId) {
      message.warning('Не удалось определить ID запуска для скачивания пакета');
      return;
    }
    const url = getSlicerPackageUrl(runId);
    window.open(url, '_blank');
  };

  const fileInputRef = useRef(null);

  const handleFileSelected = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Валидация
    if (!file.name.endsWith('.nii.gz') && !file.name.endsWith('.nii')) {
      message.error('Файл должен быть в формате NIfTI (.nii.gz или .nii)');
      e.target.value = '';
      return;
    }

    // КРИТИЧНО: читаем файл в память ДО любого setState/ре-рендера.
    // После ре-рендера браузер может отозвать доступ к File объекту
    // (ERR_ACCESS_DENIED в Chrome, NS_BINDING_ABORTED в Firefox).
    const fileName = file.name;
    const fileSize = file.size;

    let blob;
    try {
      const buffer = await file.arrayBuffer();
      blob = new Blob([buffer], { type: 'application/gzip' });
    } catch (readErr) {
      console.error('[UPLOAD] Failed to read file:', readErr);
      message.error('Не удалось прочитать файл');
      e.target.value = '';
      return;
    }

    // Сбрасываем input
    e.target.value = '';

    // Теперь можно безопасно обновлять state — файл уже в памяти
    setUploading(true);
    setUploadResult(null);

    try {
      // Создаём новый File из Blob с оригинальным именем
      const safeFile = new File([blob], fileName, { type: 'application/gzip' });
      const result = await uploadMask(entityId, datasetId, runId, safeFile);
      setUploadResult(result);
      message.success(result.message || 'Маска загружена');

      if (onMaskUploaded) {
        onMaskUploaded(result);
      }
    } catch (err) {
      console.error('Upload error:', err);
      const detail = err.response?.data?.detail || 'Не удалось загрузить маску';
      message.error(detail);
    } finally {
      setUploading(false);
    }
  };

  const openEditModal = () => {
    setUploadResult(null);
    setSlicerResult(null);
    setEditModalOpen(true);

    // Проверяем доступность Slicer Agent
    setSlicerStatus('checking');
    checkSlicerAgent()
      .then((data) => {
        setSlicerStatus(data.slicer_found ? 'available' : 'unavailable');
      })
      .catch(() => {
        setSlicerStatus('unavailable');
      });
  };

  const handleOpenInSlicer = async () => {
    if (!runId) {
      message.warning('Не удалось определить ID запуска');
      return;
    }
    setSlicerOpening(true);
    setSlicerResult(null);
    try {
      const result = await openInSlicer(runId);
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
          onClick={openEditModal}
          disabled={!runId}
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

      {/* Модалка редактирования — рендерится на document.body,
          чтобы не конфликтовать с модалкой NIfTIViewer */}
      <Modal
        title="Редактирование сегментации в 3D Slicer"
        open={editModalOpen}
        onCancel={() => {
          setEditModalOpen(false);
          stopSlicerPoll();
        }}
        getContainer={document.body}
        footer={[
          <Button key="close" onClick={() => { setEditModalOpen(false); stopSlicerPoll(); }}>
            Закрыть
          </Button>,
        ]}
        width={600}
      >
        <Steps
          direction="vertical"
          size="small"
          current={uploadResult ? 3 : (slicerResult?.success ? 2 : 0)}
          items={[
            {
              title: 'Откройте данные в 3D Slicer',
              description: (
                <Space direction="vertical" size="small" style={{ marginTop: 8 }}>
                  {slicerStatus === 'checking' && (
                    <Text type="secondary">
                      <Spin size="small" /> Проверка Slicer Agent...
                    </Text>
                  )}
                  {slicerStatus === 'available' && (
                    <>
                      <Button
                        type="primary"
                        icon={<DesktopOutlined />}
                        onClick={handleOpenInSlicer}
                        loading={slicerOpening}
                      >
                        Открыть в 3D Slicer
                      </Button>
                      {slicerResult?.success && (
                        <Alert type="success" message="Slicer запущен, данные загружены" showIcon />
                      )}
                      {slicerResult && !slicerResult.success && (
                        <Alert type="error" message={slicerResult.message} showIcon />
                      )}
                    </>
                  )}
                  {slicerStatus === 'unavailable' && (
                    <>
                      <Alert
                        type="warning"
                        message="Slicer Agent не запущен"
                        description="Запустите slicer_agent.py на этом компьютере или скачайте пакет вручную."
                        showIcon
                      />
                      <Button
                        icon={<DownloadOutlined />}
                        onClick={handleDownloadPackage}
                      >
                        Скачать пакет вручную
                      </Button>
                    </>
                  )}
                </Space>
              ),
            },
            {
              title: 'Отредактируйте маску в Segment Editor',
              description: (
                <Text type="secondary">
                  Используйте инструменты Segment Editor для правки сегментации,
                  затем сохраните маску: File → Save → выберите формат NIfTI (.nii.gz).
                </Text>
              ),
            },
            {
              title: 'Загрузите отредактированную маску',
              description: (
                <Space direction="vertical" size="small" style={{ marginTop: 8 }}>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".nii.gz,.nii"
                    style={{ display: 'none' }}
                    onChange={handleFileSelected}
                  />
                  <Button
                    icon={<UploadOutlined />}
                    loading={uploading}
                    type={uploadResult ? 'default' : 'primary'}
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                  >
                    {uploading ? 'Загрузка...' : 'Выбрать файл маски (.nii.gz)'}
                  </Button>

                  {uploadResult && (
                    <Alert
                      type={uploadResult.kappa_uploaded ? 'success' : 'warning'}
                      message={uploadResult.message}
                      description={
                        <>
                          <Text>Версия: {uploadResult.mask_version?.version}</Text>
                          <br />
                          <Text>Файл: {uploadResult.file_name}</Text>
                        </>
                      }
                      showIcon
                    />
                  )}
                </Space>
              ),
            },
          ]}
        />
      </Modal>

      {/* Модалка истории версий */}
      <Modal
        title="История версий маски"
        open={versionsModalOpen}
        onCancel={() => setVersionsModalOpen(false)}
        getContainer={document.body}
        footer={null}
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