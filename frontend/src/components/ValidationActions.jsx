/**
 * Панель действий валидации: Подтвердить / Отклонить / Редактировать
 * Показывает текущее состояние и историю голосов.
 * Кнопка «Редактировать» открывает модалку:
 *   - Скачать Slicer-пакет → редактировать в 3D Slicer → загрузить маску обратно.
 */
import { useState, useEffect, useRef } from 'react';
import {
  Button, Space, Tag, Tooltip, message, Popconfirm,
  Modal, Upload, Steps, Typography, Alert, Spin, List,
} from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  EditOutlined,
  UndoOutlined,
  DownloadOutlined,
  UploadOutlined,
  HistoryOutlined,
} from '@ant-design/icons';
import {
  validationAction,
  getEntityValidation,
  getSlicerPackageUrl,
  uploadMask,
  getMaskVersions,
} from '../services/api';

const { Text } = Typography;

const STATUS_TAG = {
  3: { text: 'Размечено', color: 'blue' },
  4: { text: 'На проверке', color: 'orange' },
  5: { text: 'Верифицировано', color: 'green' },
};

const ValidationActions = ({ entityId, datasetId, runId, onStatusChange, onMaskUploaded }) => {
  const [loading, setLoading] = useState(false);
  const [votes, setVotes] = useState(null);
  const [myVote, setMyVote] = useState(null);

  // Состояние модалки редактирования
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);

  // История версий
  const [versionsModalOpen, setVersionsModalOpen] = useState(false);
  const [versions, setVersions] = useState([]);
  const [versionsLoading, setVersionsLoading] = useState(false);

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

  const handleUploadMask = async (options) => {
    const { file, onSuccess, onError } = options;

    // Валидация на клиенте
    if (!file.name.endsWith('.nii.gz') && !file.name.endsWith('.nii')) {
      message.error('Файл должен быть в формате NIfTI (.nii.gz или .nii)');
      onError(new Error('Неверный формат'));
      return;
    }

    setUploading(true);
    setUploadResult(null);

    try {
      const result = await uploadMask(entityId, datasetId, runId, file);
      setUploadResult(result);
      onSuccess(result);
      message.success(result.message || 'Маска загружена');

      if (onMaskUploaded) {
        onMaskUploaded(result);
      }
    } catch (err) {
      console.error('Ошибка загрузки маски:', err);
      const detail = err.response?.data?.detail || 'Не удалось загрузить маску';
      message.error(detail);
      onError(err);
    } finally {
      setUploading(false);
    }
  };

  const openEditModal = () => {
    setUploadResult(null);
    setEditModalOpen(true);
  };

  // === История версий ===

  const openVersionsModal = async () => {
    setVersionsModalOpen(true);
    setVersionsLoading(true);
    try {
      const data = await getMaskVersions(entityId);
      setVersions(data.versions || []);
    } catch (err) {
      console.error('Ошибка загрузки версий:', err);
      message.error('Не удалось загрузить историю версий');
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

      {/* Модалка редактирования */}
      <Modal
        title="Редактирование сегментации в 3D Slicer"
        open={editModalOpen}
        onCancel={() => setEditModalOpen(false)}
        footer={[
          <Button key="close" onClick={() => setEditModalOpen(false)}>
            Закрыть
          </Button>,
        ]}
        width={600}
      >
        <Steps
          direction="vertical"
          size="small"
          current={uploadResult ? 2 : 0}
          items={[
            {
              title: 'Скачайте пакет для 3D Slicer',
              description: (
                <Space direction="vertical" size="small" style={{ marginTop: 8 }}>
                  <Text type="secondary">
                    Архив содержит preprocessed-изображения, нативные изображения и маску сегментации
                    с инструкцией по редактированию.
                  </Text>
                  <Button
                    type="primary"
                    icon={<DownloadOutlined />}
                    onClick={handleDownloadPackage}
                  >
                    Скачать пакет
                  </Button>
                </Space>
              ),
            },
            {
              title: 'Отредактируйте маску в 3D Slicer',
              description: (
                <Text type="secondary">
                  Откройте архив в 3D Slicer, используйте Segment Editor для правки,
                  затем экспортируйте маску в формате NIfTI (.nii.gz).
                </Text>
              ),
            },
            {
              title: 'Загрузите отредактированную маску',
              description: (
                <Space direction="vertical" size="small" style={{ marginTop: 8 }}>
                  <Upload
                    customRequest={handleUploadMask}
                    accept=".nii.gz,.nii"
                    maxCount={1}
                    showUploadList={false}
                    disabled={uploading}
                  >
                    <Button
                      icon={<UploadOutlined />}
                      loading={uploading}
                      type={uploadResult ? 'default' : 'primary'}
                    >
                      {uploading ? 'Загрузка...' : 'Выбрать файл маски (.nii.gz)'}
                    </Button>
                  </Upload>

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
          <List
            dataSource={versions}
            renderItem={(v) => (
              <List.Item>
                <List.Item.Meta
                  title={
                    <Space>
                      <Tag color={v.source === 'ai' ? 'blue' : 'green'}>
                        {v.source === 'ai' ? 'ИИ' : 'Эксперт'}
                      </Tag>
                      <Text strong>Версия {v.version}</Text>
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
        )}
      </Modal>
    </>
  );
};

export default ValidationActions;