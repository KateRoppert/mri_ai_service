/**
 * Панель валидации — список сессий, доступных для проверки врачом-экспертом
 */
import { useState, useEffect } from 'react';
import { Card, Table, Tag, Button, Alert, Spin, Space, Typography, Select } from 'antd';
import { CheckCircleOutlined, EyeOutlined, ReloadOutlined } from '@ant-design/icons';
import {
  getValidationEntities,
  getLesionTypes,
  getValidationFileUrl,
} from '../services/api';
import NIfTIViewer from './NIfTIViewer';

const { Text } = Typography;

const STATUS_COLORS = {
  New: 'default',
  Labeled: 'blue',
  'Under Verification': 'orange',
  Verified: 'green',
  'Self-Verified': 'cyan',
};

/**
 * Извлечь модальность из имени файла preprocessed.
 * sub-001_ses-001_t1.nii.gz → t1
 * sub-001_ses-001_t1c.nii.gz → t1c
 */
const extractModality = (fileName) => {
  const base = fileName.replace('.nii.gz', '');
  const parts = base.split('_');
  return parts[parts.length - 1];
};

/**
 * Построить customFiles для NIfTIViewer из сущности Каппы
 */
const buildCustomFiles = (entity, datasetId) => {
  const info = entity.dsEntityInfo || {};
  const dataFileNames = info.data_files || [];
  const predictionFileNames = info.prediction_files || [];
  const allFiles = entity.files || [];

  // Маппинг fileName → fileId
  const fileIdByName = {};
  allFiles.forEach((f) => {
    fileIdByName[f.fileName] = f.fileId;
  });

  // Берём первую маску (у нас одна на сессию)
  const maskFileName = predictionFileNames[0];
  const maskFileId = fileIdByName[maskFileName];
  const maskUrl = maskFileId ? getValidationFileUrl(datasetId, maskFileId) : null;

  // Labeled mask (MS only) and per-lesion volume map for hover tooltip
  const labelsFileName = info.lesion_labels_file;
  const labelsFileId = labelsFileName ? fileIdByName[labelsFileName] : null;
  const maskLabelsUrl = labelsFileId ? getValidationFileUrl(datasetId, labelsFileId) : null;
  const volumesByLabel = info.lesion_stats?.lesion_volumes_by_label || null;

  // bids_id в Каппе включает сессию ("sub-001_ses-002"). Разделяем на
  // пациента и сессию, чтобы метка визуализатора была согласована с
  // секциями Запуск/История (пациент / сессия / модальность).
  const bids = info.bids_id || entity.dsEntityName || '';
  let patientId = bids;
  let sessionId = '';
  const sesIdx = bids.indexOf('_ses-');
  if (sesIdx !== -1) {
    patientId = bids.slice(0, sesIdx);
    sessionId = bids.slice(sesIdx + 1);
  }

  // Для каждой модальности — отдельный объект файла
  return dataFileNames
    .map((fileName) => {
      const fileId = fileIdByName[fileName];
      if (!fileId || !maskUrl) return null;
      const modality = extractModality(fileName);
      return {
        filename: fileName,
        mask_filename: maskFileName,
        patient_id: patientId,
        session_id: sessionId,
        modality: modality.toUpperCase(),
        image_url: getValidationFileUrl(datasetId, fileId),
        mask_url: maskUrl,
        mask_labels_url: maskLabelsUrl,
        lesion_volumes_by_label: volumesByLabel,
      };
    })
    .filter(Boolean);
};

// Kappa returns entities unordered. Sort by the bids_id ("sub-002_ses-001") so a
// patient's sessions sit together and in chronological session order. numeric:true
// keeps sub-2 before sub-10 even if the ids are not zero-padded.
const sortEntitiesByPatientSession = (entities) =>
  [...(entities || [])].sort((a, b) => {
    const ka = a.dsEntityInfo?.bids_id || a.dsEntityName || '';
    const kb = b.dsEntityInfo?.bids_id || b.dsEntityName || '';
    return ka.localeCompare(kb, undefined, { numeric: true });
  });

const ValidationPanel = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [entities, setEntities] = useState([]);
  const [lesionTypes, setLesionTypes] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState(null);

  // Состояние visualizer
  const [viewerOpen, setViewerOpen] = useState(false);
  const [viewerFiles, setViewerFiles] = useState([]);
  const [viewerEntityRef, setViewerEntityRef] = useState(null);
  const [viewerLesionType, setViewerLesionType] = useState('glioblastoma');
  const [viewerEntityInfo, setViewerEntityInfo] = useState(null);

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
      setEntities(sortEntitiesByPatientSession(data.entities || []));
    } catch (err) {
      console.error('Ошибка загрузки сессий:', err);
      setError('Не удалось загрузить список сессий');
    } finally {
      setLoading(false);
    }
  };

  const handleView = (entity) => {
    const customFiles = buildCustomFiles(entity, selectedDatasetId);
    if (customFiles.length === 0) {
      setError('Не удалось подготовить файлы сессии для просмотра');
      return;
    }
    setViewerFiles(customFiles);
    setViewerEntityRef({
      entity_id: entity.dsEntityId,
      dataset_id: selectedDatasetId,
    });
    setViewerLesionType(entity.dsEntityInfo?.lesion_type || 'glioblastoma');
    setViewerEntityInfo(entity.dsEntityInfo || null);
    setViewerOpen(true);
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
      render: (date) => (date ? new Date(date).toLocaleString('ru-RU') : '—'),
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
            onClick={() => handleView(entity)}
          >
            Просмотреть
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <>
      <Card
        title={
          <Space>
            <CheckCircleOutlined />
            <span>Сессии для валидации</span>
          </Space>
        }
        extra={
          <Space>
            <Select
              value={selectedDatasetId}
              onChange={setSelectedDatasetId}
              placeholder="Тип поражения"
              style={{ minWidth: 220 }}
              options={lesionTypes
                .filter((t) => t.dataset_id)
                .map((t) => ({
                  value: t.dataset_id,
                  label: `${t.name} (dataset ${t.dataset_id})`,
                }))}
            />
            <Button
              icon={<ReloadOutlined />}
              onClick={() => selectedDatasetId && loadEntities(selectedDatasetId)}
              loading={loading}
            >
              Обновить
            </Button>
          </Space>
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

      <NIfTIViewer
        visible={viewerOpen}
        onClose={() => setViewerOpen(false)}
        customFiles={viewerFiles}
        validationRef={viewerEntityRef}
        lesionType={viewerLesionType}
        kappaReport={viewerEntityInfo}
      />
    </>
  );
};

export default ValidationPanel;