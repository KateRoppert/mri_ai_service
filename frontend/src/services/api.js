/**
 * API клиент для взаимодействия с backend
 */
import axios from 'axios';

// Базовый URL для API (прокси настроен в vite.config.js)
const API_BASE_URL = '/api';

// Создаём axios instance с настройками
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Запуск pipeline
 */
export const startPipeline = async (inputPath, outputPath, useDefaultOutput = false, lesionType = null) => {
  const kappaSessionId = localStorage.getItem('kappa_session_id');
  const response = await apiClient.post('/pipeline/start', {
    input_path: inputPath,
    output_path: outputPath,
    use_default_output: useDefaultOutput,
    kappa_session_id: kappaSessionId,
    lesion_type: lesionType,
  });
  return response.data;
};

/**
 * Получить статус выполнения pipeline
 */
export const getPipelineStatus = async (runId) => {
  const response = await apiClient.get(`/pipeline/status/${runId}`);
  return response.data;
};

/**
 * Получить историю запусков
 */
export const getPipelineHistory = async (limit = 50, offset = 0) => {
  const response = await apiClient.get('/pipeline/history', {
    params: { limit, offset },
  });
  return response.data;
};

/**
 * Получить отчёт о качестве
 */
export const getQualityReport = async (runId) => {
  const response = await apiClient.get(`/quality-report/${runId}`);
  return response.data;
};

/**
 * Получить отчёты об объёмах опухоли
 */
export const getVolumeReports = async (runId) => {
  const response = await apiClient.get(`/volume-reports/${runId}`);
  return response.data;
};

export const getLobarReports = async (runId) => {
  const response = await apiClient.get(`/lobar-reports/${runId}`);
  return response.data;
};

/**
 * Получить список доступных NIfTI файлов для визуализации
 */
export const getNIfTIFiles = async (runId) => {
  const response = await apiClient.get(`/nifti-files/${runId}`);
  return response.data;
};

/**
 * Получить URL для NIfTI файла
 * Эта функция просто возвращает URL, сам файл загружается niivue
 */
export const getNIfTIFileUrl = (url) => {
  // Преобразуем относительный URL в полный
  return `http://localhost:8000${url}`;
};

/**
 * Получить URL лобарного атласа
 */
export const getLobarAtlasUrl = (runId) => {
  return `http://localhost:8000/api/lobar-atlas/${runId}`;
};

/**
 * Получить список типов поражений
 */
export const getLesionTypes = async () => {
  const response = await apiClient.get('/kappa/lesion-types');
  return response.data;
};

/**
 * Получить список сущностей датасета валидации
 */
export const getValidationEntities = async (datasetId) => {
  const sessionId = localStorage.getItem('kappa_session_id');
  const response = await apiClient.get(`/kappa/entities/${datasetId}`, {
    params: { session_id: sessionId },
  });
  return response.data;
};

/**
 * Получить URL файла из хранилища валидации (для NiiVue)
 */
export const getValidationFileUrl = (datasetId, fileId) => {
  const sessionId = localStorage.getItem('kappa_session_id');
  return `http://localhost:8000/api/kappa/file/${datasetId}/${fileId}?session_id=${sessionId}`;
};

/**
 * Отправить действие валидации (confirm/reject/revoke)
 */
export const validationAction = async (datasetId, entityId, action, comment = null) => {
  const sessionId = localStorage.getItem('kappa_session_id');
  const response = await apiClient.post('/validation/action', {
    entity_id: entityId,
    dataset_id: datasetId,
    session_id: sessionId,
    action,
    comment,
  });
  return response.data;
};

/**
 * Получить текущее состояние валидации сущности
 */
export const getEntityValidation = async (entityId) => {
  const sessionId = localStorage.getItem('kappa_session_id');
  const response = await apiClient.get(`/validation/entity/${entityId}`, {
    params: { session_id: sessionId },
  });
  return response.data;
};

/**
 * Получить сущности Каппы для запуска пайплайна
 */
export const getEntitiesForRun = async (runId) => {
  const response = await apiClient.get(`/validation/entities-for-run/${runId}`);
  return response.data;
};

/**
 * Получить run_id по entity_id (для вкладки Валидации)
 */
export const getEntityRunInfo = async (entityId) => {
  const response = await apiClient.get(`/validation/entity-run-info/${entityId}`);
  return response.data;
};

/**
 * Скачать zip-пакет для 3D Slicer
 */
export const getSlicerPackageUrl = (runId) => {
  const sessionId = localStorage.getItem('kappa_session_id');
  return `http://localhost:8000/api/validation/download-package/${runId}?session_id=${sessionId}`;
};

/**
 * Загрузить отредактированную маску эксперта
 */
export const uploadMask = async (entityId, datasetId, runId, file) => {
  const sessionId = localStorage.getItem('kappa_session_id');
  const formData = new FormData();
  formData.append('entity_id', entityId);
  formData.append('dataset_id', String(datasetId));
  formData.append('session_id', sessionId);
  formData.append('run_id', runId);
  formData.append('file', file, file.name);

  // Используем fetch вместо axios — axios некорректно обрабатывает
  // multipart/form-data при наличии default Content-Type: application/json
  const response = await fetch('/api/validation/upload-mask', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Ошибка загрузки' }));
    throw { response: { data: error } };
  }

  return await response.json();
};

/**
 * Получить историю версий масок
 */
export const getMaskVersions = async (entityId) => {
  const sessionId = localStorage.getItem('kappa_session_id');
  const response = await apiClient.get(`/validation/mask-versions/${entityId}`, {
    params: { session_id: sessionId },
  });
  return response.data;
};

export default {
  startPipeline,
  getPipelineStatus,
  getPipelineHistory,
  getQualityReport,
  getVolumeReports,
  getNIfTIFiles,     
  getNIfTIFileUrl, 
  getLesionTypes,   
  getValidationEntities,
  getValidationFileUrl,
  validationAction,
  getEntityValidation,
  getEntitiesForRun,
  getEntityRunInfo,
  getSlicerPackageUrl,
  uploadMask,
  getMaskVersions,
};