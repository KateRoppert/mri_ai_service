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
export const startPipeline = async (inputPath, outputPath, useDefaultOutput = false) => {
  const response = await apiClient.post('/pipeline/start', {
    input_path: inputPath,
    output_path: outputPath,
    use_default_output: useDefaultOutput,
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

export default {
  startPipeline,
  getPipelineStatus,
  getPipelineHistory,
  getQualityReport,
  getVolumeReports,
  getNIfTIFiles,     
  getNIfTIFileUrl,    
};

