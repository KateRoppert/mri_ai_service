/**
 * Модальное окно с детальным отчётом о качестве изображения
 */
import { Modal, Descriptions, Tag, Space, Alert, Spin } from 'antd';
import { 
  CheckCircleOutlined, 
  WarningOutlined, 
  CloseCircleOutlined,
  FileTextOutlined 
} from '@ant-design/icons';
import { useState, useEffect } from 'react';
import { getQualityReport } from '../services/api';

const QualityReport = ({ runId, visible, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  /**
   * Загружаем отчёт при открытии модального окна
   */
  useEffect(() => {
    if (visible && runId) {
      fetchReport();
    }
  }, [visible, runId]);

  /**
   * Получить отчёт о качестве через API
   */
  const fetchReport = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await getQualityReport(runId);
      setReport(data);
    } catch (err) {
      console.error('Ошибка загрузки отчёта:', err);
      setError('Не удалось загрузить отчёт о качестве');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Определяем конфигурацию для категории качества
   */
  const getQualityConfig = (category) => {
    switch (category?.toUpperCase()) {
      case 'GOOD':
        return {
          color: 'success',
          icon: <CheckCircleOutlined />,
          text: 'Хорошее',
        };
      case 'ACCEPTABLE':
      case 'FAIR':
        return {
          color: 'warning',
          icon: <WarningOutlined />,
          text: 'Приемлемое',
        };
      case 'POOR':
      case 'BAD':
        return {
          color: 'error',
          icon: <CloseCircleOutlined />,
          text: 'Плохое',
        };
      default:
        return {
          color: 'default',
          icon: <FileTextOutlined />,
          text: 'Неизвестно',
        };
    }
  };

  /**
   * Форматируем число для отображения
   */
  const formatMetric = (value) => {
    if (typeof value === 'number') {
      return value.toFixed(3);
    }
    return value;
  };

  if (!report && !loading && !error) {
    return null;
  }

  const qualityConfig = report ? getQualityConfig(report.quality_category) : null;

  return (
    <Modal
      title={
        <Space>
          <FileTextOutlined />
          <span>Отчёт о качестве изображения</span>
        </Space>
      }
      open={visible}
      onCancel={onClose}
      width={800}
      footer={null}
    >
      {loading && (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
          <p style={{ marginTop: 16 }}>Загрузка отчёта...</p>
        </div>
      )}

      {error && (
        <Alert
          title="Ошибка"
          description={error}
          type="error"
          showIcon
        />
      )}

      {report && !loading && (
        <>
          {/* Общая оценка качества */}
          <div style={{ marginBottom: 24, textAlign: 'center' }}>
            <Space direction="vertical" size="small">
              <Tag 
                color={qualityConfig.color} 
                icon={qualityConfig.icon}
                style={{ fontSize: 16, padding: '8px 16px' }}
              >
                {qualityConfig.text}
              </Tag>
              <div style={{ fontSize: 24, fontWeight: 'bold' }}>
                {report.quality_score.toFixed(2)} / 100
              </div>
            </Space>
          </div>

          {/* Информация о файле */}
          <Descriptions 
            title="Информация о данных" 
            bordered 
            column={2}
            size="small"
            style={{ marginBottom: 24 }}
          >
            <Descriptions.Item label="Файл" span={2}>
              {report.file}
            </Descriptions.Item>
            <Descriptions.Item label="ID пациента">
              {report.patient_id}
            </Descriptions.Item>
            <Descriptions.Item label="Модальность">
              {report.modality.toUpperCase()}
            </Descriptions.Item>
          </Descriptions>

          {/* Детальные метрики */}
          <Descriptions 
            title="Детальные метрики качества" 
            bordered 
            column={1}
            size="small"
          >
            <Descriptions.Item label="SNR (Signal-to-Noise Ratio)">
              <strong>{formatMetric(report.metrics.snr)}</strong>
              <span style={{ marginLeft: 8, color: '#999' }}>
                Отношение сигнал/шум
              </span>
            </Descriptions.Item>

            <Descriptions.Item label="CNR (Contrast-to-Noise Ratio)">
              <strong>{formatMetric(report.metrics.cnr)}</strong>
              <span style={{ marginLeft: 8, color: '#999' }}>
                Контраст к шуму
              </span>
            </Descriptions.Item>

            <Descriptions.Item label="EFC (Entropy Focus Criterion)">
              <strong>{formatMetric(report.metrics.efc)}</strong>
              <span style={{ marginLeft: 8, color: '#999' }}>
                Критерий энтропийной фокусировки
              </span>
            </Descriptions.Item>

            <Descriptions.Item label="FBER (Foreground-Background Energy Ratio)">
              <strong>{formatMetric(report.metrics.fber)}</strong>
              <span style={{ marginLeft: 8, color: '#999' }}>
                Отношение энергии переднего и заднего фона
              </span>
            </Descriptions.Item>

            <Descriptions.Item label="Gradient Sharpness">
              <strong>{formatMetric(report.metrics.gradient_sharpness)}</strong>
              <span style={{ marginLeft: 8, color: '#999' }}>
                Резкость градиента
              </span>
            </Descriptions.Item>

            <Descriptions.Item label="Voxel Anisotropy">
              <strong>{formatMetric(report.metrics.voxel_anisotropy)}</strong>
              <span style={{ marginLeft: 8, color: '#999' }}>
                Анизотропия вокселя
              </span>
            </Descriptions.Item>

            <Descriptions.Item label="Intensity Variance">
              <strong>{formatMetric(report.metrics.intensity_variance)}</strong>
              <span style={{ marginLeft: 8, color: '#999' }}>
                Дисперсия интенсивности
              </span>
            </Descriptions.Item>

            <Descriptions.Item label="Coefficient of Variation">
              <strong>{formatMetric(report.metrics.coefficient_of_variation)}</strong>
              <span style={{ marginLeft: 8, color: '#999' }}>
                Коэффициент вариации
              </span>
            </Descriptions.Item>
          </Descriptions>

          {/* Пояснение */}
          <Alert
            type="info"
            showIcon
            style={{ marginTop: 16 }}
            description="Эти метрики характеризуют техническое качество МРТ изображения. Более высокое качество обеспечивает более точную сегментацию поражений."
          />
        </>
      )}
    </Modal>
  );
};

export default QualityReport;