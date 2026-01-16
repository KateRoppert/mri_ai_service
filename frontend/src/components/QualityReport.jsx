/**
 * Модальное окно с детальными отчётами о качестве всех обработанных файлов
 */
import { Modal, Collapse, Tag, Space, Alert, Spin, Statistic, Row, Col } from 'antd';
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
  const [reports, setReports] = useState([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState(null);

  /**
   * Загружаем отчёт при открытии модального окна
   */
  useEffect(() => {
    if (visible && runId) {
      fetchReports();
    }
  }, [visible, runId]);

  /**
   * Получить отчёты о качестве через API
   */
  const fetchReports = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await getQualityReport(runId);
      setReports(data.reports || []);
      setTotal(data.total || 0);
    } catch (err) {
      console.error('Ошибка загрузки отчётов:', err);
      setError('Не удалось загрузить отчёты о качестве');
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

  /**
   * Создаём элементы Collapse для каждого отчёта
   */
  const collapseItems = reports.map((report, index) => {
    const qualityConfig = getQualityConfig(report.quality_category);
    
    return {
      key: index.toString(),
      label: (
        <Space>
          <strong>{report.file}</strong>
          <Tag color={qualityConfig.color} icon={qualityConfig.icon}>
            {qualityConfig.text} ({report.quality_score.toFixed(1)}/100)
          </Tag>
        </Space>
      ),
      children: (
        <div>
          {/* Основная информация */}
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={8}>
              <Statistic 
                title="Общая оценка" 
                value={report.quality_score.toFixed(2)} 
                suffix="/ 100"
              />
            </Col>
            <Col span={8}>
              <Statistic title="ID пациента" value={report.patient_id} />
            </Col>
            <Col span={8}>
              <Statistic title="Модальность" value={report.modality.toUpperCase()} />
            </Col>
          </Row>

          {/* Метрики в компактном виде */}
          <div style={{ background: '#f5f5f5', padding: 12, borderRadius: 4 }}>
            <Row gutter={[16, 8]}>
              <Col span={12}>
                <strong>SNR:</strong> {formatMetric(report.metrics.snr)}
                <span style={{ marginLeft: 4, color: '#999', fontSize: 12 }}>
                  (Сигнал/шум)
                </span>
              </Col>
              <Col span={12}>
                <strong>CNR:</strong> {formatMetric(report.metrics.cnr)}
                <span style={{ marginLeft: 4, color: '#999', fontSize: 12 }}>
                  (Контраст/шум)
                </span>
              </Col>
              <Col span={12}>
                <strong>EFC:</strong> {formatMetric(report.metrics.efc)}
                <span style={{ marginLeft: 4, color: '#999', fontSize: 12 }}>
                  (Фокусировка)
                </span>
              </Col>
              <Col span={12}>
                <strong>FBER:</strong> {formatMetric(report.metrics.fber)}
                <span style={{ marginLeft: 4, color: '#999', fontSize: 12 }}>
                  (Передний/задний фон)
                </span>
              </Col>
              <Col span={12}>
                <strong>Резкость:</strong> {formatMetric(report.metrics.gradient_sharpness)}
              </Col>
              <Col span={12}>
                <strong>Анизотропия:</strong> {formatMetric(report.metrics.voxel_anisotropy)}
              </Col>
              <Col span={12}>
                <strong>Дисперсия:</strong> {formatMetric(report.metrics.intensity_variance)}
              </Col>
              <Col span={12}>
                <strong>Коэф. вариации:</strong> {formatMetric(report.metrics.coefficient_of_variation)}
              </Col>
            </Row>
          </div>
        </div>
      ),
    };
  });

  return (
    <Modal
      title={
        <Space>
          <FileTextOutlined />
          <span>Отчёты о качестве изображений ({total})</span>
        </Space>
      }
      open={visible}
      onCancel={onClose}
      width={900}
      footer={null}
    >
      {loading && (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
          <p style={{ marginTop: 16 }}>Загрузка отчётов...</p>
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

      {reports.length > 0 && !loading && (
        <>
          <Alert
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
            description={`Обработано ${total} файлов. Раскройте каждый для просмотра детальных метрик качества.`}
          />
          
          <Collapse 
            items={collapseItems}
            defaultActiveKey={['0']}  // Первый отчёт раскрыт по умолчанию
          />
        </>
      )}
    </Modal>
  );
};

export default QualityReport;