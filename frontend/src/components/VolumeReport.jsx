/**
 * Модальное окно с отчётами об объёмах опухоли
 */
import { Modal, Collapse, Space, Alert, Spin, Tag } from 'antd';
import {
  PieChartOutlined,
  FileTextOutlined,
} from '@ant-design/icons';
import { useState, useEffect } from 'react';
import { getVolumeReports } from '../services/api';

const VolumeReport = ({ runId, visible, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [reports, setReports] = useState([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState(null);

  /**
   * Загружаем отчёты при открытии модального окна
   */
  useEffect(() => {
    if (visible && runId) {
      fetchReports();
    }
  }, [visible, runId]);

  /**
   * Получить отчёты об объёмах через API
   */
  const fetchReports = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await getVolumeReports(runId);
      setReports(data.reports || []);
      setTotal(data.total || 0);
    } catch (err) {
      console.error('Ошибка загрузки отчётов об объёмах:', err);
      setError('Не удалось загрузить отчёты об объёмах опухоли');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Элементы Collapse — один отчёт на каждую маску
   */
  const collapseItems = reports.map((report, index) => ({
    key: index.toString(),
    label: (
      <Space>
        <strong>{report.patient_id}</strong>
        <Tag>{report.session_id}</Tag>
        <span style={{ color: '#999', fontSize: 12 }}>{report.mask_file}</span>
      </Space>
    ),
    children: (
      <pre style={{
        background: '#f5f5f5',
        padding: 16,
        borderRadius: 4,
        fontSize: 13,
        lineHeight: 1.6,
        whiteSpace: 'pre-wrap',
        margin: 0,
        fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
      }}>
        {report.report_text}
      </pre>
    ),
  }));

  return (
    <Modal
      title={
        <Space>
          <PieChartOutlined />
          <span>Отчёты об объёмах опухоли ({total})</span>
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
            description={`Рассчитаны объёмы для ${total} масок сегментации. Раскройте каждый отчёт для просмотра деталей.`}
          />

          <Collapse
            items={collapseItems}
            defaultActiveKey={['0']}
          />
        </>
      )}
    </Modal>
  );
};

export default VolumeReport;