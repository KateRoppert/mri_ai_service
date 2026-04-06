/**
 * Модальное окно с отчётами об объёмах опухоли
 */
import { Modal, Collapse, Space, Alert, Spin, Tag, Table, Statistic, Row, Col, Card } from 'antd';
import {
  PieChartOutlined,
} from '@ant-design/icons';
import { useState, useEffect } from 'react';
import { getVolumeReports } from '../services/api';

const VolumeReport = ({ runId, visible, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [reports, setReports] = useState([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (visible && runId) {
      fetchReports();
    }
  }, [visible, runId]);

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

  const renderJsonReport = (reportData) => {
    const classColumns = [
      { title: 'Класс', dataIndex: 'name', key: 'name' },
      { title: 'Воксели', dataIndex: 'voxel_count', key: 'voxels', align: 'right',
        render: (v) => v?.toLocaleString() },
      { title: 'мм³', dataIndex: 'volume_mm3', key: 'mm3', align: 'right',
        render: (v) => v?.toFixed(2) },
      { title: 'см³', dataIndex: 'volume_cm3', key: 'cm3', align: 'right',
        render: (v) => v?.toFixed(4) },
    ];

    const classData = Object.entries(reportData.classes || {}).map(([label, cls]) => ({
      key: label,
      name: cls.name,
      voxel_count: cls.voxel_count,
      volume_mm3: cls.volume_mm3,
      volume_cm3: cls.volume_cm3,
    }));

    const clinicalData = reportData.clinical
      ? Object.entries(reportData.clinical).map(([key, cls]) => ({
          key,
          name: cls.name_ru || cls.name,
          voxel_count: cls.voxel_count,
          volume_mm3: cls.volume_mm3,
          volume_cm3: cls.volume_cm3,
        }))
      : [];

    const totalTumor = reportData.total_tumor || {};

    return (
      <div>
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="Общий объём опухоли"
                value={totalTumor.volume_cm3}
                suffix="см³"
                precision={4}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="Всего вокселей"
                value={totalTumor.voxel_count}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="Размер вокселя"
                value={reportData.voxel_size_mm?.map(v => v.toFixed(3)).join(' × ')}
                suffix="мм"
              />
            </Card>
          </Col>
        </Row>

        <Table
          columns={classColumns}
          dataSource={classData}
          pagination={false}
          size="small"
          title={() => <strong>Объёмы по классам</strong>}
          style={{ marginBottom: 16 }}
        />

        {clinicalData.length > 0 && (
          <Table
            columns={classColumns}
            dataSource={clinicalData}
            pagination={false}
            size="small"
            title={() => <strong>Клиническая сводка (RANO)</strong>}
          />
        )}
      </div>
    );
  };

  const collapseItems = reports.map((report, index) => ({
    key: index.toString(),
    label: (
      <Space>
        <strong>{report.patient_id}</strong>
        <Tag>{report.session_id}</Tag>
        <span style={{ color: '#999', fontSize: 12 }}>{report.mask_file}</span>
      </Space>
    ),
    children: report.report_data
      ? renderJsonReport(report.report_data)
      : (
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
            description={`Рассчитаны объёмы для ${total} масок сегментации.`}
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