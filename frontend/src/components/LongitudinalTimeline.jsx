// frontend/src/components/LongitudinalTimeline.jsx
import { useEffect, useState } from 'react';
import { Table, Spin, Alert, Tag } from 'antd';
import { getLongitudinalReport } from '../services/api';

const LongitudinalTimeline = ({ patientId, lesionType }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!patientId) return;
    setLoading(true);
    getLongitudinalReport(patientId, lesionType)
      .then(resp => setData(resp.points))
      .catch(err => {
        if (err.response?.status === 404) setData([]);
        else setError('Не удалось загрузить динамику');
      })
      .finally(() => setLoading(false));
  }, [patientId, lesionType]);

  if (loading) return <Spin size="small" />;
  if (error) return <Alert message={error} type="warning" showIcon />;
  if (!data || data.length < 2) return null;

  const columns = [
    { title: 'Дата', dataIndex: 'scan_date', key: 'date', width: 130 },
    { title: 'Сессия', dataIndex: 'session_id', key: 'session', width: 130 },
    {
      title: 'Объём (см³)',
      dataIndex: 'total_volume_cm3',
      key: 'volume',
      align: 'right',
      render: val => val.toFixed(3),
    },
    {
      title: 'Очагов',
      dataIndex: 'lesion_count',
      key: 'count',
      align: 'right',
      render: val => val ?? '—',
    },
    {
      title: 'Δ объём',
      key: 'delta',
      align: 'right',
      render: (_, record, idx) => {
        if (idx === 0) return '—';
        const prev = data[idx - 1].total_volume_cm3;
        const delta = record.total_volume_cm3 - prev;
        const color = delta > 0 ? 'red' : delta < 0 ? 'green' : 'default';
        return <Tag color={color}>{delta > 0 ? '+' : ''}{delta.toFixed(3)}</Tag>;
      },
    },
  ];

  return (
    <Table
      columns={columns}
      dataSource={data.map((p, i) => ({ ...p, key: i }))}
      pagination={false}
      size="small"
      bordered
    />
  );
};

export default LongitudinalTimeline;
