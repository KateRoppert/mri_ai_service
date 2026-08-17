/**
 * Модальное окно с агрегированным отчётом о пациентах, потерянных на
 * любом этапе пайплайна (не только на этапе 1, как IncompletePatients) —
 * для расследования постфактум завершённого/упавшего запуска.
 */
import { useState, useEffect } from 'react';
import { Modal, Table, Tag, Spin, Alert } from 'antd';
import { getPipelineLosses } from '../services/api';

const STAGE_LABELS = {
  '01_reorganize': 'Этап 1: Анонимизация и стандартизация',
  '03_convert': 'Этап 2: Конвертация в NIfTI',
  '04_quality': 'Этап 3: Оценка качества',
  '05_preprocessing': 'Этап 4: Предобработка',
  '06_segmentation': 'Этап 5: Сегментация',
};

const PipelineLosses = ({ runId, visible, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [losses, setLosses] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (visible && runId) {
      fetchLosses();
    }
  }, [visible, runId]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchLosses = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getPipelineLosses(runId);
      setLosses(data.losses || []);
    } catch (err) {
      console.error('Ошибка загрузки отчёта о потерянных пациентах:', err);
      setError('Не удалось загрузить отчёт');
    } finally {
      setLoading(false);
    }
  };

  const columns = [
    {
      title: 'Этап',
      dataIndex: 'stage',
      key: 'stage',
      render: (stage) => <Tag color="orange">{STAGE_LABELS[stage] || stage}</Tag>,
    },
    { title: 'Пациент', dataIndex: 'patient_id', key: 'patient_id' },
    { title: 'Сессия', dataIndex: 'session_id', key: 'session_id' },
    { title: 'Причина', dataIndex: 'reason', key: 'reason' },
  ];

  return (
    <Modal
      title="Потерянные пациенты по этапам"
      open={visible}
      onCancel={onClose}
      width={800}
      footer={null}
    >
      {error && <Alert type="error" description={error} showIcon style={{ marginBottom: 16 }} />}
      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
        </div>
      ) : (
        <Table
          columns={columns}
          dataSource={losses}
          rowKey={(r, idx) => `${r.stage}_${r.patient_id}_${r.session_id}_${idx}`}
          pagination={{ pageSize: 10 }}
          locale={{ emptyText: 'Потерянных пациентов не найдено' }}
        />
      )}
    </Modal>
  );
};

export default PipelineLosses;
