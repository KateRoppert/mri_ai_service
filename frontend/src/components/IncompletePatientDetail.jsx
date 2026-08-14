/**
 * Модальное окно с деталями одной сессии, требующей внимания врача:
 * какие модальности есть/не хватает, список исключённых серий с
 * возможностью назначить их на модальность, кнопка отбросить сессию.
 */
import { useState } from 'react';
import { Modal, Tag, Space, List, Select, Button, Popconfirm, message, Divider, Typography } from 'antd';
import { DeleteOutlined } from '@ant-design/icons';
import { relabelSeries, discardSession, mergeSessions } from '../services/api';

const { Text } = Typography;

const MODALITY_OPTIONS = [
  { label: 'T1', value: 't1' },
  { label: 'T1c', value: 't1c' },
  { label: 'T2', value: 't2' },
  { label: 'FLAIR (T2fl)', value: 't2fl' },
];

const REASON_LABELS = {
  unrecognized: 'алгоритм не распознал',
  lost_deduplication: 'алгоритм распознал, но выбрал другую копию',
  replaced_by_manual_relabel: 'заменена вручную ранее',
  from_other_session: 'перенесена из другой сессии пациента',
};

const IncompletePatientDetail = ({ runId, session, sessions = [], visible, onClose, onActionComplete }) => {
  const [selectedModality, setSelectedModality] = useState({});
  const [loadingPath, setLoadingPath] = useState(null);
  const [discarding, setDiscarding] = useState(false);
  const [donorSessionId, setDonorSessionId] = useState(undefined);
  const [merging, setMerging] = useState(false);

  if (!session) return null;

  const isReadOnly = session.status === 'discarded' || session.status === 'merged';

  const handleRelabel = async (excludedEntry) => {
    const modality = selectedModality[excludedEntry.original_path] || excludedEntry.detected_modality;
    if (!modality) {
      message.error('Выберите модальность');
      return;
    }
    setLoadingPath(excludedEntry.original_path);
    try {
      const result = await relabelSeries(
        runId, session.patient_id, session.session_id, excludedEntry.original_path, modality
      );
      message.success(
        result.status === 'complete'
          ? 'Серия назначена, сессия теперь полная'
          : 'Серия назначена'
      );
      onActionComplete();
    } catch (err) {
      console.error('Ошибка переразметки:', err);
      message.error(err.response?.data?.detail || 'Не удалось назначить серию');
    } finally {
      setLoadingPath(null);
    }
  };

  const handleDiscard = async () => {
    setDiscarding(true);
    try {
      await discardSession(runId, session.patient_id, session.session_id);
      message.success('Сессия отброшена');
      onClose();
      onActionComplete();
    } catch (err) {
      console.error('Ошибка:', err);
      message.error('Не удалось отбросить сессию');
    } finally {
      setDiscarding(false);
    }
  };

  const handleMerge = async () => {
    if (!donorSessionId) {
      message.error('Выберите сессию для объединения');
      return;
    }
    setMerging(true);
    try {
      const result = await mergeSessions(runId, session.patient_id, session.session_id, donorSessionId);
      message.success(`Серии из ${donorSessionId} добавлены как альтернативы (${result.pulled_series})`);
      setDonorSessionId(undefined);
      onActionComplete();
    } catch (err) {
      console.error('Ошибка объединения:', err);
      message.error(err.response?.data?.detail || 'Не удалось объединить сессии');
    } finally {
      setMerging(false);
    }
  };

  const otherSessions = sessions.filter(
    (s) => s.patient_id === session.patient_id
      && s.session_id !== session.session_id
      && s.status !== 'merged'
      && s.status !== 'discarded'
  );

  const isAlreadyFilled = (modality) => session.available.includes(modality);

  return (
    <Modal
      title={`${session.original_id} — ${session.session_id}`}
      open={visible}
      onCancel={onClose}
      width={700}
      footer={null}
    >
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        {isReadOnly && (
          <Text type="secondary" style={{ fontStyle: 'italic' }}>
            Сессия отброшена — показано только для справки, действия недоступны.
          </Text>
        )}
        <div>
          <Text strong>Модальности: </Text>
          <Space wrap>
            {session.available.map((m) => (
              <Tag color="green" key={m}>{m}</Tag>
            ))}
            {session.missing.map((m) => (
              <Tag color="default" key={m}>{m} — нет</Tag>
            ))}
          </Space>
        </div>

        <Divider style={{ margin: '8px 0' }} />

        <div>
          <Text strong>Неотобранные серии:</Text>
          {session.excluded_series.length === 0 ? (
            <p style={{ color: '#999' }}>Нет неотобранных серий</p>
          ) : (
            <List
              dataSource={session.excluded_series}
              renderItem={(entry) => {
                const modality = selectedModality[entry.original_path] || entry.detected_modality || undefined;
                const willReplace = modality && isAlreadyFilled(modality);
                const relabelButton = (
                  <Button
                    type="primary"
                    size="small"
                    loading={loadingPath === entry.original_path}
                    disabled={!modality}
                    onClick={willReplace ? undefined : () => handleRelabel(entry)}
                  >
                    Назначить
                  </Button>
                );
                return (
                  <List.Item>
                    <Space direction="vertical" size={2} style={{ width: '100%' }}>
                      <Text>{entry.series_description} ({entry.slice_count} срезов)</Text>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {entry.detected_modality
                          ? `Похоже на: ${entry.detected_modality} — ${REASON_LABELS[entry.reason] || entry.reason}`
                          : REASON_LABELS[entry.reason] || entry.reason}
                      </Text>
                      {!isReadOnly && (
                        <Space>
                          <Select
                            size="small"
                            style={{ width: 160 }}
                            placeholder="Модальность"
                            options={MODALITY_OPTIONS}
                            value={modality}
                            onChange={(value) =>
                              setSelectedModality((prev) => ({ ...prev, [entry.original_path]: value }))
                            }
                          />
                          {willReplace ? (
                            <Popconfirm
                              title="Эта модальность уже заполнена другой серией — заменить?"
                              onConfirm={() => handleRelabel(entry)}
                              okText="Да"
                              cancelText="Нет"
                            >
                              {relabelButton}
                            </Popconfirm>
                          ) : (
                            relabelButton
                          )}
                        </Space>
                      )}
                    </Space>
                  </List.Item>
                );
              }}
            />
          )}
        </div>

        {!isReadOnly && otherSessions.length > 0 && (
          <>
            <Divider style={{ margin: '8px 0' }} />
            <div>
              <Text strong>Объединить с другой сессией пациента:</Text>
              <div style={{ marginTop: 8 }}>
                <Space>
                  <Select
                    size="small"
                    style={{ width: 220 }}
                    placeholder="Выберите сессию"
                    value={donorSessionId}
                    onChange={setDonorSessionId}
                    options={otherSessions.map((s) => ({
                      label: `${s.session_id} (${s.date})`,
                      value: s.session_id,
                    }))}
                  />
                  <Popconfirm
                    title="Перенести серии выбранной сессии сюда как альтернативы?"
                    onConfirm={handleMerge}
                    okText="Да"
                    cancelText="Нет"
                  >
                    <Button size="small" loading={merging} disabled={!donorSessionId}>
                      Объединить
                    </Button>
                  </Popconfirm>
                </Space>
              </div>
            </div>
          </>
        )}

        <Divider style={{ margin: '8px 0' }} />

        {!isReadOnly && (
          <Popconfirm
            title="Отбросить сессию? Данные не удаляются, но она уйдёт из очереди review."
            onConfirm={handleDiscard}
            okText="Да"
            cancelText="Нет"
          >
            <Button danger icon={<DeleteOutlined />} loading={discarding}>
              Отбросить сессию
            </Button>
          </Popconfirm>
        )}
      </Space>
    </Modal>
  );
};

export default IncompletePatientDetail;
