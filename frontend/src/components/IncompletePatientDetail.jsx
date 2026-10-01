/**
 * Модальное окно с деталями одной сессии, требующей внимания врача:
 * какие модальности есть/не хватает, список исключённых серий с
 * возможностью назначить их на модальность, кнопка отбросить сессию.
 */
import { useState, useEffect } from 'react';
import { Modal, Tag, Space, List, Select, Button, Popconfirm, message, Divider, Typography, Checkbox } from 'antd';
import { DeleteOutlined } from '@ant-design/icons';
import { saveAssignment, discardSession, mergeSessions } from '../services/api';

const { Text } = Typography;

const REASON_LABELS = {
  unrecognized: 'алгоритм не распознал',
  lost_deduplication: 'алгоритм распознал, но выбрал другую копию',
  replaced_by_manual_relabel: 'заменена вручную ранее',
  from_other_session: 'перенесена из другой сессии пациента',
  cleared_by_doctor: 'снята врачом',
  currently_selected: 'сейчас выбрана для другой модальности',
};

/** Черновик набора по тому, что пришло с сервера. */
const draftFromSession = (session) => {
  const draft = {};
  (session?.selected || []).forEach((s) => { draft[s.modality] = s.original_path; });
  return draft;
};

const IncompletePatientDetail = ({ runId, session, sessions = [], visible, onClose, onActionComplete }) => {
  // Черновик: модальность -> original_path. На диск ничего не уходит,
  // пока не нажата «Сохранить», так что решение можно переиграть.
  const [draft, setDraft] = useState({});
  const [saving, setSaving] = useState(false);
  useEffect(() => { setDraft(draftFromSession(session)); }, [session]);
  const [discarding, setDiscarding] = useState(false);
  const [donorSessionId, setDonorSessionId] = useState(undefined);
  const [merging, setMerging] = useState(false);

  if (!session) return null;

  const isReadOnly = session.status === 'discarded' || session.status === 'merged';

  const initialDraft = draftFromSession(session);
  const isDirty = () => {
    const keys = new Set([...Object.keys(initialDraft), ...Object.keys(draft)]);
    return [...keys].some((k) => initialDraft[k] !== draft[k]);
  };

  /** Описание серии по её пути — она может лежать в любом из двух списков. */
  const describe = (path) => (
    (session.selected || []).find((x) => x.original_path === path)
    || (session.excluded_series || []).find((x) => x.original_path === path)
    || null
  );

  const assign = (modality, path) => setDraft((prev) => {
    const next = { ...prev };
    // Одна серия не может занимать две модальности: освобождаем прежнюю.
    Object.keys(next).forEach((m) => { if (next[m] === path) delete next[m]; });
    next[modality] = path;
    return next;
  });

  const clearModality = (modality) => setDraft((prev) => {
    const next = { ...prev };
    delete next[modality];
    return next;
  });

  const handleSave = async () => {
    setSaving(true);
    try {
      const result = await saveAssignment(
        runId, session.patient_id, session.session_id, draft,
      );
      message.success(
        result.needs_reprocess
          ? 'Сохранено. Пациент будет переобработан при следующем запуске.'
          : 'Сохранено.',
      );
      if (result.kappa_warning) {
        message.warning(result.kappa_warning, 8);
      }
      onActionComplete();
    } catch (err) {
      console.error('Ошибка сохранения набора:', err);
      message.error(err.response?.data?.detail || 'Не удалось сохранить набор');
    } finally {
      setSaving(false);
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

  // Пул неотобранных: всё, что сейчас не занято черновиком. Снятые галочкой
  // выборы тоже попадают сюда — иначе вернуть серию обратно было бы нечем.
  const taken = new Set(Object.values(draft));
  const pool = (session.excluded_series || [])
    .concat((session.selected || []).map((x) => ({
      original_path: x.original_path,
      series_description: x.series_description,
      slice_count: x.slice_count,
      detected_modality: x.modality,
      reason: 'currently_selected',
    })))
    .filter((e) => !taken.has(e.original_path));
  const recognized = pool.filter((e) => e.detected_modality);
  const unrecognized = pool.filter((e) => !e.detected_modality);

  const renderPoolItem = (entry) => (
    <List.Item key={entry.original_path}>
      <Space direction="vertical" size={2} style={{ width: '100%' }}>
        <Text>{entry.series_description} ({entry.slice_count} срезов)</Text>
        <Text type="secondary" style={{ fontSize: 12 }}>
          {entry.detected_modality
            ? `Похоже на: ${entry.detected_modality} — ${REASON_LABELS[entry.reason] || entry.reason}`
            : REASON_LABELS[entry.reason] || entry.reason}
        </Text>
        {!isReadOnly && (
          <Select
            size="small"
            style={{ width: 260 }}
            placeholder="Назначить на модальность"
            value={undefined}
            onChange={(modality) => assign(modality, entry.original_path)}
            options={(session.required || []).map((m) => ({
              value: m,
              label: draft[m] ? `${m} — заменить` : `${m} — свободна`,
            }))}
          />
        )}
      </Space>
    </List.Item>
  );

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
          <Text strong>Отобранные модальности</Text>
          <List
            size="small"
            dataSource={session.required || []}
            renderItem={(modality) => {
              const path = draft[modality];
              const info = path ? describe(path) : null;
              return (
                <List.Item key={modality}>
                  <Space>
                    <Checkbox
                      checked={!!path}
                      disabled={isReadOnly || !path}
                      onChange={() => clearModality(modality)}
                    />
                    <Tag color={path ? 'green' : 'default'}>{modality}</Tag>
                    <Text type={path ? undefined : 'secondary'}>
                      {info
                        ? `${info.series_description} (${info.slice_count} срезов)`
                        : 'не назначена'}
                    </Text>
                  </Space>
                </List.Item>
              );
            }}
          />
        </div>

        <Divider style={{ margin: '8px 0' }} />

        <div>
          <Text strong>Неотобранные серии</Text>
          {recognized.length === 0 ? (
            <p style={{ color: '#999' }}>Нет распознанных неотобранных серий</p>
          ) : (
            <List size="small" dataSource={recognized} renderItem={renderPoolItem} />
          )}

          {unrecognized.length > 0 && (
            <>
              <Text strong style={{ display: 'block', marginTop: 12 }}>
                Нераспознанные серии
              </Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                Алгоритм не смог их классифицировать — назначить можно так же.
              </Text>
              <List size="small" dataSource={unrecognized} renderItem={renderPoolItem} />
            </>
          )}
        </div>

        {!isReadOnly && (
          <Space>
            <Button type="primary" disabled={!isDirty()} loading={saving}
                    onClick={handleSave}>
              Сохранить
            </Button>
            <Button disabled={!isDirty()} onClick={() => setDraft(initialDraft)}>
              Отменить
            </Button>
          </Space>
        )}

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
