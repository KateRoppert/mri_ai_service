/**
 * Содержимое клинического отчёта — переиспользуемый компонент.
 * Используется внутри модального окна ClinicalReport и встроенным под визуализацией.
 *
 * Секция 1: Основные объёмы (CE+, CE−, суммарный, отёк)
 * Секция 2: Детализация по классам сегментации
 * Секция 3: Анатомическая локализация по долям мозга
 */
import { useEffect, useState } from 'react';
import { Table, Space, Spin, Alert, Tag, Tooltip, Row, Col, Statistic, Divider, Collapse } from 'antd';
import { MedicineBoxOutlined, ExperimentOutlined, EnvironmentOutlined } from '@ant-design/icons';
import { getVolumeReports, getLobarReports, getLesionStatsReports } from '../services/api';
import LongitudinalTimeline from './LongitudinalTimeline';

// Sort report blocks by patient, then session (chronological — ses-001
// is earliest). Critical for MS where session order carries meaning.
const sortByPatientSession = (arr) =>
  [...(arr || [])].sort((a, b) => {
    const p = (a.patient_id || '').localeCompare(b.patient_id || '');
    if (p !== 0) return p;
    return (a.session_id || '').localeCompare(b.session_id || '');
  });

// MS lesion size bands (pragmatic; a ~3 mm punctate lesion ≈ 0.014 см³).
const LESION_SIZE_BANDS = [
  { key: 'large',  label: 'крупные ≥0.1 см³',      bg: '#f6ffed', border: '#b7eb8f', color: '#389e0d', test: (v) => v >= 0.1 },
  { key: 'medium', label: 'средние 0.01–0.1 см³',  bg: '#fcffe6', border: '#eaff8f', color: '#7cb305', test: (v) => v >= 0.01 && v < 0.1 },
  { key: 'small',  label: 'точечные <0.01 см³',    bg: '#fafafa', border: '#e8e8e8', color: '#8c8c8c', test: (v) => v < 0.01 },
];

const countLesionBands = (volumes) =>
  LESION_SIZE_BANDS.map((b) => ({ ...b, count: (volumes || []).filter(b.test).length }));

// Glio segmentation class label ↔ name. Shared by the text parser (local
// source) and the Kappa normalizer so both produce identical class objects.
const LABEL_NAMES = {
  1: 'Некротическое ядро (NCR)',
  2: 'Перитуморальный отёк (ED)',
  3: 'Неусиливающаяся опухоль (NET)',
  4: 'Усиливающаяся опухоль (ET)',
};
const NAME_TO_LABEL = { NCR: 1, ED: 2, NET: 3, ET: 4 };

// Split a Kappa bids_id ("sub-001_ses-002") into patient and session ids.
const splitBidsId = (bids) => {
  const s = bids || '';
  const i = s.indexOf('_ses-');
  return i !== -1
    ? { patient_id: s.slice(0, i), session_id: s.slice(i + 1) }
    : { patient_id: s, session_id: '' };
};

// Normalize one Kappa entity's dsEntityInfo into the SAME state shapes the
// local API produces, so the render path is shared (single source of truth
// for presentation). Each array holds 0 or 1 entry (one entity = one session).
const normalizeKappaEntity = (info) => {
  const { patient_id, session_id } = splitBidsId(info.bids_id);

  const lesionStatsReports = info.lesion_stats
    ? [{
        patient_id,
        session_id,
        lesion_count: info.lesion_stats.lesion_count,
        total_volume_cm3: info.lesion_stats.total_volume_cm3,
        mean_lesion_volume_cm3: info.lesion_stats.mean_lesion_volume_cm3,
        lesion_volumes_cm3: info.lesion_stats.lesion_volumes_cm3 || [],
        lesion_volumes_by_label: info.lesion_stats.lesion_volumes_by_label || {},
      }]
    : [];

  const lobarReports = info.lobar_report
    ? [{
        patient_id,
        session_id,
        atlas_name: info.lobar_report.atlas_name,
        total_lesion_volume_cm3: info.lobar_report.total_lesion_cm3,
        lobes: Object.fromEntries(
          Object.entries(info.lobar_report.lobes || {}).map(([id, l]) => [id, {
            name_ru: l.name_ru || id,
            color: l.color,
            total_volume_cm3: l.cm3,
            percent_of_lesion: l.percent,
            classes: {},
          }]),
        ),
      }]
    : [];

  const volumeReports = info.volume_report
    ? [{
        patient_id,
        session_id,
        classes: Object.entries(info.volume_report.classes || {}).map(([name, d]) => {
          const label = NAME_TO_LABEL[name];
          return {
            key: label || name,
            label,
            name: LABEL_NAMES[label] || name,
            voxel_count: d.voxels || 0,
            volume_mm3: (d.cm3 || 0) * 1000,
            volume_cm3: d.cm3 || 0,
          };
        }),
      }]
    : [];

  return { volumeReports, lobarReports, lesionStatsReports };
};

const ClinicalReportContent = ({ runId, autoLoad = false, lesionType = 'glioblastoma', kappaEntityInfo = null }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [volumeReports, setVolumeReports] = useState([]);
  const [lobarReports, setLobarReports] = useState([]);
  const [lesionStatsReports, setLesionStatsReports] = useState([]);
  const [loaded, setLoaded] = useState(false);

  // Kappa source (validation): normalize dsEntityInfo into the same state
  // shapes the local API produces, then the shared render handles it.
  useEffect(() => {
    if (kappaEntityInfo) {
      const { volumeReports: v, lobarReports: l, lesionStatsReports: s } =
        normalizeKappaEntity(kappaEntityInfo);
      setVolumeReports(v);
      setLobarReports(l);
      setLesionStatsReports(s);
      setLoaded(true);
    }
  }, [kappaEntityInfo]);

  // Reset stale data whenever the source changes. NIfTIViewer stays mounted
  // between pipeline runs (visibility is controlled via a prop, not conditional
  // rendering), so the inner ClinicalReportContent keeps loaded=true from the
  // previous run and never re-fetches when runId changes. Explicit reset fixes this.
  useEffect(() => {
    setLoaded(false);
    setVolumeReports([]);
    setLobarReports([]);
    setLesionStatsReports([]);
    setError(null);
  }, [runId, kappaEntityInfo]);

  // Local source (run/history): fetch report files by runId.
  // The !loaded guard is kept but the reset above ensures it fires on source change.
  useEffect(() => {
    if (!kappaEntityInfo && autoLoad && runId && !loaded) {
      fetchAllData();
    }
  }, [autoLoad, runId, kappaEntityInfo, loaded]);

  const fetchAllData = async () => {
    setLoading(true);
    setError(null);
    try {
      const fetches = [
        getVolumeReports(runId).catch(() => ({ reports: [] })),
        getLobarReports(runId).catch(() => ({ reports: [] })),
      ];
      if (lesionType === 'multiple_sclerosis') {
        fetches.push(getLesionStatsReports(runId).catch(() => ({ reports: [] })));
      }
      const results = await Promise.all(fetches);
      setVolumeReports(sortByPatientSession(results[0].reports));
      setLobarReports(results[1].reports || []);
      if (lesionType === 'multiple_sclerosis') {
        setLesionStatsReports(sortByPatientSession(results[2]?.reports));
      }
      setLoaded(true);
    } catch (err) {
      console.error('Ошибка загрузки отчёта:', err);
      setError('Не удалось загрузить данные');
    } finally {
      setLoading(false);
    }
  };

  // Можно вызвать извне для загрузки данных
  // (используется в ClinicalReport при visible=true)
  const load = () => {
    if (!loaded) fetchAllData();
  };

  /**
   * Парсинг текстового отчёта об объёмах
   */
  const parseVolumeReport = (reportText) => {
    const lines = reportText.split('\n');
    const classes = [];

    for (const line of lines) {
      const classMatch = line.match(/^\s+(\d)\.\s+(.+?)\s{2,}(\d+)\s+([\d.]+)\s+([\d.]+)/);
      if (classMatch) {
        const label = parseInt(classMatch[1]);
        classes.push({
          key: label,
          label,
          name: LABEL_NAMES[label] || classMatch[2].trim(),
          voxel_count: parseInt(classMatch[3]),
          volume_mm3: parseFloat(classMatch[4]),
          volume_cm3: parseFloat(classMatch[5]),
        });
      }
    }

    return classes;
  };

  /**
   * Вычисление клинических объёмов из классов
   */
  const computeClinical = (classes) => {
    const get = (label) => classes.find(c => c.label === label) || { volume_cm3: 0, volume_mm3: 0, voxel_count: 0 };
    
    const et = get(4);
    const ncr = get(1);
    const net = get(3);
    const ed = get(2);

    return {
      ce_positive: { ...et, name: 'Контраст-позитивная часть (CE+)' },
      ce_negative: {
        name: 'Контраст-негативная часть (CE−)',
        voxel_count: ncr.voxel_count + net.voxel_count,
        volume_mm3: ncr.volume_mm3 + net.volume_mm3,
        volume_cm3: +(ncr.volume_cm3 + net.volume_cm3).toFixed(4),
      },
      tumor_core: {
        name: 'Суммарный объём опухоли',
        voxel_count: et.voxel_count + ncr.voxel_count + net.voxel_count,
        volume_mm3: et.volume_mm3 + ncr.volume_mm3 + net.volume_mm3,
        volume_cm3: +(et.volume_cm3 + ncr.volume_cm3 + net.volume_cm3).toFixed(4),
      },
      edema: { ...ed, name: 'Перитуморальный отёк' },
    };
  };

  const clinicalColumns = [
    {
      title: 'Показатель',
      dataIndex: 'name',
      key: 'name',
      width: 280,
      render: (text, record) => (
        <Space>
          <div style={{
            width: 12, height: 12, borderRadius: 2,
            backgroundColor: record.color,
            border: '1px solid rgba(0,0,0,0.1)',
          }} />
          <span style={{ fontWeight: record.bold ? 600 : 400 }}>{text}</span>
        </Space>
      ),
    },
    {
      title: 'Объём (см³)',
      dataIndex: 'volume_cm3',
      key: 'cm3',
      width: 120,
      align: 'right',
      render: (val, record) => (
        <span style={{ fontWeight: record.bold ? 600 : 400 }}>{val.toFixed(3)}</span>
      ),
    },
    {
      title: 'Объём (мм³)',
      dataIndex: 'volume_mm3',
      key: 'mm3',
      width: 120,
      align: 'right',
      render: (val) => val.toFixed(2),
    },
    {
      title: 'Воксели',
      dataIndex: 'voxel_count',
      key: 'voxels',
      width: 100,
      align: 'right',
      render: (val) => val.toLocaleString(),
    },
  ];

  const getClinicalTableData = (clinical) => [
    { key: 'ce_pos',   ...clinical.ce_positive, color: '#ff4d4f' },
    { key: 'ce_neg',   ...clinical.ce_negative, color: '#faad14' },
    { key: 'tumor',    ...clinical.tumor_core,   color: '#1890ff', bold: true },
    { key: 'edema',    ...clinical.edema,         color: '#52c41a' },
  ];

  const classColumns = [
    {
      title: 'Класс сегментации',
      dataIndex: 'name',
      key: 'name',
      width: 280,
      render: (text, record) => (
        <Space>
          <div style={{
            width: 12, height: 12, borderRadius: 2,
            backgroundColor: record.color,
            border: '1px solid rgba(0,0,0,0.1)',
          }} />
          <span>{text}</span>
        </Space>
      ),
    },
    {
      title: 'Объём (см³)',
      dataIndex: 'volume_cm3',
      key: 'cm3',
      width: 120,
      align: 'right',
      render: (val) => val.toFixed(4),
    },
    {
      title: 'Объём (мм³)',
      dataIndex: 'volume_mm3',
      key: 'mm3',
      width: 120,
      align: 'right',
      render: (val) => val.toFixed(2),
    },
    {
      title: 'Воксели',
      dataIndex: 'voxel_count',
      key: 'voxels',
      width: 100,
      align: 'right',
      render: (val) => val.toLocaleString(),
    },
  ];

  const classColors = {
    1: '#ff4d4f',
    2: '#52c41a',
    3: '#faad14',
    4: '#1890ff',
  };

  const lobarColumns = [
    {
      title: 'Доля мозга',
      dataIndex: 'name_ru',
      key: 'name',
      width: 180,
      render: (text, record) => (
        <Space>
          <div style={{
            width: 12, height: 12, borderRadius: 2,
            backgroundColor: record.color,
            border: '1px solid rgba(0,0,0,0.1)',
          }} />
          <span>{text}</span>
        </Space>
      ),
    },
    {
      title: 'Объём (см³)',
      dataIndex: 'total_volume_cm3',
      key: 'volume',
      width: 110,
      align: 'right',
      render: (val) => val.toFixed(3),
      defaultSortOrder: 'descend',
      sorter: (a, b) => a.total_volume_cm3 - b.total_volume_cm3,
    },
    {
      title: '% поражения',
      dataIndex: 'percent_of_lesion',
      key: 'percent',
      width: 110,
      align: 'right',
      render: (val) => (
        <Tag color={val > 30 ? 'red' : val > 10 ? 'orange' : 'blue'}>
          {val.toFixed(1)}%
        </Tag>
      ),
    },
    {
      title: 'CE+',
      key: 'et',
      width: 80,
      align: 'right',
      render: (_, record) => {
        const cls = record.classes?.['4'];
        return cls ? cls.volume_cm3.toFixed(3) : '—';
      },
    },
    {
      title: 'CE−',
      key: 'ce_neg',
      width: 80,
      align: 'right',
      render: (_, record) => {
        const ncr = record.classes?.['1']?.volume_cm3 || 0;
        const net = record.classes?.['3']?.volume_cm3 || 0;
        const sum = ncr + net;
        return sum > 0 ? sum.toFixed(3) : '—';
      },
    },
    {
      title: 'Отёк',
      key: 'ed',
      width: 80,
      align: 'right',
      render: (_, record) => {
        const cls = record.classes?.['2'];
        return cls ? cls.volume_cm3.toFixed(3) : '—';
      },
    },
  ];

  const getLobarTableData = (report) => {
    if (!report?.lobes) return [];
    return Object.entries(report.lobes).map(([key, lobe]) => ({ key, ...lobe }));
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '40px 0' }}>
        <Spin size="large" />
        <p style={{ marginTop: 16 }}>Загрузка отчёта...</p>
      </div>
    );
  }

  if (error) {
    return <Alert description={error} type="error" showIcon style={{ marginBottom: 16 }} />;
  }

  if (!loaded || (lesionType !== 'multiple_sclerosis' && volumeReports.length === 0)) {
    return null;
  }

  // ===== MS RENDER PATH =====
  if (lesionType === 'multiple_sclerosis') {
    if (!loaded || lesionStatsReports.length === 0) return null;
    return (
      <>
        {lesionStatsReports.map((stats, idx) => {
          const bands = countLesionBands(stats.lesion_volumes_cm3);
          const perLesionRows = (stats.lesion_volumes_cm3 || []).map((v, i) => ({
            key: i, n: i + 1, cm3: v,
          }));
          return (
            <div key={idx} style={{ marginBottom: 32 }}>
              <div style={{ marginBottom: 16 }}>
                <Tag>{stats.patient_id}</Tag>
                <Tag>{stats.session_id}</Tag>
              </div>

              {/* Очаговая нагрузка */}
              <Divider orientation="left" style={{ fontSize: 14 }}>
                <Space><MedicineBoxOutlined /> Очаговая нагрузка</Space>
              </Divider>
              <Row gutter={32} style={{ marginBottom: 16 }}>
                <Col>
                  <Statistic title="Количество очагов" value={stats.lesion_count}
                    valueStyle={{ color: '#1890ff' }} />
                </Col>
                <Col>
                  <Statistic title="Суммарный объём поражения" value={stats.total_volume_cm3}
                    precision={3} suffix="см³" valueStyle={{ color: '#52c41a' }} />
                </Col>
                <Col>
                  <Statistic title="Средний объём очага" value={stats.mean_lesion_volume_cm3}
                    precision={3} suffix="см³" />
                </Col>
              </Row>

              {/* Характер поражения */}
              <Divider orientation="left" style={{ fontSize: 14 }}>
                <Space><ExperimentOutlined /> Характер поражения</Space>
              </Divider>
              <Row gutter={12} style={{ marginBottom: 16, maxWidth: 460 }}>
                {bands.map((b) => (
                  <Col span={8} key={b.key}>
                    <div style={{
                      textAlign: 'center', background: b.bg, border: `1px solid ${b.border}`,
                      borderRadius: 6, padding: '10px 4px',
                    }}>
                      <div style={{ fontSize: 22, fontWeight: 700, color: b.color }}>{b.count}</div>
                      <div style={{ fontSize: 11, color: '#888' }}>{b.label}</div>
                    </div>
                  </Col>
                ))}
              </Row>

              {/* Объёмы всех очагов — свёрнуто, для протокола */}
              <Collapse ghost style={{ marginBottom: 8 }} items={[{
                key: 'lesions',
                label: `Объёмы всех очагов (${perLesionRows.length}) — для протокола`,
                children: (
                  <Table
                    columns={[
                      { title: '№', dataIndex: 'n', key: 'n', width: 60 },
                      { title: 'Объём (см³)', dataIndex: 'cm3', key: 'cm3', align: 'right',
                        render: (v) => v.toFixed(4) },
                    ]}
                    dataSource={perLesionRows}
                    pagination={false}
                    size="small"
                    bordered
                    scroll={{ y: 220 }}
                    style={{ maxWidth: 300 }}
                  />
                ),
              }]} />

              {/* Динамика между сессиями */}
              <Divider orientation="left" style={{ fontSize: 14 }}>
                <Space>📈 Динамика между сессиями</Space>
              </Divider>
              <LongitudinalTimeline patientId={stats.patient_id} lesionType="multiple_sclerosis" />
            </div>
          );
        })}
      </>
    );
  }

  // ===== GLIO RENDER PATH =====
  return (
    <>
      {volumeReports.map((report, idx) => {
        // Kappa source provides structured classes; local source has report_text.
        const classes = report.classes || parseVolumeReport(report.report_text || '');
        const clinical = computeClinical(classes);
        const lobar = lobarReports.find(
          lr => lr.patient_id === report.patient_id && lr.session_id === report.session_id
        );

        return (
          <div key={idx} style={{ marginBottom: 32 }}>
            <div style={{ marginBottom: 16 }}>
              <Tag>{report.patient_id}</Tag>
              <Tag>{report.session_id}</Tag>
            </div>

            {/* ===== СЕКЦИЯ 1: Клинические объёмы ===== */}
            <Divider orientation="left" style={{ fontSize: 14 }}>
              <Space><MedicineBoxOutlined /> Объёмы поражения</Space>
            </Divider>

            <Row gutter={32} style={{ marginBottom: 16 }}>
              <Col>
                <Statistic
                  title="Суммарная опухоль"
                  value={clinical.tumor_core.volume_cm3}
                  precision={3}
                  suffix="см³"
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col>
                <Statistic
                  title="CE+ (активная)"
                  value={clinical.ce_positive.volume_cm3}
                  precision={3}
                  suffix="см³"
                  valueStyle={{ color: '#ff4d4f' }}
                />
              </Col>
              <Col>
                <Statistic
                  title="CE− (неактивная)"
                  value={clinical.ce_negative.volume_cm3}
                  precision={3}
                  suffix="см³"
                  valueStyle={{ color: '#faad14' }}
                />
              </Col>
              <Col>
                <Statistic
                  title="Перитуморальный отёк"
                  value={clinical.edema.volume_cm3}
                  precision={3}
                  suffix="см³"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
            </Row>

            <Table
              columns={clinicalColumns}
              dataSource={getClinicalTableData(clinical)}
              pagination={false}
              size="small"
              bordered
              style={{ marginBottom: 24 }}
            />

            {/* ===== СЕКЦИЯ 2: По классам ===== */}
            <Divider orientation="left" style={{ fontSize: 14 }}>
              <Space><ExperimentOutlined /> Детализация по классам</Space>
            </Divider>

            <Table
              columns={classColumns}
              dataSource={classes.map(c => ({ ...c, color: classColors[c.label] || '#ccc' }))}
              pagination={false}
              size="small"
              bordered
              style={{ marginBottom: 24 }}
            />

            {/* ===== СЕКЦИЯ 3: Локализация ===== */}
            {lobar && (
              <>
                <Divider orientation="left" style={{ fontSize: 14 }}>
                  <Space><EnvironmentOutlined /> Анатомическая локализация</Space>
                </Divider>

                <Row gutter={32} style={{ marginBottom: 16 }}>
                  <Col>
                    <Statistic
                      title="Общий объём поражения"
                      value={lobar.total_lesion_volume_cm3}
                      precision={3}
                      suffix="см³"
                    />
                  </Col>
                  <Col>
                    <Statistic
                      title="Затронуто долей"
                      value={Object.keys(lobar.lobes).length}
                      suffix="из 6"
                    />
                  </Col>
                  <Col>
                    <span style={{ color: '#999', fontSize: 12 }}>
                      <Tag color="geekblue">{lobar.atlas_name}</Tag>
                    </span>
                  </Col>
                </Row>

                <Table
                  columns={lobarColumns}
                  dataSource={getLobarTableData(lobar)}
                  pagination={false}
                  size="small"
                  bordered
                  summary={() => {
                    const data = getLobarTableData(lobar);
                    const totalVol = data.reduce((s, r) => s + r.total_volume_cm3, 0);
                    const totalPct = data.reduce((s, r) => s + r.percent_of_lesion, 0);
                    const uncovered = lobar.total_lesion_volume_cm3 - totalVol;
                    return (
                      <>
                        <Table.Summary.Row>
                          <Table.Summary.Cell index={0}>
                            <strong>Итого (в зонах атласа)</strong>
                          </Table.Summary.Cell>
                          <Table.Summary.Cell index={1} align="right">
                            <strong>{totalVol.toFixed(3)}</strong>
                          </Table.Summary.Cell>
                          <Table.Summary.Cell index={2} align="right">
                            <strong>{totalPct.toFixed(1)}%</strong>
                          </Table.Summary.Cell>
                          <Table.Summary.Cell index={3} colSpan={3} />
                        </Table.Summary.Row>
                        {uncovered > 0.001 && (
                          <Table.Summary.Row>
                            <Table.Summary.Cell index={0}>
                              <Tooltip title="Воксели поражения вне кортикальных зон атласа (белое вещество, подкорковые структуры)">
                                <span style={{ color: '#999' }}>Вне зон атласа</span>
                              </Tooltip>
                            </Table.Summary.Cell>
                            <Table.Summary.Cell index={1} align="right">
                              <span style={{ color: '#999' }}>{uncovered.toFixed(3)}</span>
                            </Table.Summary.Cell>
                            <Table.Summary.Cell index={2} align="right">
                              <span style={{ color: '#999' }}>{(100 - totalPct).toFixed(1)}%</span>
                            </Table.Summary.Cell>
                            <Table.Summary.Cell index={3} colSpan={3} />
                          </Table.Summary.Row>
                        )}
                      </>
                    );
                  }}
                />
              </>
            )}

            {/* Примечание */}
            <div style={{
              marginTop: 20,
              padding: '10px 14px',
              background: '#fafafa',
              borderRadius: 4,
              border: '1px solid #f0f0f0',
              fontSize: 12,
              color: '#888',
              lineHeight: 1.6,
            }}>
              Результаты анатомической локализации носят ориентировочный характер.
              Границы долей определены по популяционному атласу Harvard-Oxford
              и могут не учитывать индивидуальные особенности и масс-эффект опухоли.
              CE+ — контраст-позитивная часть опухоли (усиливающаяся при контрастировании, label&nbsp;4).
              CE− — контраст-негативная часть (некротическое ядро + неусиливающаяся опухоль, labels&nbsp;1+3).
            </div>
          </div>
        );
      })}
    </>
  );
};

export default ClinicalReportContent;