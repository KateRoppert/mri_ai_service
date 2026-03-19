/**
 * Клинический отчёт — единый документ для врача.
 * 
 * Секция 1: Основные объёмы (CE+, CE−, суммарный, отёк)
 * Секция 2: Детализация по классам сегментации
 * Секция 3: Анатомическая локализация по долям мозга
 */
import { useEffect, useState } from 'react';
import { Modal, Table, Space, Spin, Alert, Tag, Tooltip, Row, Col, Statistic, Divider } from 'antd';
import { MedicineBoxOutlined, ExperimentOutlined, EnvironmentOutlined } from '@ant-design/icons';
import { getVolumeReports, getLobarReports } from '../services/api';

const ClinicalReport = ({ runId, visible, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [volumeReports, setVolumeReports] = useState([]);
  const [lobarReports, setLobarReports] = useState([]);

  useEffect(() => {
    if (visible && runId) {
      fetchAllData();
    }
  }, [visible, runId]);

  const fetchAllData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [volData, lobarData] = await Promise.all([
        getVolumeReports(runId).catch(() => ({ reports: [] })),
        getLobarReports(runId).catch(() => ({ reports: [] })),
      ]);
      setVolumeReports(volData.reports || []);
      setLobarReports(lobarData.reports || []);
    } catch (err) {
      console.error('Ошибка загрузки отчёта:', err);
      setError('Не удалось загрузить данные');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Парсинг текстового отчёта об объёмах
   */
  const parseVolumeReport = (reportText) => {
    const lines = reportText.split('\n');
    const classes = [];

    const labelNames = {
      1: 'Некротическое ядро (NCR)',
      2: 'Перитуморальный отёк (ED)',
      3: 'Неусиливающаяся опухоль (NET)',
      4: 'Усиливающаяся опухоль (ET)',
    };

    for (const line of lines) {
      const classMatch = line.match(/^\s+(\d)\.\s+(.+?)\s{2,}(\d+)\s+([\d.]+)\s+([\d.]+)/);
      if (classMatch) {
        const label = parseInt(classMatch[1]);
        classes.push({
          key: label,
          label,
          name: labelNames[label] || classMatch[2].trim(),
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

  /**
   * Таблица клинических объёмов
   */
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

  /**
   * Таблица детализации по классам
   */
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
    1: '#ff4d4f',   // NCR - красный
    2: '#52c41a',   // ED - зелёный
    3: '#faad14',   // NET - жёлтый
    4: '#1890ff',   // ET - синий
  };

  /**
   * Таблица локализации по долям
   */
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

  return (
    <Modal
      title={
        <Space>
          <MedicineBoxOutlined />
          <span>Клинический отчёт</span>
        </Space>
      }
      open={visible}
      onCancel={onClose}
      width={900}
      footer={null}
      styles={{ body: { maxHeight: '80vh', overflowY: 'auto' } }}
    >
      {loading && (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
          <p style={{ marginTop: 16 }}>Загрузка отчёта...</p>
        </div>
      )}

      {error && (
        <Alert description={error} type="error" showIcon style={{ marginBottom: 16 }} />
      )}

      {volumeReports.map((report, idx) => {
        const classes = parseVolumeReport(report.report_text);
        const clinical = computeClinical(classes);
        const lobar = lobarReports.find(
          lr => lr.patient_id === report.patient_id && lr.session_id === report.session_id
        );

        return (
          <div key={idx} style={{ marginBottom: 32 }}>
            {/* Заголовок */}
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
    </Modal>
  );
};

export default ClinicalReport;
