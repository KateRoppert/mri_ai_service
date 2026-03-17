/**
 * Компонент для отображения отчёта о лобарной локализации поражений.
 * Содержит таблицу с объёмами по долям и SVG-схему мозга с подсветкой.
 */
import { useEffect, useState } from 'react';
import { Modal, Table, Space, Spin, Alert, Tag, Tooltip, Row, Col, Statistic } from 'antd';
import { EnvironmentOutlined } from '@ant-design/icons';
import { getLobarReports } from '../services/api';

/**
 * SVG-схема мозга (вид сбоку) с интерактивными долями
 */
const BrainDiagram = ({ lobes, onLobeHover, hoveredLobe }) => {
  // Упрощённые контуры долей мозга (вид сбоку, левое полушарие)
  const lobePaths = {
    frontal: {
      path: "M 85,155 C 75,140 65,120 60,100 C 55,80 58,60 70,45 C 82,30 100,22 120,20 C 140,18 160,20 175,25 C 185,28 192,35 195,45 L 195,95 C 192,100 188,105 182,110 C 175,118 168,128 162,140 C 155,152 145,158 130,160 Z",
      center: [125, 90],
    },
    parietal: {
      path: "M 195,45 C 200,55 205,65 210,80 C 218,100 225,115 228,130 C 230,140 228,148 222,155 C 215,162 205,165 195,162 L 195,95 C 192,100 188,105 182,110 C 175,118 168,128 162,140 C 155,152 145,158 130,160 L 130,160 C 140,158 150,152 160,140 C 168,128 175,118 182,110 C 188,105 192,100 195,95 Z",
      center: [208, 100],
    },
    temporal: {
      path: "M 85,155 C 95,160 105,168 115,178 C 130,192 150,200 175,202 C 200,204 218,198 230,188 C 238,180 240,170 235,160 C 230,152 225,148 222,155 C 215,162 205,165 195,162 C 185,160 170,158 155,155 C 140,152 125,155 110,158 C 100,160 90,158 85,155 Z",
      center: [165, 178],
    },
    occipital: {
      path: "M 228,130 C 232,140 235,150 235,160 C 238,170 240,178 238,185 C 235,192 230,196 225,195 C 218,194 212,188 208,180 C 204,172 202,162 200,155 C 198,148 196,145 195,150 L 195,162 C 205,165 215,162 222,155 C 228,148 230,140 228,130 Z",
      center: [225, 162],
    },
    insula: {
      path: "M 158,130 C 162,125 168,122 175,122 C 182,122 187,126 190,132 C 192,138 190,145 185,148 C 180,151 173,150 168,146 C 163,142 158,136 158,130 Z",
      center: [174, 136],
    },
    cingulate: {
      path: "M 120,55 C 135,48 155,45 175,48 C 188,50 195,55 198,62 C 200,70 198,78 192,82 C 186,86 178,85 170,80 C 162,75 152,72 140,72 C 128,72 118,76 112,82 C 106,75 105,65 112,58 Z",
      center: [155, 65],
    },
  };

  return (
    <svg viewBox="40 5 220 215" style={{ width: '100%', maxWidth: 380, height: 'auto' }}>
      {/* Контур мозга (фон) */}
      <ellipse cx="160" cy="115" rx="105" ry="95" 
        fill="none" stroke="#d9d9d9" strokeWidth="1.5" strokeDasharray="4,3" />
      
      {/* Доли */}
      {Object.entries(lobePaths).map(([lobeKey, { path, center }]) => {
        const lobeData = lobes?.[lobeKey];
        const isAffected = lobeData && lobeData.total_voxels > 0;
        const isHovered = hoveredLobe === lobeKey;
        const fillColor = isAffected ? lobeData.color : '#f0f0f0';
        const fillOpacity = isAffected ? (isHovered ? 0.85 : 0.55) : 0.3;

        return (
          <g key={lobeKey}
            onMouseEnter={() => onLobeHover?.(lobeKey)}
            onMouseLeave={() => onLobeHover?.(null)}
            style={{ cursor: isAffected ? 'pointer' : 'default' }}
          >
            <path
              d={path}
              fill={fillColor}
              fillOpacity={fillOpacity}
              stroke={isAffected ? fillColor : '#ccc'}
              strokeWidth={isHovered ? 2.5 : 1.2}
              strokeOpacity={isAffected ? 0.9 : 0.5}
            />
            {/* Название доли */}
            <text
              x={center[0]}
              y={center[1]}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={isAffected ? "9" : "8"}
              fontWeight={isAffected ? "600" : "400"}
              fill={isAffected ? '#222' : '#999'}
            >
              {lobeData?.name_ru?.replace(' доля', '').replace(' извилина', '').replace(' кора', '').replace('Поясная ', 'Поясн.').replace('Островковая ', 'Остр.') || lobeKey}
            </text>
            {/* Процент при наличии поражения */}
            {isAffected && (
              <text
                x={center[0]}
                y={center[1] + 12}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize="8"
                fill="#555"
              >
                {lobeData.percent_of_lesion}%
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
};

const LobarReport = ({ runId, visible, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [reports, setReports] = useState([]);
  const [hoveredLobe, setHoveredLobe] = useState(null);

  useEffect(() => {
    if (visible && runId) {
      fetchReports();
    }
  }, [visible, runId]);

  const fetchReports = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getLobarReports(runId);
      setReports(data.reports || []);
    } catch (err) {
      console.error('Ошибка загрузки лобарных отчётов:', err);
      setError('Не удалось загрузить отчёты о локализации');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Колонки таблицы
   */
  const columns = [
    {
      title: 'Доля мозга',
      dataIndex: 'name_ru',
      key: 'name_ru',
      width: 180,
      render: (text, record) => (
        <Space>
          <div style={{
            width: 12, height: 12, borderRadius: 2,
            backgroundColor: record.color,
            border: '1px solid rgba(0,0,0,0.1)'
          }} />
          <span style={{ fontWeight: hoveredLobe === record.key ? 600 : 400 }}>
            {text}
          </span>
        </Space>
      ),
      onCell: (record) => ({
        onMouseEnter: () => setHoveredLobe(record.key),
        onMouseLeave: () => setHoveredLobe(null),
      }),
    },
    {
      title: 'Объём (см³)',
      dataIndex: 'total_volume_cm3',
      key: 'volume',
      width: 110,
      align: 'right',
      render: (val) => val.toFixed(3),
      sorter: (a, b) => a.total_volume_cm3 - b.total_volume_cm3,
      defaultSortOrder: 'descend',
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
      title: 'NCR',
      key: 'ncr',
      width: 80,
      align: 'right',
      render: (_, record) => {
        const cls = record.classes?.['1'];
        return cls ? cls.volume_cm3.toFixed(3) : '—';
      },
    },
    {
      title: 'ED',
      key: 'ed',
      width: 80,
      align: 'right',
      render: (_, record) => {
        const cls = record.classes?.['2'];
        return cls ? cls.volume_cm3.toFixed(3) : '—';
      },
    },
    {
      title: 'NET',
      key: 'net',
      width: 80,
      align: 'right',
      render: (_, record) => {
        const cls = record.classes?.['3'];
        return cls ? cls.volume_cm3.toFixed(3) : '—';
      },
    },
    {
      title: 'ET',
      key: 'et',
      width: 80,
      align: 'right',
      render: (_, record) => {
        const cls = record.classes?.['4'];
        return cls ? cls.volume_cm3.toFixed(3) : '—';
      },
    },
  ];

  /**
   * Преобразовать lobes из отчёта в dataSource для таблицы
   */
  const getTableData = (report) => {
    if (!report?.lobes) return [];
    return Object.entries(report.lobes).map(([key, lobe]) => ({
      key,
      ...lobe,
    }));
  };

  return (
    <Modal
      title={
        <Space>
          <EnvironmentOutlined />
          <span>Анатомическая локализация поражений</span>
        </Space>
      }
      open={visible}
      onCancel={onClose}
      width={950}
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
          description={error}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {reports.map((report, idx) => (
        <div key={idx} style={{ marginBottom: 24 }}>
          {/* Заголовок пациента */}
          <div style={{ marginBottom: 16 }}>
            <Tag>{report.patient_id}</Tag>
            <Tag>{report.session_id}</Tag>
            <Tag color="geekblue">{report.atlas_name}</Tag>
          </div>

          <Row gutter={24}>
            {/* Схема мозга */}
            <Col span={9}>
              <div style={{
                background: '#fafafa',
                borderRadius: 8,
                padding: '16px 8px',
                textAlign: 'center',
                border: '1px solid #f0f0f0',
              }}>
                <BrainDiagram
                  lobes={report.lobes}
                  onLobeHover={setHoveredLobe}
                  hoveredLobe={hoveredLobe}
                />
                {/* Общая статистика */}
                <Row gutter={16} style={{ marginTop: 12 }}>
                  <Col span={12}>
                    <Statistic
                      title="Общий объём"
                      value={report.total_lesion_volume_cm3}
                      precision={3}
                      suffix="см³"
                      valueStyle={{ fontSize: 16 }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="Затронуто долей"
                      value={Object.keys(report.lobes).length}
                      suffix="из 6"
                      valueStyle={{ fontSize: 16 }}
                    />
                  </Col>
                </Row>
              </div>
            </Col>

            {/* Таблица */}
            <Col span={15}>
              <Table
                columns={columns}
                dataSource={getTableData(report)}
                pagination={false}
                size="small"
                bordered
                summary={() => {
                  const data = getTableData(report);
                  const totalVol = data.reduce((s, r) => s + r.total_volume_cm3, 0);
                  const totalPct = data.reduce((s, r) => s + r.percent_of_lesion, 0);
                  const uncovered = report.total_lesion_volume_cm3 - totalVol;
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
                        <Table.Summary.Cell index={3} colSpan={4} />
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
                          <Table.Summary.Cell index={3} colSpan={4} />
                        </Table.Summary.Row>
                      )}
                    </>
                  );
                }}
              />
            </Col>
          </Row>
        </div>
      ))}
    </Modal>
  );
};

export default LobarReport;
