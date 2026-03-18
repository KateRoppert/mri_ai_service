/**
 * Компонент для отображения отчёта о лобарной локализации поражений.
 * Таблица с объёмами по долям и классам поражения.
 */
import { useEffect, useState } from 'react';
import { Modal, Table, Space, Spin, Alert, Tag, Tooltip, Row, Col, Statistic } from 'antd';
import { EnvironmentOutlined } from '@ant-design/icons';
import { getLobarReports } from '../services/api';

const LobarReport = ({ runId, visible, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [reports, setReports] = useState([]);

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
      width={850}
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

          {/* Общая статистика */}
          <Row gutter={32} style={{ marginBottom: 16 }}>
            <Col>
              <Statistic
                title="Общий объём поражения"
                value={report.total_lesion_volume_cm3}
                precision={3}
                suffix="см³"
              />
            </Col>
            <Col>
              <Statistic
                title="Затронуто долей"
                value={Object.keys(report.lobes).length}
                suffix="из 6"
              />
            </Col>
          </Row>

          {/* Таблица */}
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
        </div>
      ))}
    </Modal>
  );
};

export default LobarReport;
