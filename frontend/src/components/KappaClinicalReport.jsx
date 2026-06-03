/**
 * Клинический отчёт для секции Валидации — источник данных ТОЛЬКО Каппа.
 *
 * В отличие от ClinicalReportContent (читает локальные файлы прогона по run_id),
 * этот компонент рендерит отчёт из entity.dsEntityInfo, выгруженного из Каппы.
 * Это нужно, чтобы эксперт, работающий только с Каппой (без доступа к локальным
 * файлам пайплайна), видел корректный отчёт.
 *
 * Состав отчёта:
 *  - МС  → lesion_stats (количество, объёмы очагов) + lobar_report (доли)
 *  - Глио → volume_report (классы NCR/ED/NET/ET) + lobar_report (доли)
 *
 * Подробный редизайн состава МС-отчёта — отдельная задача.
 */
import { Table, Space, Tag, Row, Col, Statistic, Divider, Empty } from 'antd';
import { MedicineBoxOutlined, ExperimentOutlined, EnvironmentOutlined } from '@ant-design/icons';

const KappaClinicalReport = ({ entityInfo, lesionType = 'glioblastoma' }) => {
  if (!entityInfo) {
    return <Empty description="Нет данных отчёта в Каппе" />;
  }

  const lobar = entityInfo.lobar_report;
  const lesionStats = entityInfo.lesion_stats;
  const volume = entityInfo.volume_report;

  // ===== Таблица по долям (общая для МС и глио) =====
  const lobarRows = lobar?.lobes
    ? Object.entries(lobar.lobes).map(([id, lobe]) => ({
        key: id,
        name: lobe.name_ru || id,
        cm3: lobe.cm3 ?? 0,
        percent: lobe.percent ?? 0,
      }))
    : [];

  const lobarColumns = [
    { title: 'Доля мозга', dataIndex: 'name', key: 'name', width: 200 },
    {
      title: 'Объём (см³)',
      dataIndex: 'cm3',
      key: 'cm3',
      align: 'right',
      render: (v) => (v ?? 0).toFixed(3),
      defaultSortOrder: 'descend',
      sorter: (a, b) => a.cm3 - b.cm3,
    },
    {
      title: '% поражения',
      dataIndex: 'percent',
      key: 'percent',
      align: 'right',
      render: (v) => (
        <Tag color={v > 30 ? 'red' : v > 10 ? 'orange' : 'blue'}>{(v ?? 0).toFixed(1)}%</Tag>
      ),
    },
  ];

  const lobarSection = lobar && lobarRows.length > 0 && (
    <>
      <Divider orientation="left" style={{ fontSize: 14 }}>
        <Space><EnvironmentOutlined /> Анатомическая локализация</Space>
      </Divider>
      <Row gutter={32} style={{ marginBottom: 16 }}>
        <Col>
          <Statistic
            title="Общий объём поражения"
            value={lobar.total_lesion_cm3 ?? 0}
            precision={3}
            suffix="см³"
          />
        </Col>
        <Col>
          <Statistic title="Затронуто долей" value={lobarRows.length} />
        </Col>
      </Row>
      <Table
        columns={lobarColumns}
        dataSource={lobarRows}
        pagination={false}
        size="small"
        bordered
        style={{ marginBottom: 24 }}
      />
    </>
  );

  // ===== МС путь =====
  if (lesionType === 'multiple_sclerosis') {
    if (!lesionStats && !lobar) {
      return <Empty description="Отчёт МС ещё не выгружен в Каппу для этой сессии" />;
    }
    return (
      <>
        {lesionStats && (
          <>
            <Divider orientation="left" style={{ fontSize: 14 }}>
              <Space><MedicineBoxOutlined /> Показатели МС</Space>
            </Divider>
            <Row gutter={32} style={{ marginBottom: 16 }}>
              <Col>
                <Statistic
                  title="Суммарный объём очагов"
                  value={lesionStats.total_volume_cm3 ?? 0}
                  precision={3}
                  suffix="см³"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col>
                <Statistic
                  title="Количество очагов"
                  value={lesionStats.lesion_count ?? 0}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col>
                <Statistic
                  title="Средний объём очага"
                  value={lesionStats.mean_lesion_volume_cm3 ?? 0}
                  precision={3}
                  suffix="см³"
                />
              </Col>
            </Row>
          </>
        )}
        {lobarSection}
      </>
    );
  }

  // ===== Глио путь =====
  const volumeClasses = volume?.classes
    ? Object.entries(volume.classes).map(([name, data]) => ({
        key: name,
        name,
        cm3: data.cm3 ?? 0,
        voxels: data.voxels ?? 0,
      }))
    : [];

  const volumeColumns = [
    { title: 'Класс', dataIndex: 'name', key: 'name', width: 120 },
    {
      title: 'Объём (см³)',
      dataIndex: 'cm3',
      key: 'cm3',
      align: 'right',
      render: (v) => (v ?? 0).toFixed(3),
    },
    {
      title: 'Воксели',
      dataIndex: 'voxels',
      key: 'voxels',
      align: 'right',
      render: (v) => (v ?? 0).toLocaleString(),
    },
  ];

  if (!volume && !lobar) {
    return <Empty description="Отчёт ещё не выгружен в Каппу для этой сессии" />;
  }

  return (
    <>
      {volume && volumeClasses.length > 0 && (
        <>
          <Divider orientation="left" style={{ fontSize: 14 }}>
            <Space><ExperimentOutlined /> Объёмы опухоли</Space>
          </Divider>
          <Row gutter={32} style={{ marginBottom: 16 }}>
            <Col>
              <Statistic
                title="Суммарный объём опухоли"
                value={volume.total_tumor_cm3 ?? 0}
                precision={3}
                suffix="см³"
                valueStyle={{ color: '#1890ff' }}
              />
            </Col>
          </Row>
          <Table
            columns={volumeColumns}
            dataSource={volumeClasses}
            pagination={false}
            size="small"
            bordered
            style={{ marginBottom: 24 }}
          />
        </>
      )}
      {lobarSection}
    </>
  );
};

export default KappaClinicalReport;
