/**
 * Клинический отчёт — модальное окно.
 * Оборачивает ClinicalReportContent в Modal. Для прогонов с несколькими
 * пациентами показывает выпадающий список для выбора — без него отчёты
 * всех пациентов рендерились одним полотном подряд.
 */
import { useCallback, useState } from 'react';
import { Modal, Space, Select } from 'antd';
import { MedicineBoxOutlined } from '@ant-design/icons';
import ClinicalReportContent from './ClinicalReportContent';

const ClinicalReport = ({ runId, visible, onClose, lesionType = 'glioblastoma' }) => {
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(null);

  // No separate reset-on-runId-change effect needed here: ClinicalReportContent
  // already clears its report state when runId changes, which flows back
  // through onPatientsChange([]) below and resets patients/selectedPatientId
  // as a natural consequence — one less synchronous setState-in-effect.

  // Stable identity: this is a dependency of an effect inside
  // ClinicalReportContent — a fresh function reference every render would
  // re-fire that effect every render right back, in a loop.
  const handlePatientsChange = useCallback((list) => {
    setPatients(list);
    setSelectedPatientId((prev) =>
      prev && list.some((p) => p.patient_id === prev) ? prev : (list[0]?.patient_id ?? null)
    );
  }, []);

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
      {patients.length > 1 && (
        <Select
          value={selectedPatientId}
          onChange={setSelectedPatientId}
          style={{ width: 280, marginBottom: 16 }}
          options={patients.map((p) => ({
            value: p.patient_id,
            label: p.original_id ? `${p.patient_id} (${p.original_id})` : p.patient_id,
          }))}
        />
      )}
      <ClinicalReportContent
        runId={runId}
        autoLoad={visible}
        lesionType={lesionType}
        selectedPatientId={selectedPatientId}
        onPatientsChange={handlePatientsChange}
      />
    </Modal>
  );
};

export default ClinicalReport;