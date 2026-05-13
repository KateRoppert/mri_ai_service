/**
 * Клинический отчёт — модальное окно.
 * Оборачивает ClinicalReportContent в Modal.
 */
import { Modal, Space } from 'antd';
import { MedicineBoxOutlined } from '@ant-design/icons';
import ClinicalReportContent from './ClinicalReportContent';

const ClinicalReport = ({ runId, visible, onClose }) => {
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
      <ClinicalReportContent runId={runId} autoLoad={visible} />
    </Modal>
  );
};

export default ClinicalReport;