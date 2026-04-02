/**
 * Форма для запуска pipeline
 */
import { useState, useEffect } from 'react';
import { Form, Input, Button, Card, message, Space, Checkbox, Select } from 'antd';
import { PlayCircleOutlined, FolderOpenOutlined, MedicineBoxOutlined } from '@ant-design/icons';
import { startPipeline, getLesionTypes } from '../services/api';

const PipelineForm = ({ onPipelineStarted }) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [useDefault, setUseDefault] = useState(false);
  const [lesionTypes, setLesionTypes] = useState([]);

  useEffect(() => {
    const fetchLesionTypes = async () => {
      try {
        const types = await getLesionTypes();
        setLesionTypes(types);
        // Если есть хотя бы один тип — выставляем по умолчанию
        if (types.length > 0) {
          form.setFieldsValue({ lesionType: types[0].id });
        }
      } catch (err) {
        console.error('Ошибка загрузки типов поражений:', err);
      }
    };
    fetchLesionTypes();
  }, [form]);

  const handleSubmit = async (values) => {
    setLoading(true);

    try {
      const response = await startPipeline(
        values.inputPath,
        useDefault ? null : values.outputPath,
        useDefault,
        values.lesionType
      );

      message.success('Pipeline запущен успешно!');

      if (onPipelineStarted) {
        onPipelineStarted(response);
      }

      form.resetFields();

    } catch (error) {
      console.error('Ошибка запуска pipeline:', error);
      message.error(
        error.response?.data?.detail || 'Ошибка запуска pipeline. Проверьте пути и попробуйте снова.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card
      title={
        <Space>
          <PlayCircleOutlined />
          <span>Запуск обработки данных</span>
        </Space>
      }
      style={{ marginBottom: 24 }}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          inputPath: '',
          outputPath: '',
          lesionType: '',
        }}
      >
        {/* Тип поражения */}
        <Form.Item
          label="Тип поражения"
          name="lesionType"
          rules={[{ required: true, message: 'Выберите тип поражения' }]}
          tooltip="Определяет модель сегментации и целевой датасет в Каппе"
        >
          <Select
            placeholder="Выберите тип поражения"
            size="large"
            suffixIcon={<MedicineBoxOutlined />}
          >
            {lesionTypes.map((lt) => (
              <Select.Option key={lt.id} value={lt.id}>
                {lt.name}
              </Select.Option>
            ))}
          </Select>
        </Form.Item>

        {/* Входная директория */}
        <Form.Item
          label="Входные данные (DICOM)"
          name="inputPath"
          rules={[
            { required: true, message: 'Укажите путь к директории с DICOM данными' },
          ]}
          tooltip="Путь к директории с DICOM файлами пациента"
        >
          <Input
            prefix={<FolderOpenOutlined />}
            placeholder="/path/to/dicom/data"
            size="large"
          />
        </Form.Item>

        {/* Чекбокс "Использовать путь по умолчанию" */}
        <Form.Item>
          <Checkbox
            checked={useDefault}
            onChange={(e) => setUseDefault(e.target.checked)}
          >
            Использовать путь по умолчанию для результатов
          </Checkbox>
        </Form.Item>

        {/* Выходная директория */}
        {!useDefault && (
          <Form.Item
            label="Директория для результатов"
            name="outputPath"
            rules={[
              { required: !useDefault, message: 'Укажите путь для сохранения результатов' },
            ]}
            tooltip="Путь для сохранения результатов обработки"
          >
            <Input
              prefix={<FolderOpenOutlined />}
              placeholder="/path/to/results"
              size="large"
            />
          </Form.Item>
        )}

        {/* Кнопка запуска */}
        <Form.Item>
          <Button
            type="primary"
            htmlType="submit"
            size="large"
            icon={<PlayCircleOutlined />}
            loading={loading}
            block
          >
            {loading ? 'Запуск...' : 'Запустить обработку'}
          </Button>
        </Form.Item>
      </Form>
    </Card>
  );
};

export default PipelineForm;