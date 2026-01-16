/**
 * Форма для запуска pipeline
 */
import { useState } from 'react';
import { Form, Input, Button, Card, message, Space, Checkbox } from 'antd';
import { PlayCircleOutlined, FolderOpenOutlined } from '@ant-design/icons';
import { startPipeline } from '../services/api';

const PipelineForm = ({ onPipelineStarted }) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [useDefault, setUseDefault] = useState(false);

  /**
   * Обработчик отправки формы
   */
  const handleSubmit = async (values) => {
    setLoading(true);
    
    try {
      // Вызываем API для запуска pipeline
      const response = await startPipeline(
        values.inputPath,
        useDefault ? null : values.outputPath,
        useDefault
      );
      
      message.success('Pipeline запущен успешно!');
      
      // Уведомляем родительский компонент
      if (onPipelineStarted) {
        onPipelineStarted(response);
      }
      
      // Очищаем форму
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
        }}
      >
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

        {/* Выходная директория (показываем только если не useDefault) */}
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