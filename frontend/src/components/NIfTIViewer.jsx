/**
 * Компонент для 3D визуализации NIfTI файлов с niivue
 */
import { useEffect, useRef, useState } from 'react';
import { Modal, Select, Spin, Alert, Space, Button, Slider, Row, Col } from 'antd';
import { 
  EyeOutlined, 
  EyeInvisibleOutlined,
  RotateLeftOutlined,
  RotateRightOutlined 
} from '@ant-design/icons';
import { Niivue } from '@niivue/niivue';
import { getNIfTIFiles, getNIfTIFileUrl } from '../services/api';

const NIfTIViewer = ({ runId, visible, onClose }) => {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [maskOpacity, setMaskOpacity] = useState(0.5);
  const [showMask, setShowMask] = useState(true);

  /**
   * Загружаем список файлов при открытии модального окна
   */
  useEffect(() => {
    if (visible && runId) {
      fetchFiles();
    }
  }, [visible, runId]);

  /**
   * Инициализируем niivue при монтировании
   */
  useEffect(() => {
    if (canvasRef.current && !nvRef.current) {
      try {
        // Создаём экземпляр Niivue
        const nv = new Niivue({
          backColor: [0, 0, 0, 1],  // Чёрный фон
          show3Dcrosshair: true,     // Показывать кроссхейр
          crosshairWidth: 1,         // Тонкий кроссхейр
          multiplanarLayout: 2,      // 2 = Grid (квадрат 2x2)
          onLocationChange: handleLocationChange,
        });
        
        nvRef.current = nv;
        nv.attachToCanvas(canvasRef.current);
        
        // Устанавливаем мультипланарный режим
        nv.setSliceType(nv.sliceTypeMultiplanar);
        
        console.log('Niivue инициализирован успешно');
      } catch (err) {
        console.error('Ошибка инициализации Niivue:', err);
        setError(`Ошибка инициализации 3D визуализации: ${err.message}`);
      }
    }
  }, [canvasRef.current]);

  /**
   * Загружаем список доступных файлов
   */
  const fetchFiles = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await getNIfTIFiles(runId);
      setFiles(data.files || []);
      
      // Автоматически загружаем первый файл
      if (data.files && data.files.length > 0) {
        loadNIfTI(data.files[0]);
        setSelectedFile(data.files[0]);
      }
    } catch (err) {
      console.error('Ошибка загрузки списка файлов:', err);
      setError('Не удалось загрузить список файлов для визуализации');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Загружаем NIfTI файлы в niivue
   */
  const loadNIfTI = async (file) => {
    if (!nvRef.current) {
      console.error('Niivue не инициализирован');
      return;
    }

    setLoading(true);
    setError(null);  // Сброс предыдущей ошибки
    
    try {
      const nv = nvRef.current;
      
      // Формируем полные URL
      const imageUrl = getNIfTIFileUrl(file.image_url);
      const maskUrl = getNIfTIFileUrl(file.mask_url);
      
      console.log('Загрузка изображения:', imageUrl);
      console.log('Загрузка маски:', maskUrl);
      
      // Загружаем основное изображение и маску
      await nv.loadVolumes([
        {
          url: imageUrl,
          colormap: 'gray',  // Чёрно-белое изображение
          opacity: 1.0,
        },
        {
          url: maskUrl,
          colormap: 'red',   // Маска красным цветом
          opacity: maskOpacity,
        }
      ]);
      
      // Устанавливаем мультипланарный режим после загрузки
      nv.setSliceType(nv.sliceTypeMultiplanar);
      
      console.log('Файлы успешно загружены');
    } catch (err) {
      console.error('Ошибка загрузки NIfTI:', err);
      setError(`Не удалось загрузить изображение: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Обработчик изменения позиции курсора
   */
  const handleLocationChange = (data) => {
    // Можно добавить отображение координат
    console.log('Позиция:', data);
  };

  /**
   * Изменение выбранного файла
   */
  const handleFileChange = (value) => {
    const file = files.find(f => f.filename === value);
    if (file) {
      setSelectedFile(file);
      loadNIfTI(file);
    }
  };

  /**
   * Переключение видимости маски
   */
  const toggleMask = () => {
    if (!nvRef.current || !nvRef.current.volumes || nvRef.current.volumes.length < 2) {
      return;
    }
    
    const newOpacity = showMask ? 0 : maskOpacity;
    nvRef.current.setOpacity(1, newOpacity);  // Индекс 1 = маска
    setShowMask(!showMask);
  };

  /**
   * Изменение прозрачности маски
   */
  const handleOpacityChange = (value) => {
    setMaskOpacity(value);
    
    if (nvRef.current && nvRef.current.volumes && nvRef.current.volumes.length >= 2 && showMask) {
      nvRef.current.setOpacity(1, value);
    }
  };

  /**
   * Сброс вида
   */
  const resetView = () => {
    if (nvRef.current) {
      nvRef.current.setSliceType(nvRef.current.sliceTypeMultiplanar);
    }
  };

  return (
    <Modal
      title={
        <Space>
          <EyeOutlined />
          <span>3D Визуализация результатов сегментации</span>
        </Space>
      }
      open={visible}
      onCancel={onClose}
      width={1400}
      footer={null}
      style={{ top: 20 }}
    >
      {loading && files.length === 0 && (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
          <p style={{ marginTop: 16 }}>Загрузка визуализации...</p>
        </div>
      )}

      {error && (
        <Alert
          title="Ошибка"
          description={error}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {files.length > 0 && (
        <>
          {/* Панель управления */}
          <div style={{ marginBottom: 16 }}>
            <Row gutter={16} align="middle">
              {/* Выбор файла */}
              <Col span={10}>
                <Space direction="vertical" style={{ width: '100%' }} size="small">
                  <span style={{ fontSize: 12, color: '#999' }}>Выберите файл:</span>
                  <Select
                    value={selectedFile?.filename}
                    onChange={handleFileChange}
                    style={{ width: '100%' }}
                    options={files.map(f => ({
                      label: `${f.patient_id} - ${f.modality}`,
                      value: f.filename,
                    }))}
                  />
                </Space>
              </Col>

              {/* Управление маской */}
              <Col span={8}>
                <Space direction="vertical" style={{ width: '100%' }} size="small">
                  <span style={{ fontSize: 12, color: '#999' }}>Прозрачность маски:</span>
                  <Slider
                    min={0}
                    max={1}
                    step={0.1}
                    value={maskOpacity}
                    onChange={handleOpacityChange}
                    disabled={!showMask}
                  />
                </Space>
              </Col>

              {/* Кнопки */}
              <Col span={6}>
                <Space>
                  <Button
                    icon={showMask ? <EyeInvisibleOutlined /> : <EyeOutlined />}
                    onClick={toggleMask}
                  >
                    {showMask ? 'Скрыть' : 'Показать'}
                  </Button>
                  <Button
                    icon={<RotateLeftOutlined />}
                    onClick={resetView}
                  >
                    Сброс
                  </Button>
                </Space>
              </Col>
            </Row>
          </div>

          {/* Canvas для niivue */}
          <div style={{ 
            border: '1px solid #d9d9d9', 
            borderRadius: 4,
            overflow: 'hidden',
            background: '#000'
          }}>
            <canvas
              ref={canvasRef}
              style={{ 
                width: '100%', 
                height: '700px',
                display: 'block'
              }}
            />
          </div>

          {/* Инструкция */}
          <Alert
            type="info"
            showIcon
            style={{ marginTop: 16 }}
            description={
              <div>
                <strong>Управление:</strong>
                <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20 }}>
                  <li>ЛКМ + перетаскивание - панорамирование</li>
                  <li>ПКМ + перетаскивание - изменение контрастности/яркости</li>
                  <li>Колесо мыши - навигация по срезам</li>
                  <li>Двойной клик - центрирование на точке</li>
                </ul>
              </div>
            }
          />
        </>
      )}
    </Modal>
  );
};

export default NIfTIViewer;