/**
 * Компонент для 3D визуализации NIfTI файлов с niivue
 */
import { useEffect, useRef, useState } from 'react';
import { Modal, Select, Spin, Alert, Space, Button, Slider, Row, Col, Popover } from 'antd';
import { 
  EyeOutlined, 
  EyeInvisibleOutlined,
  RotateLeftOutlined,
  RotateRightOutlined,
  QuestionCircleOutlined 
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
    // Небольшая задержка чтобы canvas точно был готов
    const timer = setTimeout(() => {
      if (canvasRef.current && !nvRef.current) {
        try {
          console.log('Инициализация Niivue...');
          
          // Создаём экземпляр Niivue
          const nv = new Niivue({
            backColor: [0, 0, 0, 1],
            show3Dcrosshair: true,
            crosshairWidth: 1,
            logging: false,  // ← Убрать warnings
          });
          
          nvRef.current = nv;
          nv.attachToCanvas(canvasRef.current);
          
          console.log('Niivue инициализирован успешно');
        } catch (err) {
          console.error('Ошибка инициализации Niivue:', err);
          setError(`Ошибка инициализации 3D визуализации: ${err.message}`);
        }
      }
    }, 100);

    return () => clearTimeout(timer);
  }, [visible]);  // Зависимость от visible - переинициализируем при открытии

  /**
   * Загружаем список доступных файлов
   */
  const fetchFiles = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await getNIfTIFiles(runId);
      setFiles(data.files || []);
      
      // Ждём пока Niivue инициализируется
      const maxAttempts = 10;
      let attempts = 0;
      
      while (!nvRef.current && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 100));
        attempts++;
      }
      
      if (!nvRef.current) {
        setError('Не удалось инициализировать 3D визуализацию');
        return;
      }
      
      // Автоматически загружаем первый файл
      if (data.files && data.files.length > 0) {
        setSelectedFile(data.files[0]);
        await loadNIfTI(data.files[0]);
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
    console.log('=== loadNIfTI вызвана ===');
    console.log('nvRef.current:', nvRef.current);
    console.log('file:', file);
    
    if (!nvRef.current) {
      const errorMsg = 'Niivue не инициализирован';
      console.error(errorMsg);
      setError(errorMsg);
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const nv = nvRef.current;
      
      // Формируем полные URL
      const imageUrl = getNIfTIFileUrl(file.image_url);
      const maskUrl = getNIfTIFileUrl(file.mask_url);
      
      console.log('Загрузка изображения:', imageUrl);
      console.log('Загрузка маски:', maskUrl);
      
      // Сначала проверим что файлы доступны
      const imageResponse = await fetch(imageUrl);
      if (!imageResponse.ok) {
        throw new Error(`Не удалось загрузить изображение: ${imageResponse.status}`);
      }
      
      const maskResponse = await fetch(maskUrl);
      if (!maskResponse.ok) {
        throw new Error(`Не удалось загрузить маску: ${maskResponse.status}`);
      }
      
      console.log('Файлы доступны, загружаем в niivue...');
      
      // Загружаем в niivue
      await nv.loadVolumes([
        {
          url: imageUrl,
          colormap: 'gray',
          opacity: 1.0,
        },
        {
          url: maskUrl,
          colormap: 'red',
          opacity: maskOpacity,
        }
      ]);
      
      // Устанавливаем grid layout
      nv.setSliceType(nv.sliceTypeMultiplanar);
      nv.opts.multiplanarLayout = 2;

      // Настройки для лучшего отображения
      nv.opts.multiplanarPadPixels = 2;  // Отступ между срезами
      nv.opts.crosshairGap = 2;          // Зазор в кроссхейре

      // Устанавливаем размеры для каждой панели в grid
      nv.opts.multiplanarForceRender = true;
      nv.opts.isRadiologicalConvention = false;

      // Увеличиваем масштаб до максимума
      // volScaleMultiplier увеличивает размер всех volume
      nv.setScale(2.0);  // 2.0 = увеличение в 2 раза

      nv.drawScene();
      
      console.log('Файлы успешно загружены и отображены');
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
      width="95%"
      footer={null}
      style={{ top: 10 }}
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
          <div style={{ marginBottom: 8 }}>
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
            background: '#000',
            marginBottom: 8,
            width: '100%',    // ← Добавь
            height: '85vh'    // ← Добавь
          }}>
            <canvas
              ref={canvasRef}
              style={{ 
                width: '100%', 
                height: '100%',  // 100% от родительского div
                display: 'block'
              }}
            />
          </div>

          {/* Инструкция - показывается при наведении на иконку */}
          <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Popover
              content={
                <div style={{ maxWidth: 400 }}>
                  <strong>Управление:</strong>
                  <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20 }}>
                    <li>ЛКМ + перетаскивание - панорамирование</li>
                    <li>ПКМ + перетаскивание - изменение контрастности/яркости</li>
                    <li>Колесо мыши - навигация по срезам</li>
                    <li>Двойной клик - центрирование на точке</li>
                  </ul>
                </div>
              }
              title="Управление визуализацией"
              trigger="hover"
            >
              <Button 
                type="text" 
                icon={<QuestionCircleOutlined />}
                style={{ color: '#1890ff' }}
              >
                Как управлять?
              </Button>
            </Popover>
          </div>
        </>
      )}
    </Modal>
  );
};

export default NIfTIViewer;