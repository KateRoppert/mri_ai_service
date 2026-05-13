/**
 * Компонент для 3D визуализации NIfTI файлов с niivue
 */
import { useEffect, useRef, useState } from 'react';
import { Modal, Select, Spin, Alert, Space, Button, Slider, Row, Col, Popover, Radio, message } from 'antd';
import { 
  EyeOutlined, 
  EyeInvisibleOutlined,
  RotateLeftOutlined,
  RotateRightOutlined,
  QuestionCircleOutlined,
  EnvironmentOutlined 
} from '@ant-design/icons';
import { Niivue } from '@niivue/niivue';
import { getNIfTIFiles, getNIfTIFileUrl, getLobarAtlasUrl, getEntityRunInfo, getMaskFileUrl } from '../services/api';
import ValidationActions from './ValidationActions';
import ClinicalReportContent from './ClinicalReportContent';

/**
 * Создаём кастомную цветовую карту для multi-class сегментации
 * Label 0: прозрачный (фон)
 * Label 1: красный (NCR - некротическое ядро)
 * Label 2: зелёный (ED - отёк)
 * Label 3: жёлтый (NET - неусиливающаяся опухоль)
 * Label 4: синий (ET - усиливающаяся опухоль)
 */
const createSegmentationColormap = () => {
  // Создаём массивы RGBA для каждого label (0-4)
  const colors = {
    R: [0,   255, 0,   255, 0],      // 0: черный, 1: красный, 2: черный, 3: желтый, 4: черный
    G: [0,   0,   255, 255, 0],      // 0: черный, 1: черный, 2: зеленый, 3: желтый, 4: черный
    B: [0,   0,   0,   0,   255],    // 0: черный, 1: черный, 2: черный, 3: черный, 4: синий
    A: [0,   255, 255, 255, 255],    // 0: прозрачный, остальные непрозрачные
  };
  
  return colors;
};

const NIfTIViewer = ({ runId, visible, onClose, customFiles = null, validationRef = null }) => {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [maskOpacity, setMaskOpacity] = useState(0.5);
  const [showMask, setShowMask] = useState(true);
  const [showAtlas, setShowAtlas] = useState(false);
  const [viewMode, setViewMode] = useState('atlas'); // 'atlas' or 'native'
  const [resolvedRunId, setResolvedRunId] = useState(null);
  const [activeMaskLabel, setActiveMaskLabel] = useState(null); // для отображения выбранной версии

  /**
   * Резолвим runId: если передан напрямую — используем,
   * иначе запрашиваем по entity_id из patient_registry
   */
  useEffect(() => {
    if (runId) {
      setResolvedRunId(runId);
    } else if (visible && validationRef?.entity_id && !runId) {
      getEntityRunInfo(validationRef.entity_id)
        .then((info) => {
          setResolvedRunId(info.run_id || null);
        })
        .catch((err) => {
          console.warn('Не удалось получить run_id по entity_id:', err);
          setResolvedRunId(null);
        });
    }
  }, [runId, visible, validationRef]);


  /**
   * Загружаем список файлов при открытии модального окна
   */
  useEffect(() => {
    if (visible && (runId || customFiles)) {
      fetchFiles();
    }
  }, [visible, runId, customFiles]);

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
      // Если customFiles переданы, используем их вместо fetch
      let filesData;
      if (customFiles && customFiles.length > 0) {
        filesData = customFiles;
      } else {
        const data = await getNIfTIFiles(runId);
        filesData = data.files || [];
      }
      setFiles(filesData);
      
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
      if (filesData.length > 0) {
        setSelectedFile(filesData[0]);
        await loadNIfTI(filesData[0]);
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
  const loadNIfTI = async (file, mode = viewMode) => {
    setShowAtlas(false);
    if (!nvRef.current) {
      setError('Niivue не инициализирован');
      return;
    }

    // Определяем URL в зависимости от режима
    let imageUrlPath, maskUrlPath;

    if (mode === 'native' && file.native_image_url && file.native_mask_url) {
      imageUrlPath = file.native_image_url;
      maskUrlPath = file.native_mask_url;
    } else {
      imageUrlPath = file.image_url;
      maskUrlPath = file.mask_url;
    }

    setLoading(true);
    setError(null);
    
    try {
      const nv = nvRef.current;
      
      // Если URL уже абсолютный (http/https), используем как есть
      const imageUrl = imageUrlPath.startsWith('http')
        ? imageUrlPath
        : getNIfTIFileUrl(imageUrlPath);
      const maskUrl = maskUrlPath.startsWith('http')
        ? maskUrlPath
        : getNIfTIFileUrl(maskUrlPath);
      
      // Проверяем доступность файлов
      const imageResponse = await fetch(imageUrl);
      if (!imageResponse.ok) {
        throw new Error(`Не удалось загрузить изображение: ${imageResponse.status}`);
      }
      
      const maskResponse = await fetch(maskUrl);
      if (!maskResponse.ok) {
        throw new Error(`Не удалось загрузить маску: ${maskResponse.status}`);
      }
      
      // Colormap
      const segColormap = createSegmentationColormap();
      nv.addColormap('seg_custom', segColormap);

      // Загружаем в niivue. Передаём name для определения формата,
      // т.к. URL из прокси не содержит расширения .nii.gz
      await nv.loadVolumes([
        {
          url: imageUrl,
          name: file.filename,
          colormap: 'gray',
          opacity: 1.0,
        },
        {
          url: maskUrl,
          name: file.mask_filename,
          colormap: 'seg_custom',
          opacity: maskOpacity,
          cal_min: 0,
          cal_max: 4,
        }
      ]);
      
      // Настройки отображения
      nv.setSliceType(nv.sliceTypeMultiplanar);
      nv.opts.multiplanarLayout = 2;
      nv.opts.multiplanarPadPixels = 2;
      nv.opts.crosshairGap = 2;
      nv.opts.multiplanarForceRender = true;
      nv.opts.isRadiologicalConvention = false;
      nv.setScale(2.0);
      nv.drawScene();
      
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
      setActiveMaskLabel(null);
      loadNIfTI(file);
    }
  };

  /**
   * Переключение между atlas и native пространством
   */
  const handleViewModeChange = (e) => {
    const mode = e.target.value;
    setViewMode(mode);
    setActiveMaskLabel(null);
    if (selectedFile) {
      loadNIfTI(selectedFile, mode);
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
   * Переключение отображения лобарного атласа
   */
  const toggleAtlas = async () => {
    if (!nvRef.current) return;
    const nv = nvRef.current;

    if (showAtlas) {
      // Убираем атлас (третий volume)
      if (nv.volumes && nv.volumes.length > 2) {
        nv.removeVolumeByIndex(2);
        nv.drawScene();
      }
      setShowAtlas(false);
    } else {
      // Добавляем атлас как третий слой
      try {
        const atlasUrl = getLobarAtlasUrl(runId);
        await nv.addVolumeFromUrl({
          url: atlasUrl,
          colormap: 'freesurfer',
          opacity: 0.3,
          cal_min: 0.5,
          cal_max: 48,
        });
        nv.drawScene();
        setShowAtlas(true);
      } catch (err) {
        console.error('Ошибка загрузки лобарного атласа:', err);
      }
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

  /**
   * Загрузить конкретную версию маски (по URL из mask_service)
   * Заменяет текущий overlay (volume index 1) на новую маску.
   */
  const loadMaskVersion = async (versionInfo) => {
    if (!nvRef.current || !versionInfo?.maskUrl) return;

    const nv = nvRef.current;
    setLoading(true);
    setError(null);

    try {
      // Убираем атлас если показан
      if (showAtlas && nv.volumes && nv.volumes.length > 2) {
        nv.removeVolumeByIndex(2);
        setShowAtlas(false);
      }

      // Удаляем текущую маску (index 1)
      if (nv.volumes && nv.volumes.length >= 2) {
        nv.removeVolumeByIndex(1);
      }

      // Проверяем доступность нового файла маски
      const maskResponse = await fetch(versionInfo.maskUrl);
      if (!maskResponse.ok) {
        throw new Error(`Не удалось загрузить маску v${versionInfo.version}: ${maskResponse.status}`);
      }

      // Создаём colormap
      const segColormap = createSegmentationColormap();
      nv.addColormap('seg_custom', segColormap);

      // Добавляем новую маску
      await nv.addVolumeFromUrl({
        url: versionInfo.maskUrl,
        name: versionInfo.file_name || `mask_v${versionInfo.version}.nii.gz`,
        colormap: 'seg_custom',
        opacity: maskOpacity,
        cal_min: 0,
        cal_max: 4,
      });

      nv.drawScene();
      setShowMask(true);

      // Верификация: читаем статистику загруженной маски
      let statsLabel = '';
      if (nv.volumes && nv.volumes.length >= 2) {
        const maskVol = nv.volumes[nv.volumes.length - 1];
        if (maskVol && maskVol.img) {
          const data = maskVol.img;
          let nonZero = 0;
          const classCounts = {};
          for (let i = 0; i < data.length; i++) {
            if (data[i] > 0) {
              nonZero++;
              classCounts[data[i]] = (classCounts[data[i]] || 0) + 1;
            }
          }
          const classStr = Object.entries(classCounts)
            .map(([k, v]) => `c${k}:${v}`)
            .join(' ');
          statsLabel = ` | ${nonZero} вокс. [${classStr}]`;
        }
      }

      setActiveMaskLabel(`v${versionInfo.version} (${versionInfo.source === 'ai' ? 'ИИ' : 'Эксперт'})${statsLabel}`);
      message.success(`Маска переключена на версию ${versionInfo.version}`);
    } catch (err) {
      console.error('Ошибка загрузки версии маски:', err);
      setError(`Ошибка загрузки маски: ${err.message}`);
    } finally {
      setLoading(false);
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
              <Col span={6}>
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

              {/* Переключатель пространства */}
              <Col span={4}>
                <Space direction="vertical" style={{ width: '100%' }} size="small">
                  <span style={{ fontSize: 12, color: '#999' }}>Пространство:</span>
                  <Radio.Group
                    value={viewMode}
                    onChange={handleViewModeChange}
                    size="small"
                    optionType="button"
                    buttonStyle="solid"
                  >
                    <Radio.Button value="atlas">Атлас</Radio.Button>
                    <Radio.Button
                      value="native"
                      disabled={!selectedFile?.native_mask_url}
                    >
                      Нативное
                    </Radio.Button>
                  </Radio.Group>
                </Space>
              </Col>

              {/* Управление маской */}
              <Col span={5}>
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
              <Col span={9}>
                <Space>
                  <Button
                    icon={showMask ? <EyeInvisibleOutlined /> : <EyeOutlined />}
                    onClick={toggleMask}
                  >
                    {showMask ? 'Скрыть' : 'Показать'}
                  </Button>
                  {activeMaskLabel && (
                    <span style={{ fontSize: 12, color: '#1890ff', fontWeight: 500 }}>
                      Маска: {activeMaskLabel}
                    </span>
                  )}
                  <Button
                    icon={<RotateLeftOutlined />}
                    onClick={resetView}
                  >
                    Сброс
                  </Button>
                  <Button
                    icon={<EnvironmentOutlined />}
                    onClick={toggleAtlas}
                    type={showAtlas ? 'primary' : 'default'}
                  >
                    {showAtlas ? 'Скрыть доли' : 'Показать доли'}
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

          {/* Панель валидации */}
          {validationRef && (
            <div style={{
              display: 'flex',
              justifyContent: 'flex-end',
              padding: '8px 0',
              borderTop: '1px solid #f0f0f0',
              marginTop: 8,
            }}>
              <ValidationActions
                entityId={validationRef.entity_id}
                datasetId={validationRef.dataset_id}
                runId={resolvedRunId}
                onCloseViewer={() => {
                  // Уничтожаем Niivue instance для освобождения HTTP-соединений
                  if (nvRef.current) {
                    try {
                      nvRef.current.closeDrawing();
                      nvRef.current = null;
                    } catch (e) {
                      console.warn('Niivue cleanup error:', e);
                      nvRef.current = null;
                    }
                  }
                  onClose();
                }}
                onMaskUploaded={(result) => {
                  // Автообновление: перезагружаем текущий файл чтобы показать новую маску
                  if (selectedFile) {
                    loadNIfTI(selectedFile);
                  }
                  setActiveMaskLabel(null);
                }}
                onMaskVersionSelect={loadMaskVersion}
              />
            </div>
          )}

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
            {/* Легенда цветов сегментации */}
            <Space size="large" style={{ fontSize: 13 }}>
              <Space size="small">
                <div style={{ 
                  width: 16, 
                  height: 16, 
                  background: 'rgb(255, 0, 0)', 
                  border: '1px solid #ccc',
                  borderRadius: 2
                }} />
                <span>Некротическое ядро (NCR)</span>
              </Space>
              <Space size="small">
                <div style={{ 
                  width: 16, 
                  height: 16, 
                  background: 'rgb(0, 255, 0)', 
                  border: '1px solid #ccc',
                  borderRadius: 2
                }} />
                <span>Отёк (ED)</span>
              </Space>
              <Space size="small">
                <div style={{ 
                  width: 16, 
                  height: 16, 
                  background: 'rgb(255, 255, 0)', 
                  border: '1px solid #ccc',
                  borderRadius: 2
                }} />
                <span>Неусиливающаяся опухоль (NET)</span>
              </Space>
              <Space size="small">
                <div style={{ 
                  width: 16, 
                  height: 16, 
                  background: 'rgb(0, 0, 255)', 
                  border: '1px solid #ccc',
                  borderRadius: 2
                }} />
                <span>Усиливающаяся опухоль (ET)</span>
              </Space>
            </Space>
          </div>

          {/* Клинический отчёт — встроен под визуализацией */}
          {resolvedRunId && (
            <div style={{
              marginTop: 24,
              padding: '16px 0',
              borderTop: '2px solid #f0f0f0',
            }}>
              <ClinicalReportContent runId={resolvedRunId} autoLoad={true} />
            </div>
          )}
        </>
      )}
    </Modal>
  );
};

export default NIfTIViewer;