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

// Binary colormap for MS: 0=transparent background, 1=green lesion
const createMsColormap = () => ({
  R: [0, 82],
  G: [0, 196],
  B: [0, 26],
  A: [0, 255],
});

// Default niivue render scale — 3D volume tile kept in proportion with slices.
const DEFAULT_VIEW_SCALE = 1.5;

// Canonical modality order for display/sorting (not alphabetical).
const MODALITY_ORDER = { T1: 0, T1C: 1, T2: 2, T2FL: 3 };

// Sort files by patient, then session (chronological — ses-001 is earliest),
// then canonical modality order. Used so viewers and reports are consistent.
const sortNiftiFiles = (arr) =>
  [...arr].sort((a, b) => {
    const p = (a.patient_id || '').localeCompare(b.patient_id || '');
    if (p !== 0) return p;
    const s = (a.session_id || '').localeCompare(b.session_id || '');
    if (s !== 0) return s;
    return (MODALITY_ORDER[a.modality] ?? 99) - (MODALITY_ORDER[b.modality] ?? 99);
  });

// Build a full, unambiguous label: patient / session / modality.
// Skips empty parts so validation (session folded into patient_id) stays clean.
const fileLabel = (f) =>
  [f.patient_id, f.session_id, f.modality].filter(Boolean).join(' / ');

const NIfTIViewer = ({ runId, visible, onClose, customFiles = null, validationRef = null, lesionType = 'glioblastoma', kappaReport = null, onValidationRefChange = null }) => {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  // Ref to the currently selected file — niivue callbacks close over the init
  // render, so they must read the latest value via a ref rather than state.
  const selectedFileRef = useRef(null);
  // True only while the displayed mask is the per-lesion LABELED mask (MS, atlas
  // space). Hover lookup is valid only then — the native/binary mask has no labels.
  const labeledMaskActiveRef = useRef(false);

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
  // Tooltip: per-lesion volume shown when cursor hovers a labeled voxel (MS only)
  const [hoverVolume, setHoverVolume] = useState(null); // { cm3: number } | null

  // Keep the ref in sync so niivue's onLocationChange reads the latest file.
  useEffect(() => { selectedFileRef.current = selectedFile; }, [selectedFile]);

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

          // Per-lesion hover tooltip: read the labeled voxel under the cursor
          // and look up its volume in the per-session map.
          nv.onLocationChange = (location) => {
            const file = selectedFileRef.current;
            const byLabel = file?.lesion_volumes_by_label;
            // Only valid when the labeled mask is the one on screen.
            if (!labeledMaskActiveRef.current || !byLabel) {
              setHoverVolume(null);
              return;
            }
            // Read the MASK volume's value by name (robust to an atlas overlay
            // being added as a third volume, which would shift "last").
            const values = location?.values || [];
            const maskVal = values.find((v) => v.name === file.mask_filename);
            const labelVal = maskVal ? Math.round(maskVal.value) : 0;
            if (labelVal > 0 && byLabel[String(labelVal)] != null) {
              setHoverVolume({ cm3: byLabel[String(labelVal)] });
            } else {
              setHoverVolume(null);
            }
          };

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
      filesData = sortNiftiFiles(filesData);
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
      
      // Colormap: binary green for MS, multi-class for GBM
      const segColormap = lesionType === 'multiple_sclerosis'
        ? createMsColormap()
        : createSegmentationColormap();
      nv.addColormap('seg_custom', segColormap);

      // For MS in ATLAS space, display the labeled mask (each voxel = lesion
      // instance label) so the hover handler can look up per-lesion volumes.
      // The labeled mask exists only in atlas space — in native space it would
      // be misaligned, so there we keep the (native) binary mask and disable
      // hover. Visual output is unchanged because cal_max:1 clamps all labels
      // ≥1 to the single green colormap entry. GBM always uses the binary mask.
      const useLabeled =
        lesionType === 'multiple_sclerosis' &&
        !!file.mask_labels_url &&
        mode !== 'native';
      labeledMaskActiveRef.current = useLabeled;
      const maskVolumeUrl = useLabeled
        ? (file.mask_labels_url.startsWith('http')
            ? file.mask_labels_url
            : getNIfTIFileUrl(file.mask_labels_url))
        : maskUrl;

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
          url: maskVolumeUrl,
          name: file.mask_filename,
          colormap: 'seg_custom',
          opacity: maskOpacity,
          cal_min: 0,
          cal_max: lesionType === 'multiple_sclerosis' ? 1 : 4,
        }
      ]);
      
      // Настройки отображения
      nv.setSliceType(nv.sliceTypeMultiplanar);
      nv.opts.multiplanarLayout = 2;
      nv.opts.multiplanarPadPixels = 2;
      nv.opts.crosshairGap = 2;
      nv.opts.multiplanarForceRender = true;
      nv.opts.isRadiologicalConvention = false;
      // 1.5 keeps the 3D render in proportion with the slice tiles (2.0 was
      // too large, 1.0 too small).
      nv.setScale(DEFAULT_VIEW_SCALE);
      nv.drawScene();
      
    } catch (err) {
      console.error('Ошибка загрузки NIfTI:', err);
      setError(`Не удалось загрузить изображение: ${err.message}`);
    } finally {
      setLoading(false);
    }
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
      // For multi-patient runs the validationRef (entity_id for Slicer) must
      // track the selected patient. If the parent provided a callback and the
      // validationRef carries all_entities, resolve the correct entity.
      if (onValidationRefChange && validationRef?.all_entities) {
        const subject = file.patient_id; // e.g. "sub-003"
        const match = validationRef.all_entities.find(
          (e) => e.bids_id && (e.bids_id === subject || e.bids_id.startsWith(subject + '_'))
        );
        if (match) {
          onValidationRefChange({ entity_id: match.entity_id, dataset_id: match.dataset_id });
        }
      }
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
        // В режиме валидации runId не передаётся напрямую — берём resolvedRunId
        const atlasUrl = getLobarAtlasUrl(runId || resolvedRunId);
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
    const nv = nvRef.current;
    if (!nv) return;
    // Full reset to the initial state: layout, zoom, and the slice/crosshair
    // position. The user may have scrolled slices (wheel) or clicked to move
    // the crosshair — recentre it (slices follow the crosshair).
    nv.setSliceType(nv.sliceTypeMultiplanar);
    nv.setScale(DEFAULT_VIEW_SCALE);
    if (nv.scene) {
      nv.scene.crosshairPos = [0.5, 0.5, 0.5]; // fractional volume centre
    }
    nv.drawScene();
  };

  /**
   * Загрузить конкретную версию маски (по URL из mask_service)
   * Заменяет текущий overlay (volume index 1) на новую маску.
   */
  const loadMaskVersion = async (versionInfo) => {
    if (!nvRef.current || !versionInfo?.maskUrl) return;

    const nv = nvRef.current;
    // Versioned masks are binary (no per-lesion label map) — disable hover lookup
    // and drop any tooltip so it can't show a stale/misleading per-lesion volume.
    labeledMaskActiveRef.current = false;
    setHoverVolume(null);
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

      // Создаём colormap: binary green for MS, multi-class for GBM
      const segColormap = lesionType === 'multiple_sclerosis'
        ? createMsColormap()
        : createSegmentationColormap();
      nv.addColormap('seg_custom', segColormap);

      // Добавляем новую маску
      await nv.addVolumeFromUrl({
        url: versionInfo.maskUrl,
        name: versionInfo.file_name || `mask_v${versionInfo.version}.nii.gz`,
        colormap: 'seg_custom',
        opacity: maskOpacity,
        cal_min: 0,
        cal_max: lesionType === 'multiple_sclerosis' ? 1 : 4,
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
                      label: fileLabel(f),
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
            position: 'relative',  // needed so the hover tooltip badge is positioned inside
            border: '1px solid #d9d9d9',
            borderRadius: 4,
            overflow: 'hidden',
            background: '#000',
            marginBottom: 8,
            width: '100%',
            height: '85vh',
          }}>
            <canvas
              ref={canvasRef}
              style={{
                width: '100%',
                height: '100%',  // 100% от родительского div
                display: 'block'
              }}
            />
            {/* Per-lesion volume tooltip — appears when cursor hovers an MS lesion */}
            {hoverVolume && (
              <div style={{
                position: 'absolute', top: 8, right: 8, zIndex: 5,
                background: 'rgba(0,0,0,0.75)', color: '#fff',
                padding: '4px 10px', borderRadius: 4, fontSize: 13,
                pointerEvents: 'none',
              }}>
                Очаг: {hoverVolume.cm3.toFixed(3)} см³
              </div>
            )}
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
            {/* Легенда цветов сегментации — зависит от типа поражения */}
            <Space size="large" style={{ fontSize: 13 }}>
              {(lesionType === 'multiple_sclerosis'
                ? [
                    { color: 'rgb(82, 196, 26)', label: 'Очаги РС' },
                  ]
                : [
                    { color: 'rgb(255, 0, 0)', label: 'Некротическое ядро (NCR)' },
                    { color: 'rgb(0, 255, 0)', label: 'Отёк (ED)' },
                    { color: 'rgb(255, 255, 0)', label: 'Неусиливающаяся опухоль (NET)' },
                    { color: 'rgb(0, 0, 255)', label: 'Усиливающаяся опухоль (ET)' },
                  ]
              ).map((item) => (
                <Space size="small" key={item.label}>
                  <div style={{
                    width: 16,
                    height: 16,
                    background: item.color,
                    border: '1px solid #ccc',
                    borderRadius: 2,
                  }} />
                  <span>{item.label}</span>
                </Space>
              ))}
            </Space>
          </div>

          {/* Клинический отчёт — встроен под визуализацией. Один компонент,
              один стиль. В валидации источник — Каппа (kappaReport →
              kappaEntityInfo), в запуске/истории — локальные файлы по runId. */}
          {(kappaReport || resolvedRunId) && (
            <div style={{
              marginTop: 24,
              padding: '16px 0',
              borderTop: '2px solid #f0f0f0',
            }}>
              <ClinicalReportContent
                runId={resolvedRunId}
                autoLoad={true}
                lesionType={lesionType}
                kappaEntityInfo={kappaReport}
                selectedPatientId={kappaReport ? undefined : selectedFile?.patient_id}
              />
            </div>
          )}
        </>
      )}
    </Modal>
  );
};

export default NIfTIViewer;