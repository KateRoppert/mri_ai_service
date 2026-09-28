/**
 * Главный компонент приложения
 */
import { useState, useEffect } from 'react';
import { Layout, Typography, Space, Divider, Tabs, Card, Button, Alert, Modal, message } from 'antd';
import { RocketOutlined, HistoryOutlined, LogoutOutlined, CheckCircleOutlined } from '@ant-design/icons';
import KappaLogin from './components/KappaLogin';
import PipelineForm from './components/PipelineForm';
import ProgressMonitor from './components/ProgressMonitor';
import PipelineHistory from './components/PipelineHistory';
import QualityReport from './components/QualityReport';
import ClinicalReport from './components/ClinicalReport';
import NIfTIViewer from './components/NIfTIViewer';
import ValidationPanel from './components/ValidationPanel';
import IncompletePatients from './components/IncompletePatients';
import PipelineLosses from './components/PipelineLosses';
import './App.css';
import { getEntitiesForRun, getKappaMe, getKappaHealth } from './services/api';

const { Header, Content } = Layout;
const { Title, Text } = Typography;

function App() {
  const [activeRun, setActiveRun] = useState(null);
  const [completedRuns, setCompletedRuns] = useState([]);
  const [activeTabKey, setActiveTabKey] = useState('pipeline');

  const [historyQualityReportRunId, setHistoryQualityReportRunId] = useState(null);
  const [historyVisualizationRunId, setHistoryVisualizationRunId] = useState(null);
  const [historyVisualizationLesionType, setHistoryVisualizationLesionType] = useState('glioblastoma');
  const [showHistoryQualityReport, setShowHistoryQualityReport] = useState(false);
  const [showHistoryVisualization, setShowHistoryVisualization] = useState(false);
  const [historyClinicalReportRunId, setHistoryClinicalReportRunId] = useState(null);
  const [historyClinicalReportLesionType, setHistoryClinicalReportLesionType] = useState('glioblastoma');
  const [showHistoryClinicalReport, setShowHistoryClinicalReport] = useState(false);
  const [historyIncompletePatientsRunId, setHistoryIncompletePatientsRunId] = useState(null);
  const [historyIncompletePatientsStatus, setHistoryIncompletePatientsStatus] = useState(null);
  const [showHistoryIncompletePatients, setShowHistoryIncompletePatients] = useState(false);
  const [historyPipelineLossesRunId, setHistoryPipelineLossesRunId] = useState(null);
  const [showHistoryPipelineLosses, setShowHistoryPipelineLosses] = useState(false);
  const [historyValidationRef, setHistoryValidationRef] = useState(null);
  const [kappaSession, setKappaSession] = useState(null);
  const [showLogin, setShowLogin] = useState(false);
  // null = ещё выясняем. Пока не знаем, не показываем ни форму входа, ни
  // предупреждение: иначе на долю секунды мелькает не то состояние.
  const [kappaReachable, setKappaReachable] = useState(null);
  const [sessionChecked, setSessionChecked] = useState(false);
  // Логин, под которым бэкенд войдёт сам, когда связь появится (без пароля).
  const [pendingLogin, setPendingLogin] = useState(null);
  // Оператор явно выбрал работать без входа. Намеренно не сохраняется:
  // после перезагрузки страницы снова предлагаем войти, а не молча
  // оставляем в анонимном режиме.
  const [workOffline, setWorkOffline] = useState(false);

  // Сессия живёт в БД бэкенда и переживает перезагрузку страницы. Без её
  // восстановления обновление F5 выбрасывало оператора в «работу без входа»,
  // хотя он был авторизован, а Kappa уже отвечала.
  useEffect(() => {
    let cancelled = false;

    const restore = async () => {
      const sessionId = localStorage.getItem('kappa_session_id');
      if (sessionId) {
        try {
          const me = await getKappaMe(sessionId);
          if (!cancelled) setKappaSession({ ...me, session_id: sessionId });
        } catch {
          // Сессия протухла или её нет — забываем и предлагаем войти заново.
          localStorage.removeItem('kappa_session_id');
        }
      }
      try {
        const health = await getKappaHealth();
        if (!cancelled) {
          setKappaReachable(health?.reachable !== false);
          setPendingLogin(health?.pending_login ?? null);
        }
      } catch {
        if (!cancelled) setKappaReachable(false);
      }
      if (!cancelled) setSessionChecked(true);
    };

    restore();
    return () => { cancelled = true; };
  }, []);

  // Пока входа нет, раз в 15 секунд проверяем, не ответила ли Kappa. Без
  // этого страница узнавала о возврате сервиса только по F5, из-за чего
  // предупреждение висело поверх уже работающей Kappa.
  useEffect(() => {
    if (kappaSession) return undefined;

    const interval = setInterval(async () => {
      try {
        const health = await getKappaHealth();
        setKappaReachable(health?.reachable !== false);
        setPendingLogin(health?.pending_login ?? null);

        // Бэкенд мог войти сам по отложенным учётным данным — тогда сессия
        // уже есть, и спрашивать оператора больше не о чем.
        const sessionId = localStorage.getItem('kappa_session_id');
        if (sessionId) {
          const me = await getKappaMe(sessionId);
          setKappaSession({ ...me, session_id: sessionId });
        }
      } catch {
        // Молча: это фоновая проверка, а не действие оператора.
      }
    }, 15000);
    return () => clearInterval(interval);
  }, [kappaSession]);

  // Форму показываем, пока оператор не вошёл, не поставил вход в очередь и
  // не отказался от него явно.
  const showLoginPage = sessionChecked && !kappaSession
    && !pendingLogin && !workOffline;

  const handleLoginSuccess = (data) => {
    setKappaSession(data);
    localStorage.setItem('kappa_session_id', data.session_id);
    setShowLogin(false);
    setKappaReachable(true);
    setWorkOffline(false);
    // Logging in adopts runs that finished while Kappa was down and clears
    // their backoff, so the operator is told delivery is moving again.
    if (data.resumed_uploads) {
      message.success(
        `Возобновлена выгрузка в Kappa: ${data.resumed_uploads} запуск(ов)`,
      );
    }
  };

  const handleLogout = async () => {
    if (kappaSession?.session_id) {
      try {
        await fetch(`/api/kappa/logout?session_id=${kappaSession.session_id}`, {
          method: 'POST',
        });
      } catch (e) {
        console.error('Logout error:', e);
      }
    }
    setKappaSession(null);
    localStorage.removeItem('kappa_session_id');
    // Бэкенд при выходе забывает и отложенный вход; сбрасываем и здесь,
    // чтобы форма появилась сразу, а не после ближайшего опроса здоровья.
    setPendingLogin(null);
    setWorkOffline(false);
  };

  /**
   * Показать отчёт из истории
   */
  const handleShowHistoryQualityReport = (runId) => {
    setHistoryQualityReportRunId(runId);
    setShowHistoryQualityReport(true);
  };

  /**
   * Показать визуализацию из истории
   */
  const handleShowHistoryVisualization = async (runId, lesionType = 'glioblastoma') => {
    setHistoryVisualizationRunId(runId);
    setHistoryVisualizationLesionType(lesionType);
    setHistoryValidationRef(null);
    setShowHistoryVisualization(true);

    // Keep all_entities so the viewer can correct the entity when the
    // user switches patients/sessions in a multi-patient run.
    try {
      const result = await getEntitiesForRun(runId);
      if (result.entities && result.entities.length > 0) {
        const e = result.entities[0];
        setHistoryValidationRef({
          entity_id: e.entity_id,
          dataset_id: e.dataset_id,
          all_entities: result.entities,
        });
      }
    } catch (err) {
      console.error('Ошибка загрузки entity для валидации:', err);
    }
  };

  const handleShowHistoryClinicalReport = (runId, lesionType = 'glioblastoma') => {
    setHistoryClinicalReportRunId(runId);
    setHistoryClinicalReportLesionType(lesionType);
    setShowHistoryClinicalReport(true);
  };

  /**
   * Показать неполных пациентов из истории
   */
  const handleShowHistoryIncompletePatients = (runId, status) => {
    setHistoryIncompletePatientsRunId(runId);
    setHistoryIncompletePatientsStatus(status);
    setShowHistoryIncompletePatients(true);
  };

  /**
   * Показать отчёт о потерянных пациентах по всем этапам
   */
  const handleShowHistoryPipelineLosses = (runId) => {
    setHistoryPipelineLossesRunId(runId);
    setShowHistoryPipelineLosses(true);
  };

  /**
   * Обработчик успешного запуска pipeline
   */
  const handlePipelineStarted = (response) => {
    console.log('Pipeline запущен:', response);
    setActiveRun({
      runId: response.run_id,
      status: response.status,
      createdAt: response.created_at,
      lesionType: response.lesion_type || 'glioblastoma',
    });
  };

  /**
   * Возобновление создаёт НОВЫЙ запуск, а не оживляет остановленный.
   * Показываем именно его: иначе вкладка запуска продолжает показывать
   * остановленный прогон с неактивными полосами и без кнопки остановки.
   */
  const handleRunResumed = (response) => {
    if (!response?.run_id) return;
    handlePipelineStarted(response);
    setActiveTabKey('pipeline');
  };

  /**
   * Переключить на вкладку истории запусков (используется баннером-ссылкой
   * в ProgressMonitor, ведущей к исходному запуску после requeue)
   */
  const handleSwitchToHistoryTab = () => {
    setActiveTabKey('history');
  };

  /**
   * Обработчик завершения pipeline
   */
  const handlePipelineComplete = (data) => {
    console.log('Pipeline завершён:', data);
    
    // Добавляем в список завершённых
    setCompletedRuns(prev => [...prev, {
      runId: data.run_id,
      status: data.status,
      completedAt: new Date().toISOString(),
    }]);
    
    // Убираем из активных
    // (оставляем на экране для просмотра результатов)
    // setActiveRun(null);
  };

  return (
    <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
      {/* The title is long enough to wrap on ~1366px laptops once the user
          name sits beside it. antd's Header is a fixed 64px with a 64px line
          height, so wrapped text spilled out of the bar; let it grow instead. */}
      <Layout.Header style={{
        background: '#1890ff',
        padding: '8px 24px',
        height: 'auto',
        minHeight: 64,
        lineHeight: 1.3,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: 16,
      }}>
        <Typography.Title level={4} style={{ color: 'white', margin: 0 }}>
          🧠 ИИ-система дистанционной диагностики и мониторинга социально значимых заболеваний
        </Typography.Title>
        {kappaSession ? (
          <Space style={{ flexShrink: 0 }}>
            <Text style={{ color: 'white', whiteSpace: 'nowrap' }}>
              {kappaSession.first_name} {kappaSession.last_name}
            </Text>
            <Button
              icon={<LogoutOutlined />}
              onClick={handleLogout}
              size="small"
              ghost
            >
              Выход
            </Button>
          </Space>
        ) : (
          <Space style={{ flexShrink: 0 }}>
            <Text style={{ color: '#ffd666', whiteSpace: 'nowrap' }}>
              Без входа в Kappa
            </Text>
            <Button size="small" ghost onClick={() => setShowLogin(true)}>
              Войти в Kappa
            </Button>
          </Space>
        )}
      </Layout.Header>

      <Layout.Content style={{ padding: '24px', maxWidth: 1400, margin: '0 auto', width: '100%' }}>
        {/* Форма входа — точка входа всегда, доступна Kappa или нет.
            Прятать её при недоступной Kappa значило лишать оператора
            возможности поставить вход в очередь: именно ввод логина её и
            запускает. Обходим форму только если вход уже поставлен в
            очередь или оператор явно выбрал работать без него. */}
        {showLoginPage && (
          <KappaLogin
            onLoginSuccess={handleLoginSuccess}
            kappaReachable={kappaReachable}
            onSkip={kappaReachable === false ? () => setWorkOffline(true) : undefined}
          />
        )}
        {sessionChecked && !kappaSession && !showLoginPage && kappaReachable === false && (
          <Alert
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
            message="Kappa сейчас недоступна — работаем без входа"
            description={
              pendingLogin
                ? `Вход под «${pendingLogin}» выполнится автоматически, как `
                  + 'только связь восстановится, и результаты уйдут в Kappa '
                  + 'сами. Обработку можно запускать прямо сейчас.'
                : 'Обработку можно запускать: результаты сохранятся локально '
                  + 'и уйдут в Kappa автоматически, как только вы войдёте и '
                  + 'сервис станет доступен. Пациентам присвоят номера при '
                  + 'выгрузке.'
            }
            action={
              !pendingLogin && (
                <Button size="small" onClick={() => setShowLogin(true)}>
                  Войти
                </Button>
              )
            }
          />
        )}
        <Modal
          open={showLogin}
          onCancel={() => setShowLogin(false)}
          footer={null}
          title="Вход в Kappa"
          destroyOnClose
        >
          <KappaLogin onLoginSuccess={handleLoginSuccess} />
        </Modal>
        {!showLoginPage && (
          <>
            <Tabs
              activeKey={activeTabKey}
              onChange={setActiveTabKey}
              size="large"
              items={[
                {
                  key: 'pipeline',
                  label: (
                    <span>
                      <RocketOutlined />
                      Запуск обработки
                    </span>
                  ),
                  children: (
                    <>
                      {!activeRun ? (
                        <Card>
                          <PipelineForm onPipelineStarted={handlePipelineStarted} />
                        </Card>
                      ) : (
                        <ProgressMonitor
                          // Remount on a new run instead of reusing the
                          // previous one's state. Without this React keeps
                          // the same instance across runs, so a finished
                          // run's status/stages leak into the next one — a
                          // run started after a stop inherited status
                          // "stopped" and never rendered its Stop button.
                          key={activeRun.runId}
                          runId={activeRun.runId}
                          lesionType={activeRun.lesionType}
                          onComplete={handlePipelineComplete}
                          onRequeued={handlePipelineStarted}
                          onSwitchToHistory={handleSwitchToHistoryTab}
                        />
                      )}
                      {completedRuns.length > 0 && (
                        <>
                          <Divider>Новая обработка</Divider>
                          <Card>
                            <PipelineForm onPipelineStarted={handlePipelineStarted} />
                          </Card>
                        </>
                      )}
                    </>
                  ),
                },
                {
                  key: 'history',
                  label: (
                    <span>
                      <HistoryOutlined />
                      История запусков
                    </span>
                  ),
                  children: (
                    <PipelineHistory
                      onRunResumed={handleRunResumed}
                      onShowQualityReport={handleShowHistoryQualityReport}
                      onShowVisualization={handleShowHistoryVisualization}
                      onShowClinicalReport={handleShowHistoryClinicalReport}
                      onShowIncompletePatients={handleShowHistoryIncompletePatients}
                      onShowPipelineLosses={handleShowHistoryPipelineLosses}
                    />
                  ),
                },
                {
                  key: 'validation',
                  label: (
                    <span>
                      <CheckCircleOutlined />
                      Валидация
                    </span>
                  ),
                  children: <ValidationPanel />,
              },
              ]}
            />

            {showHistoryQualityReport && (
              <QualityReport
                runId={historyQualityReportRunId}
                visible={showHistoryQualityReport}
                onClose={() => setShowHistoryQualityReport(false)}
              />
            )}
            {showHistoryVisualization && (
              <NIfTIViewer
                runId={historyVisualizationRunId}
                visible={showHistoryVisualization}
                onClose={() => setShowHistoryVisualization(false)}
                validationRef={historyValidationRef}
                lesionType={historyVisualizationLesionType}
                onValidationRefChange={(ref) =>
                  setHistoryValidationRef((prev) => ({ ...prev, ...ref }))
                }
              />
            )}
            {showHistoryClinicalReport && (
              <ClinicalReport
                runId={historyClinicalReportRunId}
                visible={showHistoryClinicalReport}
                onClose={() => setShowHistoryClinicalReport(false)}
                lesionType={historyClinicalReportLesionType}
              />
            )}
            {showHistoryIncompletePatients && (
              <IncompletePatients
                runId={historyIncompletePatientsRunId}
                visible={showHistoryIncompletePatients}
                onClose={() => setShowHistoryIncompletePatients(false)}
                canRequeue={
                  historyIncompletePatientsStatus === 'completed' ||
                  historyIncompletePatientsStatus === 'failed'
                }
                onRequeued={handlePipelineStarted}
              />
            )}
            {showHistoryPipelineLosses && (
              <PipelineLosses
                runId={historyPipelineLossesRunId}
                visible={showHistoryPipelineLosses}
                onClose={() => setShowHistoryPipelineLosses(false)}
              />
            )}
          </>
        )}
      </Layout.Content>
    </Layout>
  );
}

export default App;