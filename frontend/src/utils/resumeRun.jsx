/**
 * Возобновление остановленного запуска.
 *
 * Lives outside both call sites because the flow is not a single request:
 * when settings changed since the stop, the backend answers 409 with the
 * list of differences and the operator has to choose. Duplicating that
 * across the history table and the progress view would mean two dialogs
 * drifting apart.
 */
import { Modal, message } from 'antd';
import { resumePipelineRun } from '../services/api';

/**
 * Read whatever the backend put in `detail` into a displayable string.
 * FastAPI sends a plain string for simple errors and an object for the
 * structured ones this endpoint raises.
 */
const errorText = (detail, fallback) => {
  if (typeof detail === 'string') return detail;
  return detail?.message || fallback;
};

/**
 * Resume `runId`, asking the operator first if settings drifted.
 *
 * onResumed receives the newly created run so the caller can switch to it —
 * resuming starts a NEW run (linked by parent_run_id), it does not revive
 * the stopped one, so anything showing the old run must move on.
 */
export const confirmAndResume = async (runId, { onResumed } = {}) => {
  try {
    const result = await resumePipelineRun(runId);
    message.success('Обработка возобновлена');
    onResumed?.(result);
    return result;
  } catch (error) {
    const detail = error.response?.data?.detail;

    if (error.response?.status === 409 && detail?.differences) {
      Modal.confirm({
        title: 'Настройки изменились с момента остановки',
        content: (
          <div>
            <p>После остановки изменилось следующее:</p>
            <ul>
              {detail.differences.map((d) => (
                <li key={d.setting}>
                  {d.setting}: <b>{d.was}</b> → <b>{d.now}</b>
                </li>
              ))}
            </ul>
            <p>
              Если продолжить на новых настройках, часть пациентов будет
              обработана иначе, чем остальные.
            </p>
          </div>
        ),
        okText: 'На прежних настройках',
        cancelText: 'Отмена',
        onOk: async () => {
          try {
            const result = await resumePipelineRun(runId, true);
            message.success('Обработка возобновлена на сохранённых настройках');
            onResumed?.(result);
            return result;
          } catch (okError) {
            const okDetail = okError.response?.data?.detail;
            if (
              okError.response?.status === 409 &&
              typeof okDetail === 'object' &&
              okDetail?.reason === 'snapshot_unavailable'
            ) {
              // The retained config is gone, so resuming on the old settings
              // is not something we can honestly offer.
              message.error(okDetail.message || 'Снимок настроек недоступен');
            } else {
              message.error(errorText(okDetail, 'Не удалось возобновить обработку'));
            }
            throw okError;
          }
        },
      });
      return null;
    }

    if (
      error.response?.status === 409 &&
      typeof detail === 'object' &&
      detail?.reason === 'snapshot_unavailable'
    ) {
      message.error(detail.message || 'Снимок настроек недоступен');
    } else {
      message.error(errorText(detail, 'Не удалось возобновить обработку'));
    }
    return null;
  }
};
