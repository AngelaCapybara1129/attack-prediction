import { useEffect, useState } from 'react'
import { BookOpen, Pause, Play, ShieldAlert, X } from 'lucide-react'
import { apiFetch, withToken } from '../api/client'
import { ACTION_NAMES, LEVEL_META, STATUS_META, formatTime } from '../constants/risk'
import type { EventItem, EventStatus } from '../types/intent'
import IndexBars from './IndexBars'
import RiskBadge from './RiskBadge'
import Sparkline from './Sparkline'

type EventDetailDrawerProps = {
  eventId: number | null
  onClose: () => void
  onUpdated?: (ev: EventItem) => void
}

const STATUS_ORDER: EventStatus[] = ['pending', 'handled', 'escalated', 'dismissed']

/** 告警卡詳情：關鍵畫面／事件前回放、X1–X9、觸發規則、RAG 說明與處置建議、相似案例、處置狀態 */
export default function EventDetailDrawer({ eventId, onClose, onUpdated }: EventDetailDrawerProps) {
  const [ev, setEv] = useState<EventItem | null>(null)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')
  const [saving, setSaving] = useState(false)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    setEv(null)
    setError('')
    setFrame(0)
    setPlaying(false)
    if (eventId === null) return
    let alive = true
    apiFetch<EventItem>(`/api/events/${eventId}`)
      .then((data) => {
        if (!alive) return
        setEv(data)
        setNote(data.note ?? '')
        setPlaying(data.clip_frames > 0)
      })
      .catch((err: unknown) => alive && setError(err instanceof Error ? err.message : '讀取失敗'))
    return () => {
      alive = false
    }
  }, [eventId])

  useEffect(() => {
    if (!playing || !ev || ev.clip_frames <= 0) return
    const timer = window.setInterval(() => {
      setFrame((f) => (f + 1) % ev.clip_frames)
    }, 200)
    return () => window.clearInterval(timer)
  }, [playing, ev])

  if (eventId === null) return null

  const save = async (status?: EventStatus) => {
    if (!ev) return
    setSaving(true)
    try {
      const body: { note: string; status?: EventStatus } = { note }
      if (status) body.status = status
      const updated = await apiFetch<EventItem>(`/api/events/${ev.id}`, {
        method: 'PATCH',
        body: JSON.stringify(body),
      })
      setEv(updated)
      onUpdated?.(updated)
    } catch (err) {
      setError(err instanceof Error ? err.message : '更新失敗')
    } finally {
      setSaving(false)
    }
  }

  const showClip = ev && ev.clip_frames > 0 && (playing || frame > 0)
  const imgSrc = ev
    ? showClip
      ? withToken(`/api/events/${ev.id}/clip/${frame}`)
      : ev.has_snapshot
        ? withToken(`/api/events/${ev.id}/snapshot`)
        : ''
    : ''

  return (
    <div className="fixed inset-0 z-[60] flex justify-end bg-slate-900/30 backdrop-blur-sm" onClick={onClose}>
      <aside
        className="h-full w-full max-w-xl overflow-y-auto bg-white p-4 shadow-2xl md:p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-center gap-2">
          <ShieldAlert className="h-5 w-5 text-slate-700" />
          <h2 className="text-sm font-black text-slate-900">告警事件 #{eventId}</h2>
          <button onClick={onClose} className="ml-auto rounded p-1 text-slate-500 hover:bg-slate-100" aria-label="關閉">
            <X className="h-4 w-4" />
          </button>
        </div>

        {error && <p className="mb-3 rounded bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>}
        {!ev ? (
          !error && <p className="py-10 text-center text-xs text-slate-500">載入中…</p>
        ) : (
          <div className="space-y-4">
            <section className={`rounded-xl border p-3 ${LEVEL_META[ev.level].border} ${LEVEL_META[ev.level].soft}`}>
              <div className="flex flex-wrap items-center gap-2">
                <RiskBadge level={ev.level} withLabel size="md" />
                <span className="text-xs font-bold text-slate-800">
                  {ev.camera_id} ・ 人物 P{ev.person_id}
                </span>
                <span className="text-[11px] text-slate-500">{formatTime(ev.occurred_at)}</span>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-slate-700">
                <span>
                  風險分數 <b className="text-base tabular-nums">{ev.score.toFixed(2)}</b>
                </span>
                {ev.model_score !== null && (
                  <span>
                    規則 {ev.rule_score?.toFixed(2)} ／ 模型 {ev.model_score.toFixed(2)}
                  </span>
                )}
                {ev.action && (
                  <span className="rounded bg-indigo-100 px-1.5 py-0.5 text-indigo-800">
                    ST-GCN：{ACTION_NAMES[ev.action] ?? ev.action}
                    {ev.action_conf !== null && ` ${(ev.action_conf * 100).toFixed(0)}%`}
                  </span>
                )}
                {ev.appearance?.upper_zh && (
                  <span className="rounded bg-slate-100 px-1.5 py-0.5">
                    上身{ev.appearance.upper_zh}色{ev.appearance.lower_zh ? `／下身${ev.appearance.lower_zh}色` : ''}
                  </span>
                )}
              </div>
              {ev.score_history && ev.score_history.length > 1 && (
                <div className="mt-2">
                  <Sparkline points={ev.score_history} width={300} height={36} color={LEVEL_META[ev.level].hex} />
                </div>
              )}
            </section>

            {imgSrc && (
              <section>
                <div className="relative overflow-hidden rounded-xl bg-slate-950">
                  <img src={imgSrc} alt="事件畫面" className="w-full object-contain" />
                  {ev.clip_frames > 0 && (
                    <div className="absolute bottom-2 left-2 right-2 flex items-center gap-2 rounded-lg bg-black/50 px-2 py-1 text-white">
                      <button onClick={() => setPlaying((p) => !p)} aria-label={playing ? '暫停' : '播放'}>
                        {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                      </button>
                      <input
                        type="range"
                        min={0}
                        max={ev.clip_frames - 1}
                        value={frame}
                        onChange={(e) => {
                          setPlaying(false)
                          setFrame(Number(e.target.value))
                        }}
                        className="flex-1"
                      />
                      <span className="text-[10px] tabular-nums">事件前回放 {frame + 1}/{ev.clip_frames}</span>
                    </div>
                  )}
                </div>
              </section>
            )}

            <section>
              <h3 className="mb-2 text-xs font-black text-slate-500">九項攻擊意圖指數（前 5 秒峰值｜藍線＝預測下一秒）</h3>
              <IndexBars values={ev.indices_peak ?? ev.indices} predicted={ev.predicted_next} />
            </section>

            {ev.triggers && ev.triggers.length > 0 && (
              <section>
                <h3 className="mb-1 text-xs font-black text-slate-500">觸發規則（表 2）</h3>
                <ul className="space-y-1 text-xs text-slate-800">
                  {ev.triggers.map((t) => (
                    <li key={t.rule} className="flex items-center gap-2">
                      <RiskBadge level={t.level} />
                      {t.description}
                      <span className="text-slate-500">（{t.duration}s）</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {ev.explanation && (
              <section>
                <h3 className="mb-1 text-xs font-black text-slate-500">判斷說明（RAG）</h3>
                <p className="whitespace-pre-wrap rounded-lg bg-slate-50 p-3 text-xs leading-6 text-slate-800">{ev.explanation}</p>
              </section>
            )}

            {ev.recommendations && ev.recommendations.length > 0 && (
              <section>
                <h3 className="mb-1 text-xs font-black text-slate-500">建議處置</h3>
                <ol className="list-decimal space-y-0.5 pl-5 text-xs text-slate-800">
                  {ev.recommendations.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ol>
              </section>
            )}

            {ev.similar && ev.similar.length > 0 && (
              <section>
                <h3 className="mb-1 flex items-center gap-1 text-xs font-black text-slate-500">
                  <BookOpen className="h-3.5 w-3.5" /> 相似參考
                </h3>
                <ul className="space-y-1.5">
                  {ev.similar.map((s) => (
                    <li key={s.key} className="rounded-lg border border-slate-200 p-2 text-xs">
                      <div className="flex items-center gap-2">
                        {s.level && <RiskBadge level={s.level} />}
                        <span className="font-bold text-slate-800">{s.title}</span>
                        <span className="ml-auto tabular-nums text-slate-500">{(s.similarity * 100).toFixed(0)}%</span>
                      </div>
                      {s.summary && <p className="mt-1 text-slate-600">{s.summary}</p>}
                      {s.kind === 'case' && (
                        <p className="mt-0.5 text-[10px] text-amber-700">示範情境（非真實案件）{s.references?.length ? `・依據：${s.references.join('、')}` : ''}</p>
                      )}
                      {s.note && <p className="mt-0.5 text-[11px] text-emerald-700">處置紀錄：{s.note}</p>}
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {ev.camera_path && ev.camera_path.length > 1 && (
              <section className="text-xs text-slate-700">
                <h3 className="mb-1 text-xs font-black text-slate-500">跨鏡頭軌跡</h3>
                {ev.camera_path.map((c) => c.camera_id).join(' → ')}
              </section>
            )}

            <section className="rounded-xl border border-slate-200 p-3">
              <h3 className="mb-2 text-xs font-black text-slate-500">處置狀態</h3>
              <div className="mb-2 flex flex-wrap gap-1.5">
                {STATUS_ORDER.map((s) => (
                  <button
                    key={s}
                    disabled={saving}
                    onClick={() => void save(s)}
                    className={`rounded-lg px-2.5 py-1 text-[11px] font-bold transition disabled:opacity-50 ${
                      ev.status === s ? 'ring-2 ring-slate-800 ' + STATUS_META[s].chip : STATUS_META[s].chip + ' opacity-70 hover:opacity-100'
                    }`}
                  >
                    {STATUS_META[s].label}
                  </button>
                ))}
              </div>
              <textarea
                value={note}
                onChange={(e) => setNote(e.target.value)}
                rows={3}
                placeholder="處置紀錄（會寫入向量庫，成為往後相似事件的參考）"
                className="w-full rounded-lg border border-slate-200 p-2 text-xs outline-none focus:border-slate-400"
              />
              <div className="mt-1 flex items-center">
                <span className="text-[10px] text-slate-500">
                  {ev.handled_by ? `${ev.handled_by}・${formatTime(ev.handled_at)}` : '尚未處置'}
                </span>
                <button
                  disabled={saving}
                  onClick={() => void save()}
                  className="ml-auto rounded-lg bg-slate-800 px-3 py-1 text-[11px] font-bold text-white disabled:opacity-50"
                >
                  儲存紀錄
                </button>
              </div>
            </section>
          </div>
        )}
      </aside>
    </div>
  )
}
