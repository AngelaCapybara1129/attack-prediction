import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { Clock, Bell, X, Camera, LogOut, RefreshCw, Users, Wifi, WifiOff, History } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import YoloViewer from '../components/YoloViewer'
import MultiCameraViewer from '../components/MultiCameraViewer'
import AlertCard from '../components/AlertCard'
import EventDetailDrawer from '../components/EventDetailDrawer'
import LivePersonsPanel from '../components/LivePersonsPanel'
import RiskBadge from '../components/RiskBadge'
import { apiFetch } from '../api/client'
import { LEVEL_META, levelIndex } from '../constants/risk'
import { useAlertSocket } from '../hooks/useAlertSocket'
import { playAlarm } from '../utils/alarm'
import type { EventItem, EventListResponse, EventStats, SocketMessage } from '../types/intent'

import PageBackground from '../components/PageBackground'

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

type ViewMode = 'single' | 'multi'
type SidePanel = 'alerts' | 'persons'

type CameraItem = {
  id: string
  name: string
  source: string
  streamUrl?: string
}

type CameraListResponse = {
  count: number
  cameras: CameraItem[]
}

const glassCard =
  'relative rounded-[16px] 2xl:rounded-[18px] border border-white/50 bg-white/45 backdrop-blur-2xl backdrop-saturate-150 shadow-[0_10px_30px_rgba(15,23,42,0.08)] ring-1 ring-white/60'

const controlButton =
  'rounded-lg px-2 py-1.5 md:px-3 md:py-2 text-[10px] md:text-xs font-bold transition disabled:cursor-not-allowed disabled:opacity-50 whitespace-nowrap'

const venues: string[] = ['車站A', '車站B', '車站C', '商圈']

const Dashboard: React.FC = () => {
  const navigate = useNavigate()

  const [currentVenue, setCurrentVenue] = useState<string>(venues[0])
  const [viewMode, setViewMode] = useState<ViewMode>('multi')
  const [isMonitoring, setIsMonitoring] = useState<boolean>(false)
  const [selectedCamera, setSelectedCamera] = useState<string>('0')
  const [alerts, setAlerts] = useState<EventItem[]>([])
  const [currentTime, setCurrentTime] = useState<Date>(new Date())

  const [cameras, setCameras] = useState<CameraItem[]>([])
  const [cameraLoading, setCameraLoading] = useState<boolean>(true)
  const [cameraError, setCameraError] = useState<string>('')

  const [taskId, setTaskId] = useState<string>('')
  const [sidePanel, setSidePanel] = useState<SidePanel>('alerts')
  const [onlyMarked, setOnlyMarked] = useState<boolean>(true)
  const [stats, setStats] = useState<EventStats | null>(null)
  const [openEventId, setOpenEventId] = useState<number | null>(null)
  const [toast, setToast] = useState<EventItem | null>(null)

  const handleLogout = (): void => {
    localStorage.removeItem('token')
    navigate('/login')
  }

  const loadStats = useCallback(async () => {
    try {
      setStats(await apiFetch<EventStats>('/api/events/stats?hours=24'))
    } catch {
      // 後端未啟動時忽略
    }
  }, [])

  const loadAlerts = useCallback(async () => {
    try {
      const data = await apiFetch<EventListResponse>('/api/events?limit=50&min_level=L1')
      setAlerts(data.items)
    } catch {
      // 後端未啟動時忽略
    }
  }, [])

  const onSocketMessage = useCallback(
    (msg: SocketMessage) => {
      if (msg.type === 'alert') {
        const ev = msg.event
        setAlerts((prev) => [ev, ...prev.filter((a) => a.id !== ev.id)].slice(0, 100))
        if (levelIndex(ev.level) >= 3) {
          playAlarm(ev.level)
          setToast(ev)
        }
        void loadStats()
      } else if (msg.type === 'event_updated') {
        const ev = msg.event
        setAlerts((prev) => prev.map((a) => (a.id === ev.id ? ev : a)))
        void loadStats()
      }
    },
    [loadStats],
  )
  const { connected } = useAlertSocket(onSocketMessage)

  const visibleAlerts = useMemo(
    () => (onlyMarked ? alerts.filter((a) => levelIndex(a.level) >= 2) : alerts),
    [alerts, onlyMarked],
  )

  const handleTaskChange = useCallback((id: string) => setTaskId(id), [])

  const onEventUpdated = useCallback(
    (ev: EventItem) => {
      setAlerts((prev) => prev.map((a) => (a.id === ev.id ? ev : a)))
      void loadStats()
    },
    [loadStats],
  )

  const cameraOptions = useMemo(
    () => cameras.map((camera) => camera.source),
    [cameras]
  )

  const multiSources = useMemo(
    () => cameras.map((camera) => camera.source),
    [cameras]
  )

  const fetchCameras = async () => {
    try {
      setCameraLoading(true)
      setCameraError('')

      const res = await fetch(`${BASE_URL}/yolo/cameras`)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }

      const data: CameraListResponse = await res.json()
      const nextCameras = Array.isArray(data.cameras) ? data.cameras : []

      setCameras(nextCameras)

      if (nextCameras.length > 0) {
        const hasSelected = nextCameras.some((camera) => camera.source === selectedCamera)
        if (!hasSelected) {
          setSelectedCamera(nextCameras[0].source)
        }
      }
    } catch (err) {
      console.error(err)
      setCameraError('無法取得鏡頭資料，請確認 FastAPI /cameras 是否正常')
      setCameras([])
    } finally {
      setCameraLoading(false)
    }
  }

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    fetchCameras()
  }, [])

  useEffect(() => {
    void loadAlerts()
    void loadStats()
    const timer = setInterval(() => void loadStats(), 30000)
    return () => clearInterval(timer)
  }, [loadAlerts, loadStats])

  useEffect(() => {
    if (!toast) return
    const timer = setTimeout(() => setToast(null), 10000)
    return () => clearTimeout(timer)
  }, [toast])

  /** 只從面板移除，不影響資料庫中的事件紀錄 */
  const clearAllAlerts = () => setAlerts([])

  const modeButton = (active: boolean) =>
    `${controlButton} ${
      active
        ? 'bg-emerald-400 text-emerald-950 shadow-sm'
        : 'bg-slate-900/8 text-slate-700 hover:bg-slate-900/12'
    }`

  return (
    <div className="relative min-h-screen lg:h-[100dvh] w-full overflow-x-hidden bg-white text-slate-900">
      <PageBackground />

      <div className="relative z-10 flex min-h-screen flex-col gap-3 p-3 md:p-4 lg:h-[100dvh] lg:overflow-hidden lg:p-5">
        <header className={`${glassCard} shrink-0 p-3`}>
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="flex min-w-0 items-center gap-3">
              <Clock className="h-5 w-5 shrink-0 text-emerald-500" />
              <div className="min-w-0">
                <p className="truncate text-sm font-black tabular-nums md:text-lg text-slate-900">
                  {currentTime.toLocaleString('zh-TW', { hour12: false })}
                </p>
                <p className="flex items-center gap-1 text-[11px] text-slate-500">
                  AI 風險監視儀表板・{currentVenue}
                  {connected ? (
                    <Wifi className="ml-1 h-3 w-3 text-emerald-500" aria-label="即時告警已連線" />
                  ) : (
                    <WifiOff className="ml-1 h-3 w-3 text-slate-400" aria-label="即時告警未連線" />
                  )}
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <span className="rounded-lg bg-orange-100 px-2 py-1 font-bold text-orange-800">
                L3 {stats?.by_level.L3 ?? 0}
              </span>
              <span className="rounded-lg bg-red-100 px-2 py-1 font-bold text-red-700">
                L4 {stats?.by_level.L4 ?? 0}
              </span>
              <span className="rounded-lg bg-slate-100 px-2 py-1 text-slate-700">
                24h 處置率 {stats ? Math.round(stats.handled_rate * 100) : 100}%
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <select
                value={currentVenue}
                onChange={(e) => setCurrentVenue(e.target.value)}
                className="rounded-lg border border-slate-200/80 bg-white/70 px-2 py-1 text-xs text-slate-700 outline-none backdrop-blur"
              >
                {venues.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>

              <div className="flex gap-1 rounded-xl bg-slate-900/5 p-1">
                {(['single', 'multi'] as ViewMode[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => setViewMode(m)}
                    className={modeButton(viewMode === m)}
                  >
                    {m === 'single' ? '單鏡頭' : `多鏡頭(${cameras.length})`}
                  </button>
                ))}
              </div>

              {viewMode === 'single' && (
                <div className="flex items-center gap-1 rounded-xl bg-slate-900/5 p-1">
                  <Camera className="h-4 w-4 text-slate-500" />
                  <select
                    value={selectedCamera}
                    onChange={(e) => setSelectedCamera(e.target.value)}
                    className="rounded-md bg-transparent px-1 py-1 text-xs text-slate-700 outline-none"
                    disabled={cameras.length === 0}
                  >
                    {cameraOptions.length === 0 ? (
                      <option value="">無鏡頭</option>
                    ) : (
                      cameraOptions.map((cam) => (
                        <option key={cam} value={cam} className="bg-white text-slate-800">
                          {cameras.find((c) => c.source === cam)?.name ?? `Cam ${cam}`}
                        </option>
                      ))
                    )}
                  </select>
                </div>
              )}

              <button
                onClick={fetchCameras}
                className={`${controlButton} flex items-center gap-1.5 bg-slate-200 text-slate-800 hover:bg-slate-300`}
              >
                <RefreshCw className="h-3.5 w-3.5" />
                <span>刷新鏡頭</span>
              </button>

              <button
                onClick={() => setIsMonitoring((prev) => !prev)}
                className={`${controlButton} ${
                  isMonitoring
                    ? 'bg-red-500 text-white hover:bg-red-600'
                    : 'bg-lime-400 text-lime-950 hover:bg-lime-500'
                }`}
              >
                {isMonitoring ? '停止監測' : '開始監測'}
              </button>

              <button
                onClick={handleLogout}
                className={`${controlButton} flex items-center gap-1.5 bg-red-800 text-white hover:bg-red-600`}
                aria-label="登出系統"
              >
                <LogOut className="h-3.5 w-3.5" />
                <span>登出</span>
              </button>
            </div>
          </div>
        </header>

        <main className="grid flex-1 grid-cols-1 gap-4 lg:min-h-0 lg:grid-cols-4 xl:grid-cols-5">
          <section
            className={`${glassCard} flex min-h-[500px] flex-col p-3 lg:col-span-3 lg:min-h-0 xl:col-span-4`}
          >
            <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain custom-scrollbar rounded-lg bg-white/25">
              {cameraLoading ? (
                <div className="flex h-full min-h-[300px] items-center justify-center text-sm text-slate-500">
                  讀取鏡頭中...
                </div>
              ) : cameraError ? (
                <div className="flex h-full min-h-[300px] items-center justify-center text-sm text-red-500">
                  {cameraError}
                </div>
              ) : cameras.length === 0 ? (
                <div className="flex h-full min-h-[300px] items-center justify-center text-sm text-slate-500">
                  目前沒有可用鏡頭
                </div>
              ) : viewMode === 'single' ? (
                <YoloViewer isMonitoring={isMonitoring} source={selectedCamera} onTaskChange={handleTaskChange} />
              ) : (
                <MultiCameraViewer
                  isMonitoring={isMonitoring}
                  sources={multiSources}
                  onTaskChange={handleTaskChange}
                />
              )}
            </div>
          </section>

          <aside className="flex min-h-0 flex-col gap-4 lg:col-span-1">
            <div
              className={`${glassCard} flex min-h-[300px] flex-1 flex-col overflow-hidden p-4 lg:min-h-0`}
            >
              <div className="mb-3 flex items-center gap-1 rounded-xl bg-slate-900/5 p-1">
                <button
                  onClick={() => setSidePanel('alerts')}
                  className={modeButton(sidePanel === 'alerts') + ' flex flex-1 items-center justify-center gap-1'}
                >
                  <Bell className="h-3.5 w-3.5" />
                  即時告警
                </button>
                <button
                  onClick={() => setSidePanel('persons')}
                  className={modeButton(sidePanel === 'persons') + ' flex flex-1 items-center justify-center gap-1'}
                >
                  <Users className="h-3.5 w-3.5" />
                  人物風險
                </button>
              </div>

              {sidePanel === 'alerts' ? (
                <>
                  <div className="mb-2 flex items-center gap-2 text-[10px] text-slate-500">
                    <label className="flex cursor-pointer items-center gap-1">
                      <input
                        type="checkbox"
                        checked={onlyMarked}
                        onChange={(e) => setOnlyMarked(e.target.checked)}
                      />
                      只看 L2 以上
                    </label>
                    <Link to="/events" className="ml-auto flex items-center gap-0.5 hover:text-slate-800">
                      <History className="h-3 w-3" />
                      全部紀錄
                    </Link>
                    {alerts.length > 0 && (
                      <button onClick={clearAllAlerts} className="transition hover:text-slate-800">
                        清空
                      </button>
                    )}
                  </div>

                  <div className="flex-1 overflow-y-auto overscroll-contain custom-scrollbar pr-1">
                    {visibleAlerts.length === 0 ? (
                      <div className="flex h-full min-h-[180px] items-center justify-center rounded-xl border border-dashed border-slate-200 bg-white/40 text-center">
                        <div>
                          <Bell className="mx-auto mb-2 h-5 w-5 text-slate-300" />
                          <p className="text-xs font-semibold text-slate-700">目前沒有告警</p>
                          <p className="mt-1 text-[11px] text-slate-500">
                            系統將在偵測到攻擊前兆時即時顯示
                          </p>
                        </div>
                      </div>
                    ) : (
                      visibleAlerts.map((alert) => (
                        <AlertCard key={alert.id} event={alert} onOpen={setOpenEventId} />
                      ))
                    )}
                  </div>
                </>
              ) : (
                <div className="flex-1 overflow-y-auto overscroll-contain custom-scrollbar pr-1">
                  <LivePersonsPanel taskId={taskId} />
                </div>
              )}
            </div>
          </aside>
        </main>
      </div>

      {toast && (
        <div className="fixed left-1/2 top-4 z-[70] w-[min(92vw,420px)] -translate-x-1/2">
          <div
            className={`flex items-start gap-3 rounded-xl border-2 bg-white p-3 shadow-2xl ${LEVEL_META[toast.level].border} ${toast.level === 'L4' ? 'animate-pulse' : ''}`}
          >
            <RiskBadge level={toast.level} withLabel size="md" />
            <button
              className="min-w-0 flex-1 text-left"
              onClick={() => {
                setOpenEventId(toast.id)
                setToast(null)
              }}
            >
              <p className="text-xs font-black text-slate-900">
                {toast.camera_id} ・ 人物 P{toast.person_id} ・ R {toast.score.toFixed(2)}
              </p>
              <p className="mt-0.5 line-clamp-2 text-[11px] text-slate-600">
                {toast.triggers?.[0]?.description ?? toast.explanation?.split('\n')[1] ?? '偵測到高風險攻擊前兆'}
              </p>
            </button>
            <button onClick={() => setToast(null)} className="text-slate-400 hover:text-slate-700" aria-label="關閉通知">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      <EventDetailDrawer
        eventId={openEventId}
        onClose={() => setOpenEventId(null)}
        onUpdated={onEventUpdated}
      />

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; height: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(100, 116, 139, 0.35);
          border-radius: 999px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
      `}</style>
    </div>
  )
}

export default Dashboard