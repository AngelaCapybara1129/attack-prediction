export type RiskLevel = 'L0' | 'L1' | 'L2' | 'L3' | 'L4'
export type IndexKey = 'X1' | 'X2' | 'X3' | 'X4' | 'X5' | 'X6' | 'X7' | 'X8' | 'X9'
export type EventStatus = 'pending' | 'handled' | 'escalated' | 'dismissed'
export type IndexMap = Record<string, number>

export interface Trigger {
  rule: string
  level: RiskLevel
  description: string
  duration: number
}

export interface CameraPathItem {
  camera_id: string
  first_ts: number
  last_ts: number
}

export interface ScorePoint {
  ts: number
  score: number
  level: RiskLevel
}

export interface SimilarItem {
  key: string
  kind: 'case' | 'event' | null
  event_id?: number | null
  title?: string | null
  level?: RiskLevel | null
  similarity: number
  summary?: string | null
  status?: EventStatus | null
  note?: string | null
  handling?: string[] | null
  references?: string[] | null
  synthetic?: boolean
  occurred_at?: string | null
}

export interface Appearance {
  upper?: string | null
  lower?: string | null
  upper_zh?: string | null
  lower_zh?: string | null
}

export interface EventItem {
  id: number
  alert_id: string
  task_id: string | null
  camera_id: string
  person_id: number
  track_id: number | null
  level: RiskLevel
  score: number
  rule_score: number | null
  model_score: number | null
  indices: IndexMap
  indices_peak: IndexMap | null
  predicted_next: IndexMap | null
  triggers: Trigger[] | null
  aux: Record<string, unknown> | null
  action: string | null
  action_conf: number | null
  bbox: number[] | null
  trajectory: number[][] | null
  camera_path: CameraPathItem[] | null
  score_history: ScorePoint[] | null
  appearance: Appearance | null
  explanation: string | null
  recommendations: string[] | null
  similar: SimilarItem[] | null
  clip_frames: number
  has_snapshot: boolean
  status: EventStatus
  handled_by: string | null
  handled_at: string | null
  note: string | null
  occurred_at: string
  created_at: string | null
}

export interface EventListResponse {
  total: number
  items: EventItem[]
}

export interface EventStats {
  total: number
  by_level: Record<RiskLevel, number>
  by_status: Record<EventStatus, number>
  handled_rate: number
  since: string
}

export interface TopIndex {
  key: IndexKey
  name: string
  value: number
}

export interface LivePerson {
  person_id: number
  track_id: number
  camera_id: string
  ts: number
  level: RiskLevel
  level_color: string
  reaction: string
  score: number
  rule_score: number
  model_score: number | null
  indices: IndexMap
  indices_peak: IndexMap
  top_indices: TopIndex[]
  triggers: Trigger[]
  action: string | null
  action_name: string | null
  action_conf: number | null
  predicted_next: IndexMap | null
  predicted_level: RiskLevel | null
  camera_path: CameraPathItem[]
  history: ScorePoint[]
  has_pose: boolean
}

export interface LiveSnapshot {
  task_id: string
  running: boolean
  started_at: number
  frames_processed: number
  alerts_emitted: number
  model_loaded: boolean
  persons: LivePerson[]
}

export interface TextSearchItem {
  event_id: number
  score: number
  reasons: string[]
  event: EventItem
}

export interface TextSearchResponse {
  query: string
  parsed: {
    indices: IndexKey[]
    colors: string[]
    levels: RiskLevel[]
    statuses: EventStatus[]
    cameras: string[]
    person_id: number | null
    since: string | null
    until: string | null
  }
  items: TextSearchItem[]
}

export type SocketMessage =
  | { type: 'hello'; user: string; ts: number }
  | { type: 'ping'; ts: number }
  | { type: 'alert'; event: EventItem }
  | { type: 'event_updated'; event: EventItem }
