import type { ReactNode } from 'react'
import { INDEX_META, LEVEL_META, STATUS_META, formatClock, topIndices } from '../constants/risk'
import type { EventItem } from '../types/intent'
import RiskBadge from './RiskBadge'

type AlertCardProps = {
  event: EventItem
  onOpen: (id: number) => void
  extra?: ReactNode
}

export default function AlertCard({ event, onOpen, extra }: AlertCardProps) {
  const tops = topIndices(event.indices_peak ?? event.indices, 2)
  const status = STATUS_META[event.status]
  return (
    <button
      type="button"
      onClick={() => onOpen(event.id)}
      className={`mb-2 block w-full rounded-lg border-l-4 bg-white/60 p-3 text-left shadow-sm transition hover:bg-white/90 ${LEVEL_META[event.level].border}`}
      style={{ borderLeftColor: LEVEL_META[event.level].hex }}
    >
      <div className="mb-1.5 flex items-center gap-2">
        <RiskBadge level={event.level} />
        <span className="text-[11px] tabular-nums text-slate-500">{formatClock(event.occurred_at)}</span>
        <span className={`ml-auto rounded px-1.5 py-0.5 text-[10px] font-bold ${status.chip}`}>{status.label}</span>
      </div>
      <div className="text-xs font-bold text-slate-900">
        {event.camera_id} ・ 人物 P{event.person_id}
        <span className="ml-2 font-normal text-slate-500">R {event.score.toFixed(2)}</span>
      </div>
      {tops.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {tops.map((t) => (
            <span key={t.key} className="rounded bg-slate-900/5 px-1.5 py-0.5 text-[10px] text-slate-700">
              {t.key} {INDEX_META[t.key].name} {t.value.toFixed(2)}
            </span>
          ))}
        </div>
      )}
      {event.triggers && event.triggers.length > 0 && (
        <p className="mt-1 line-clamp-2 text-[11px] leading-4 text-slate-600">{event.triggers[0].description}</p>
      )}
      {extra}
    </button>
  )
}
