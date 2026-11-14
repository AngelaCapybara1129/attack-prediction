import { LEVEL_META } from '../constants/risk'
import type { RiskLevel } from '../types/intent'

type RiskBadgeProps = {
  level: RiskLevel
  withLabel?: boolean
  size?: 'sm' | 'md'
}

export default function RiskBadge({ level, withLabel = false, size = 'sm' }: RiskBadgeProps) {
  const meta = LEVEL_META[level]
  const sizeCls = size === 'md' ? 'px-2 py-1 text-xs' : 'px-1.5 py-0.5 text-[10px]'
  return (
    <span className={`inline-flex items-center gap-1 rounded font-black ${sizeCls} ${meta.badge}`}>
      {level}
      {withLabel && <span className="font-bold opacity-90">{meta.reaction}</span>}
    </span>
  )
}
