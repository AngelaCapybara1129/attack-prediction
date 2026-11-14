import type { ScorePoint } from '../types/intent'

type SparklineProps = {
  points: ScorePoint[]
  max?: number
  width?: number
  height?: number
  color?: string
}

/** 風險分數走勢（0~10），虛線為 L2/L3/L4 門檻 */
export default function Sparkline({ points, max = 10, width = 120, height = 28, color = '#f97316' }: SparklineProps) {
  if (points.length < 2) {
    return <div style={{ width, height }} className="rounded bg-slate-100" />
  }
  const t0 = points[0].ts
  const span = Math.max(1, points[points.length - 1].ts - t0)
  const xy = points.map((p) => {
    const x = ((p.ts - t0) / span) * width
    const y = height - (Math.min(max, Math.max(0, p.score)) / max) * height
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })
  const thresholds = [2.0, 3.5, 5.0]
  return (
    <svg width={width} height={height} className="overflow-visible rounded bg-slate-50" aria-label="風險分數走勢">
      {thresholds.map((t) => {
        const y = height - (t / max) * height
        return <line key={t} x1={0} x2={width} y1={y} y2={y} stroke="#cbd5e1" strokeDasharray="2 2" strokeWidth={1} />
      })}
      <polyline points={xy.join(' ')} fill="none" stroke={color} strokeWidth={1.5} />
    </svg>
  )
}
