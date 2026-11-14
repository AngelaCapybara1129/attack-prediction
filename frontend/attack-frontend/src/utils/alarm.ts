import type { RiskLevel } from '../types/intent'

let ctx: AudioContext | null = null

/** L4 立即警告：三短音；L3 推播：一聲。以 WebAudio 產生，不需音檔。 */
export function playAlarm(level: RiskLevel): void {
  if (level !== 'L3' && level !== 'L4') return
  try {
    ctx = ctx ?? new AudioContext()
    if (ctx.state === 'suspended') void ctx.resume()
    const beeps = level === 'L4' ? 3 : 1
    const freq = level === 'L4' ? 880 : 660
    for (let i = 0; i < beeps; i += 1) {
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.type = 'square'
      osc.frequency.value = freq
      gain.gain.value = 0.06
      osc.connect(gain)
      gain.connect(ctx.destination)
      const t = ctx.currentTime + i * 0.32
      osc.start(t)
      osc.stop(t + 0.18)
    }
  } catch {
    // 瀏覽器可能在使用者互動前封鎖音訊
  }
}
