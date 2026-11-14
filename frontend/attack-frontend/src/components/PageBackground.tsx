import dashboardBg from '../assets/hlogo_al.png'
import workspaceOverlay from '../assets/work-space.svg'

/** 主畫面與事件紀錄共用的固定背景（工作區線稿 + 校徽 + 白色霧面） */
export default function PageBackground() {
  return (
    <div className="fixed inset-0 z-0 pointer-events-none">
      <img src={workspaceOverlay} alt="" className="absolute inset-0 h-full w-full object-cover opacity-25" />
      <img src={dashboardBg} alt="" className="absolute inset-0 h-full w-full object-cover opacity-18" />
      <div className="absolute inset-0 bg-white/78" />
    </div>
  )
}
