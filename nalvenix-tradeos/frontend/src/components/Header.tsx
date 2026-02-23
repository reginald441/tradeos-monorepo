import { useAuthStore } from '../store/authStore'
import { Bell, Search } from 'lucide-react'

export default function Header() {
  const { user } = useAuthStore()

  return (
    <header className="h-16 glass-panel border-b border-white/5 flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search markets, strategies..."
            className="w-64 bg-slate-800/50 border border-white/10 rounded-lg pl-10 pr-4 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50"
          />
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button className="relative p-2 text-slate-400 hover:text-white transition-colors">
          <Bell className="w-5 h-5" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-cyan-500 rounded-full"></span>
        </button>
        
        <div className="flex items-center gap-3">
          <div className="text-right hidden sm:block">
            <p className="text-sm font-medium text-white">{user?.full_name}</p>
            <p className="text-xs text-cyan-400">Enterprise Plan</p>
          </div>
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
            <span className="text-sm font-bold text-white">
              {user?.full_name?.split(' ').map(n => n[0]).join('') || 'RK'}
            </span>
          </div>
        </div>
      </div>
    </header>
  )
}
