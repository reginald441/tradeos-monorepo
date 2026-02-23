import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './store/authStore'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Trading from './pages/Trading'
import Strategies from './pages/Strategies'
import Backtest from './pages/Backtest'
import Risk from './pages/Risk'
import Analytics from './pages/Analytics'
import Billing from './pages/Billing'
import Settings from './pages/Settings'
import Layout from './components/Layout'

function App() {
  const { isAuthenticated } = useAuthStore()

  if (!isAuthenticated) {
    return <Login />
  }

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/trading" element={<Trading />} />
        <Route path="/strategies" element={<Strategies />} />
        <Route path="/backtest" element={<Backtest />} />
        <Route path="/risk" element={<Risk />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/billing" element={<Billing />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}

export default App
