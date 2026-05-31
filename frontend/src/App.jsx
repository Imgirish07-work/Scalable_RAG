import { Navigate, Route, Routes } from 'react-router-dom'
import AppLayout from './components/templates/AppLayout'
import ChatPage from './pages/ChatPage'
import LibraryPage from './pages/LibraryPage'

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<ChatPage />} />
        <Route path="/library" element={<LibraryPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
